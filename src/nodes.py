"""
Graph node functions.

Architecture:

  Main graph:
    START → orchestrator ─┬─ finance_agent ──→ synthesizer → END
                          |─ market_agent ──→ synthesizer → END
                          └─ tax_agent ──↗
  Each agent is internally a compiled subgraph with a model ⇄ tools loop:
    START → model ─┬─ tools → model (loop back)
                   └─ END    (no tool calls → done)

Key concepts:
  • Tool nodes: tools are separate graph nodes, not manual loops.
    LangGraph controls the model ⇄ tools cycle natively.
  • Parallel dispatch via Send()
  • Response synthesis for multi-agent queries
"""

from __future__ import annotations

import operator
import streamlit as st
import yfinance as yf

from typing import Annotated, Literal, TypedDict
from urllib import response

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

from src.config import get_logger, llm, fin_vector_db
from src.state import FinAsstState, ClassificationResult, WorkerInput
from src.tools import (
    get_finance_answers,
    get_tax_info,
    get_market_data,
)

logger = get_logger("nodes")


# ── Agent Prompts ────────────────────────────────────────────


FINANCE_PROMPT = f"""\
You are the professional finance agent for FinancialAssistant.
ROLE: Handle financial education enquiries and ensure you always provide accurate and helpful information with a professional tone.
ALWAYS provide citations for any financial information you provide, and if you cannot find a relevant citation, be honest and say so.
Final answers should be concise, clear, and should always include a disclaimer that the information provided is for educational purposes only and should not be considered financial advice.
If the customer asks for specific financial advice, politely decline and suggest they consult a licensed financial advisor.

TOOLS:
  get_finance_answers   – semantic search over our finance knowledge base (PDFs, articles, and other resources embedded into our Vector DB) to provide accurate and relevant information.

DOCUMENTS:
{fin_vector_db}

GUIDELINES:
-- Be empathetic and professional.
- After retrieving information, respond directly to the customer.
"""

MARKET_DATA_PROMPT = """\
You are a professional stock market agent for FinancialAssistant.

ROLE: Help customers find and learn about stock market information. A stock symbol is required for looking up specific market data.
If a stock symbol is not provided, ask the customer for one before calling the get_market_data tool.

TOOLS:
  get_market_data – retrieve current market data passing only the stock symbol

GUIDELINES: 
- For market questions, always ensure you have a stock symbol before calling the get_market_data tool.
- If market information is unavailable, suggest alternatives or direct the customer to additional resources.
- If the search returns stock market information the customer has already seen or that
  doesn't match what they asked for (wrong symbol, wrong year, etc.),
  be honest and say we don't currently have what they're looking for.
  Do NOT present irrelevant stock market information as if it matches the request.
- Keep responses concise and helpful.
- Final answers should always include a disclaimer that the information provided is for educational purposes only and should not be considered financial advice.
"""

TAX_PROMPT = """\
You are a professional tax advisor agent for FinancialAssistant.

ROLE: Help customers find and learn about tax-related information. 

TOOLS:
  get_tax_info – semantic search using open ai chat

GUIDELINES:
- Highlight key metrics, facts and savings.
- If tax information is unavailable, suggest alternatives or direct the customer to additional resources.
- If the search returns tax information the customer has already seen or that
  doesn't match what they asked for (wrong jurisdiction, wrong year, etc.),
  be honest and say we don't currently have what they're looking for.
  Do NOT present irrelevant tax information as if it matches the request.
- Keep responses concise and helpful.
- Final answers should always include a disclaimer that the information provided is for educational purposes only and should not be considered financial advice.
"""

# ── Tool bindings ────────────────────────────────────────────
finance_tools = [get_finance_answers]
finance_tools_by_name = {t.name: t for t in finance_tools}
finance_llm   = llm.bind_tools(finance_tools)

tax_tools = [get_tax_info]
tax_tools_by_name = {t.name: t for t in tax_tools} 
tax_llm       = llm.bind_tools(tax_tools)

market_analysis_tools = [get_market_data]
market_tools_by_name = {t.name: t for t in market_analysis_tools}
market_llm = llm.bind_tools(market_analysis_tools)


# ═══════════════════════════════════════════════════════════
#  Agent Subgraphs (model ⇄ tools)
# ═══════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """Minimal state for the model ⇄ tools subgraph loop."""
    messages: Annotated[list[AnyMessage], operator.add]


def should_continue(state: AgentState) -> str:
    """Route after model node: tool_calls → tools, otherwise → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

# ── Finance subgraph ─────────────────────────────────────────

def finance_model(state: AgentState) -> dict:
    """Call the finance LLM. If it asks for info without calling tools,
    use interrupt() to pause the graph and collect user input."""
    response = finance_llm.invoke(state["messages"])
    logger.info("[finance:model] tool_calls=%s", bool(response.tool_calls))

    # If no tool calls and no tools have been called yet,
    # the agent is asking for missing info — interrupt for HITL
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[finance:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[finance:model] HITL: user replied %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def finance_tools(state: AgentState) -> dict:
    """Execute tool calls from the finance LLM."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[finance:tools] %s(%s)", name, args)
        out = finance_tools_by_name[name].invoke(args) if name in finance_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


def finance_should_continue(state: AgentState) -> str:
    """Route after finance model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


sb = StateGraph(AgentState)
sb.add_node("model", finance_model)
sb.add_node("tools", finance_tools)
sb.add_edge(START, "model")
sb.add_conditional_edges("model", finance_should_continue)
sb.add_edge("tools", "model")
finance_subgraph = sb.compile()


# ── Tax subgraph ─────────────────────────────────────────

def tax_model(state: AgentState) -> dict:
    """Call the tax LLM. If it asks for info without calling tools,
    use interrupt() to pause the graph and collect user input."""
    response = tax_llm.invoke(state["messages"])
    logger.info("[tax:model] tool_calls=%s", bool(response.tool_calls))
    # If no tool calls and no tools have been called yet,
    # the agent is asking for missing info — interrupt for HITL
    if not response.tool_calls:
        any_tools_called = any(isinstance(m, ToolMessage) for m in state["messages"])
        if not any_tools_called:
            logger.info("[tax:model] HITL: interrupting to collect user info")
            user_reply = interrupt(response.content)
            logger.info("[tax:model] HITL: user replied %r", user_reply)
            return {"messages": [response, HumanMessage(content=str(user_reply))]}

    return {"messages": [response]}


def tax_tools(state: AgentState) -> dict:
    "Execute tool calls from the tax LLM."
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[tax:tools] %s(%s)", name, args)
        out = tax_tools_by_name[name].invoke(args) if name in tax_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))
    return {"messages": results}


def tax_should_continue(state: AgentState) -> str:
    """Route after tax model node. If the last message is a
    HumanMessage (user answered via HITL interrupt), loop back to model."""
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        return "model"
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


sb = StateGraph(AgentState)
sb.add_node("model", tax_model)
sb.add_node("tools", tax_tools)
sb.add_edge(START, "model")
sb.add_conditional_edges("model", tax_should_continue)
sb.add_edge("tools", "model")
tax_subgraph = sb.compile()

# ── Market subgraph ─────────────────────────────────────────

def market_model(state: AgentState) -> dict:
    "Call the yFinance API (with tools bound)."
    response = market_llm.invoke(state["messages"])
    logger.info("[market:model] tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}


def market_tools(state: AgentState) -> dict:
    "Execute tool calls from the yFinance API."
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        logger.info("[market:tools] %s(%s)", name, args)
        out = market_tools_by_name[name].invoke(args) if name in market_tools_by_name else f"Unknown tool: {name}"
        results.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

    return {"messages": results}


pb = StateGraph(AgentState)
pb.add_node("model", market_model)
pb.add_node("tools", market_tools)
pb.add_edge(START, "model")
pb.add_conditional_edges("model", should_continue)
pb.add_edge("tools", "model")
market_subgraph = pb.compile()



# ── Conversation context helper ──────────────────────────────

def build_context(messages: list[AnyMessage]) -> str:
    """Format prior conversation turns as text for agent context."""
    if not messages:
        return ""
    parts = []
    for m in messages:
        if isinstance(m, HumanMessage):
            parts.append(f"Customer: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
    if not parts:
        return ""
    return "CONVERSATION SO FAR:\n" + "\n".join(parts) + "\n\n"


# ═══════════════════════════════════════════════════════════
#  NODE 1 — Orchestrator
# ═══════════════════════════════════════════════════════════

def orchestrator_node(state: FinAsstState) -> Command[Literal["finance_agent", "tax_agent", "market_agent", "synthesizer"]]:
    """Classify the user query and dispatch to the right agent(s)."""
    user_query = state.get("user_query", "")
    if not user_query and state.get("messages"):
        user_query = state["messages"][-1].content

    logger.info("Orchestrator  query=%r", user_query)

    prompt = (
        f'Analyse this customer query and decide which agent(s) should handle it.\n\n'
        f'QUERY: "{user_query}"\n\n'
        'AGENTS:\n'
        '  finance_agent   – financial analysis, investment advice, portfolio management\n'
        '  market_data_agent – market research, stock ticker data\n'
        '  tax_agent       – tax questions, compliance, planning and optimization strategies\n'
        '  synthesizer     – combines insights from multiple agents\n\n'
        'RULES:\n'
        '1. Greetings, chitchat, general questions (hi, hello, thanks, how are you)\n'
        '   → finance_agent only\n'
        '2. Financial-only queries  → finance_agent only\n'
        '3. Market data requests → market_agent only\n'
        '4. Tax-related queries → tax_agent only\n'
        '5. Mixed queries         → ALL agents, requires_synthesis = true\n'
        '\nIMPORTANT: Only route to a specific agent when the query clearly involves their domain. When in doubt, use the synthesizer.\n'
    )

    classifier = llm.with_structured_output(ClassificationResult)
    try:
        classification = classifier.invoke(prompt)
    except Exception:
        logger.exception("Classification failed — defaulting to support_agent")
        classification = ClassificationResult(
            tasks=[], requires_synthesis=False,
            reasoning="Fallback: classification error",
        )

    logger.info("  routing=%s  synthesis=%s",
                [t.agent for t in classification.tasks],
                classification.requires_synthesis)

    targets: list[Send] = []
    for task in classification.tasks:
        targets.append(Send(task.agent, {
            "messages": state.get("messages", []),
            "user_query": user_query,
            "task_description": task.task_description,
        }))

    if not targets:
        targets = [Send("synthesizer", {})]

    return Command(
        update={
            "tasks": classification.tasks,
            "requires_synthesis": classification.requires_synthesis,
            "user_query": user_query,
            "agent_results": [],  # reset stale results from prior turns
        },
        goto=targets,
    )


# ═══════════════════════════════════════════════════════════
#  NODE 2 — Finance Agent
# ═══════════════════════════════════════════════════════════

def finance_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the finance agent via its model ⇄ tools subgraph."""
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Finance Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = finance_subgraph.invoke({"messages": [
        SystemMessage(content=FINANCE_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "finance", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════
#  NODE 3 — Tax Agent
# ═══════════════════════════════════════════════════════════

def tax_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the tax management agent via its model ⇄ tools subgraph.
    HITL is handled through conversation persistence: if the agent
    needs info (e.g. order ID), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Tax Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = tax_subgraph.invoke({"messages": [
        SystemMessage(content=TAX_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "tax", "response": answer}]},
        goto="synthesizer",
    )

# ═══════════════════════════════════════════════════════════
#  NODE 4 — Market Data Agent
# ═══════════════════════════════════════════════════════════

def market_agent(state: WorkerInput) -> Command[Literal["synthesizer"]]:
    """Run the market data agent via its model ⇄ tools subgraph.
    HITL is handled through conversation persistence: if the agent
    needs info (e.g. stock symbol or fund ticker), it responds with a question. The
    user's answer arrives on the next turn via the message history.
    """
    user_query = state.get("user_query", "")
    task_desc  = state.get("task_description", user_query)
    logger.info("Market Data Agent  task=%r", task_desc)

    context = build_context(state.get("messages", []))

    result = market_subgraph.invoke({"messages": [
        SystemMessage(content=MARKET_DATA_PROMPT),
        HumanMessage(content=f"{context}Task: {task_desc}\nCustomer query: {user_query}"),
    ]})

    answer = result["messages"][-1].content

    return Command(
        update={"agent_results": [{"source": "market_data", "response": answer}]},
        goto="synthesizer",
    )


# ═══════════════════════════════════════════════════════════
#  NODE 5 — Synthesizer
# ═══════════════════════════════════════════════════════════

def synthesizer_node(state: FinAsstState) -> dict:
    "Merge results from one or more agents into a single user-facing reply."
    results = state.get("agent_results", [])
    user_query = state.get("user_query", "")

    if not results:
        logger.warning("Synthesizer received no agent results")
        st.write("Sorry, I couldn't process that request. Please try again.")
        return {"final_answer": "Sorry, I couldn't process that request. Please try again."}

    if len(results) == 1:
        logger.info("Synthesizer  single-agent pass-through")
        st.write(f"{results[0]['response']}")
        return {"final_answer": results[0]["response"]}

    logger.info("Synthesizer  merging %d agent responses", len(results))
   
    parts = "\n\n".join(
        f"[{r['source'].upper()}]:\n{r['response']}" for r in results
    )
    prompt = (
        f"You are combining responses from multiple specialist agents.\n\n"
        f"CUSTOMER QUERY: {user_query}\n\n"
        f"AGENT RESPONSES:\n{parts}\n\n"
        "Write a single, coherent reply that addresses every part of the "
        "customer's query. Be concise. Speak as 'Financial Assistant'."
    )

    merged = llm.invoke(prompt)
    st.write(f"{merged.content}")
    return {"final_answer": merged.content}