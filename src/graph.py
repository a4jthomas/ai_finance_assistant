"""
Build and compile the LangGraph StateGraph.

Graph topology:

  START → orchestrator ─┬─ finance_agent ──→ synthesizer → END
                        |─ market_agent ──→ synthesizer → END
                        └─ tax_agent ──↗

Each agent is internally a subgraph with a model ⇄ tools loop.
The MemorySaver checkpointer persists conversation history across
turns, enabling multi-turn HITL (agent asks a question on one turn,
user answers on the next).
"""

import streamlit as st

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import get_logger
from src.nodes import orchestrator_node, finance_agent, tax_agent, market_agent, synthesizer_node
"""#REMOVE THIS LINE AND add as new agents are coded
#from src.nodes import planning_agent, portfolio_agent,
"""

from src.state import FinAsstState

logger = get_logger("graph")


def build_graph() -> StateGraph:
    """Create, wire, and compile the FinAsst multi-agent graph."""

    builder = StateGraph(FinAsstState)

    # ── Add nodes ────────────────────────────────────────
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("finance_agent", finance_agent)
    builder.add_node("tax_agent", tax_agent)
    builder.add_node("market_agent", market_agent)

    """#REMOVE THIS LINE AND add as new agents are coded
   # builder.add_node("planning_agent", planning_agent)
   # builder.add_node("portfolio_agent", portfolio_agent)
    """
    builder.add_node("synthesizer", synthesizer_node)

    # ── Add edges ────────────────────────────────────────
    builder.add_edge(START, "orchestrator")
    builder.add_edge("synthesizer", END)

    # ── Compile with checkpointer ────────────────────────
    # MemorySaver persists graph state so that interrupt()-based
    # HITL can pause and resume, and conversation history carries
    # forward between queries.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    logger.info("Graph compiled  (with MemorySaver for conversation persistence)")
    
    return graph


# Module-level singleton
FinAsst_graph = build_graph()

