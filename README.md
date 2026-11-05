# Finnie— Multi-Agent Voice-Enabled Financial Assistant

Finnie is a **multi-agent AI system** built with [LangGraph](https://langchain-ai.github.io/langgraph/) that powers an e-financial customer assistant. It combines financial question discovery (via RAG semantic search into a private vector db), market insight analysis (via API) and tax education (via RAG semantic search) and Voice IO - all orchestrated through a graph of specialised agents that can run in parallel, ask follow-up questions mid-conversation, and synthesize multi-source answers into a single coherent reply and displayed from a web interface (Streamlit).  It features a portfolio analysis, financial goal planning, and tax calculator functionality in the form of tool widgets from web interface. 

---

## What We're Building

### The Problem

E-financial support queries are diverse. A customer might ask "What's the latest price of MSFT?"_ (financial question search), _"How do stocks and bonds differ?"_ , "What is the expected interest rate going to cost me if I buy a new house in Texas?" or even multiple at once: _"Can I afford the down payment on half a million dollar house at the current Texas interest rate from selling my MSFT stocks?"_. A single monolithic chatbot struggles to handle all of these well.

### The Solution — A Multi-Agent Architecture

Finnie splits the work across **5 cooperating nodes** inside a LangGraph `StateGraph`:

```
 START → orchestrator  ─┬─ finance_agent ──→ synthesizer → END
                        |─ market_agent ──→ synthesizer → END
                        └─ tax_agent ──↗
```

1. **Orchestrator** — An LLM-powered classifier that reads the customer's message and decides which specialist(s) should handle it. It can dispatch to one agent or all in parallel.

2. **Finance Agent** — A subgraph with its own model ⇄ tools loop. It has access to a semantic search tool (`get_finance_answers`) backed by a vector store (RAG). It handles financial questions, recommendations, and general chitchat.

3. **Market Agent** — Another subgraph with a model ⇄ tools loop.  It calls another tools API to provide real-time stock quotes and financial data.

4. **Tax Agent** — Another subgraph with its own model ⇄ tools loop. It has access to a semantic search tool (`get_tax`) using traditional LLM question and answer loop. It handles tax related questions, and general chitchat.

5. **Synthesizer** — Merges results when multiple agents were invoked. For single-agent queries it passes the response through; for multi-agent queries it calls the LLM to weave both answers into one coherent reply.

### Key Capabilities

| Capability                  | How It Works                                                                                                                                                                                                        |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Parallel agent dispatch** | The orchestrator uses LangGraph's `Send()` to fan out to multiple agents simultaneously.                                                                                                        |
| **RAG product search**      | Financial questions and answers are embedded into a vector store; the finance agent performs similarity search to find relevant items.  Tax questions are handled by a traditional LLM query with state management. |
| **Market data**             | Real time market insights are pulled via another tool's API call (Yahoo real time data via yFinance).                                                                                                             |
| **Web interface**           | An interactive user experience with tools for Portfolio analysis, Planning and Stock ticker evaluation is delivered via widget tools implemented in Streamlit app pages.
| **Conversation memory**     | A `MemorySaver` checkpointer persists state across turns, so multi-turn conversations "just work".                                                                                                          |
| **Voice I/O**               | Microphone input + OpenAI TTS output for a fully spoken conversational experience.                                                                                                            |
| **Response synthesis**      | When multiple agents contribute, an LLM merges the results into a single, natural reply.                                                                                                                          
|

## Project Structure

```
main.py               # CLI entry point — text REPL, voice loop, or single query
|
financial_docs/       # 50+ individual financial documents loaded into a chroma_db vector database at startup
|
pages/
├── Market Data.py    # Streamlit app page implemented to pull most recent stock ticker data and graph metrics returned
├── Planning Goals.py # Streamlit app page implemented to pull last month of stock ticker data and graph metrics returned
├── Portfolio.py      # Streamlit app page implemented to pull multiple stock tickers configurable time and metrics returned
├── Taxes.py          # Streamlit app page implemented to calculate taxes based upon configurable entries
src/
├── config.py         # Logger, API keys, OpenAI/LangChain client setup
├── state.py          # LangGraph state definitions (FinAsstState, WorkerInput, etc.)
├── graph.py          # Builds & compiles the StateGraph with MemorySaver checkpointer
├── nodes.py          # All four graph nodes + agent subgraphs (model ⇄ tools loops)
├── tools.py          # Tool function implementations: finance search, fetch market data
├── rag.py            # Vector store setup for financial documentation
└── voice.py          # VoiceRecorder & VoiceSpeaker wrappers
```


## Prerequisites

- **Python 3.14+**
- **[uv](https://docs.astral.sh/uv/)** (recommended) or pip
- **OpenAI API key** with access to `gpt-4o` and `text-embedding-3-small`
- A microphone + speakers for voice mode

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/a4jthomas/ai_finance_assistant
cd ai_finance_assistant
```

### 2. Install dependencies

With `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install -r requirements.txt
```

 **Note for pip users:** Since you're not using `uv`, replace `uv run python` with just `python` in all the usage commands below. For example:
 ```bash
 # Text mode
 python -m src.main
 
 # Voice mode
 python -m src.main --voice
 ```
 
 It's also recommended to use a virtual environment:
 
 ```bash
 python -m venv .venv

 ``` 
 **Windows (PowerShell):**
  ```bash
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1
 ``` 
**Windows (CMD):**
 ```bash
 .venv\Scripts\activate.bat
 ``` 
**macOS/Linux:** 
 ```bash
source .venv/bin/activate
 ``` 
 **Tip**: Your prompt will change to show `(.venv)` when the environment is active.To deactivate at any time, run `deactivate`. 

**Core dependencies:**

| Package            | Purpose                                             |
| ------------------ | --------------------------------------------------- |
| `langgraph`        | Graph orchestration, checkpointing, HITL interrupts |
| `langchain-core`   | Message types, tool decorator                       |
| `langchain-openai` | ChatOpenAI LLM + OpenAI Embeddings                  |
| `openai`           | Raw client for TTS / Whisper (voice mode)           |
| `python-dotenv`    | `.env` file loading                                 |
| `pydantic`         | Structured output schemas                           |

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
env
OPENAI_API_KEY=sk-your-openai-api-key-here
LANGCHAIN_API_KEY=lsv2_your-langchain-api-key-here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-financial-assistant

```

## Usage

The system supports two modes of interaction, all launched from `main`.

### Text Mode (default) — Interactive REPL

```bash
run python -m streamlit run main.py
```

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.240:8501

LangSmith tracing enabled: True
📂 Loading PDFs from: ./financial_docs
13:22:09 | config             | INFO    | 📄 Loaded 199 pages/documents.
13:22:09 | config             | INFO    | ✂️ Chunking text...
13:22:09 | config             | INFO    | 🧩 Created 368 text chunks.
13:22:09 | config             | INFO    | 🧠 Generating embeddings and storing in ChromaDB...
13:22:09 | config             | INFO    | OpenAI clients initialised  (model: gpt-4o, embeddings: text-embedding-3-small)
13:22:14 | config             | INFO    | ✅ Successfully created vector database at ./chroma_db!
13:22:15 | graph              | INFO    | Graph compiled  (with MemorySaver for conversation persistence)
13:22:15 | main               | INFO    | Assistant ready  (voice=False)
13:22:15 | main               | INFO    | Assistant ready  (voice=False)

🛒  Financial Assistant  (type 'quit' to exit)
```


### Voice Mode

```bash
python -m main --voice
```

Requires a working microphone. The assistant speaks a welcome greeting, listens for your question (5-second recording windows), processes it through the full agent pipeline, and speaks the answer back via OpenAI TTS. Say "goodbye" to end the session.

---

## Conversation Examples

### General Financial Question Discovery

```
You: What's the difference between a stock and a bond?
 Agent says: Hello! I'm your Financial Assistant. Here's a concise overview of the differences between stocks and bonds:
```
Note*:  That the financial agent will answer this question via a vector database similarity search of pre encoded documents


### General Stock Question Discovery

```
You: What is the current stock price of Microsoft?
Assistant: The current stock price for Microsoft (MSFT) is approximately $509.01. If you need more detailed information or historical data, feel free to ask!
```

Note*:  That the market agent will be responsible for answering this question via the yFinance API

### General Tax Question Discovery

```
You: What is a capital gains tax?
Assistant: A capital gains tax is a tax on the profit realized from the sale of a non-inventory asset that was greater in value than.....
```


## Architecture Deep Dive

### Agent Subgraph Pattern

Each specialist agent is not a single function call — it's a **compiled subgraph** with its own internal loop:

```
START → model ─┬─ tools → model  (loop back if more tool calls)
               └─ END             (no tool calls → done)
```

This means the LLM can chain multiple tool calls in sequence (e.g., search for a product, then search again with refined terms) without any custom loop code. LangGraph manages the cycle natively.

### State Management

The `FinAsstState` TypedDict flows through the entire graph. Key fields:

| Field                | Type               | Purpose                                        |
| -------------------- | ------------------ | ---------------------------------------------- |
| `messages`           | `list[AnyMessage]` | Full conversation history (additive reducer)   |
| `user_query`         | `str`              | The current user message                       |
| `tasks`              | `list[AgentTask]`  | Routing decisions from the orchestrator        |
| `requires_synthesis` | `bool`             | Whether the synthesizer should merge results   |
| `agent_results`      | `list[dict]`       | Collected outputs from agents (custom reducer) |
| `final_answer`       | `str`              | The response returned to the user              |

The `agent_results_reducer` is notable: sending an empty list resets the field (clears stale results from previous turns), while a non-empty list appends.

### Checkpointing & Memory

The graph is compiled with `MemorySaver`, and every invocation uses a consistent `thread_id`. This means:

- Conversation history persists across turns within a session.
- The orchestrator sees prior context, so follow-up questions work naturally.

## Configuration Reference

### Environment Variables

| Variable         | Required | Description                                             |
| ---------------- | -------- | ------------------------------------------------------- |
| `OPENAI_API_KEY` | Yes      | OpenAI API key for GPT-4o, embeddings, TTS, and Whisper |

### Models Used

| Model                    | Usage                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| `gpt-4o`                 | Orchestrator classification, agent reasoning, response synthesis |
| `text-embedding-3-small` | Product catalog vector embeddings for RAG                        |
| `whisper-1`              | Voice transcription (voice mode only)                            |
| `tts-1`                  | Text-to-speech output (voice mode only)                          |

---



## Troubleshooting

| Issue                            | Fix                                                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY is missing`      | Copy `.env.example` to `.env` and add your key                                                       |
| `ModuleNotFoundError: src.rag`   | Create `src/rag.py` — see the Installation section above                                             |
| `ModuleNotFoundError: src.voice` | Create `src/voice.py` or run without `--voice`                                                       |
| Stale agent results across turns | The `agent_results_reducer` resets on empty list — this is handled automatically by the orchestrator |
| Voice mode: no audio             | Check microphone permissions and that `openai` package is installed                                  |

---
"# ai-financial-assistant" 
#   f i n n i e _ a i _ f i n a n c i a l _ a s s t  
 #   f i n n i e _ a i _ f i n a n c i a l _ a s s t  
 #   f i n n i e _ a i _ f i n a n c i a l _ a s s t  
 #   f i n n i e _ a i _ f i n a n c i a l _ a s s t  
 #   f i n n i e _ a i _ f i n a n c i a l _ a s s t  
 