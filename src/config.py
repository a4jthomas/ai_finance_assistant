"""
Configuration: Logger, API clients, and settings.

This module is imported by everything else, so it must have zero
dependencies on other project modules.
"""

import logging
import os
import sys

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


from dotenv import load_dotenv
from openai import OpenAI

# ── Load .env ────────────────────────────────────────────────
load_dotenv()

# ── Logger ───────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a module-level logger with a readable format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("config")

# ── Validate API key ────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
    logger.error("OPENAI_API_KEY is missing. Copy .env.example → .env and add your key.")
    sys.exit(1)

# ── Clients ──────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# - Enable LangSmith tracing ---
os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
print(f"LangSmith tracing enabled: {tracing_enabled}")

# ── RAG Vector Store ───────────────────────────────────────
"""
RAG: Build a ChromaDB vector store from a directory of PDF documents.

The vector store is created once at config time and re-used by the
get_finance_answers tool.
"""

"""
Loads PDFs from a directory, chunks them, generates embeddings, 
and saves them into a local Chroma vector database.
"""
print(f"📂 Loading PDFs from: ./financial_docs")

# Step 1: Load all PDF financial documents from the directory
loader = PyPDFDirectoryLoader("./financial_docs")
raw_documents = loader.load()
logger.info(f"📄 Loaded {len(raw_documents)} pages/documents.")

logger.info("✂️ Chunking text...")

# Step 2: Split text into optimal chunks for financial data
# We use a smaller chunk size and overlap to keep financial numbers/context tight
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
chunks = text_splitter.split_documents(raw_documents)
logger.info(f"🧩 Created {len(chunks)} text chunks.")

logger.info("🧠 Generating embeddings and storing in ChromaDB...")

# Step 3 & 4: Initialize Embedding Model and Vector Store

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
logger.info("OpenAI clients initialised  (model: gpt-4o, embeddings: text-embedding-3-small)")

# This automatically embeds the chunks and stores them locally
fin_vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

logger.info(f"✅ Successfully created vector database at ./chroma_db!")

   

     
 