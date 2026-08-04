# RAGVerse_AI — Production-Grade Technical RAG Framework

Enterprise Retrieval-Augmented Generation (RAG) system specialized in technical documentation (LangChain, LangGraph, LangSmith, Langfuse, and LangFlow). Built with modular state graphs, multi-query expansion, hybrid sparse/dense retrieval, cross-encoder re-ranking, and persistent multi-turn session checkpoints.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20%2F%20LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Vector Store](https://img.shields.io/badge/VectorStore-FAISS%20%2F%20Qdrant-green)](https://qdrant.tech/)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-blue)](https://www.postgresql.org/)

---

## 🚀 Features

- **Multi-Format Technical Ingestion:** Parses `.pdf`, `.md`, `.docx`, `.html`, and `.txt` files into clean, metadata-enriched chunks.
- **Advanced Retrieval Pipeline:** Combines Multi-Query LLM expansion, Dense Similarity Search, BM25 Keyword Hybrid Search, and Cross-Encoder Re-Ranking (`ms-marco-MiniLM-L-6-v2`).
- **Stateful Memory via LangGraph:** Persistent multi-turn conversation memory backed by PostgreSQL checkpointers.
- **Strict Grounding Rules:** System prompts engineered to prevent hallucinations and enforce source attribution.
- **Structured Application Logging:** Rotating log files (`logs/app.log`) and telemetry hooks for LangSmith/Langfuse observability.

---

## 🛠️ Environment Setup

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set the required environment variables:

```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_chatbot

# Embedding Configuration (Separate keys to avoid collisions)
GEMINI_EMBEDDING_MODEL_NAME=gemini-embedding-001
HUGGINGFACE_EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5

# Observability (LangSmith)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=RAGVerse_AI
```

---

## 📂 Data Ingestion

Place your raw documentation files (`.md`, `.pdf`, `.html`, `.docx`) inside `data/raw/documents/`, then run:

```bash
python -m app.retrieval.ingest
```

---

## 💻 Running the Chatbot

Start the conversational CLI:

```bash
python run.py
```

---

## 🧪 Running Unit Tests

Execute the automated test suite:

```bash
pytest
```
