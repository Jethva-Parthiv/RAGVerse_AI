# RAGVerse_AI

Enterprise-oriented conversational RAG framework built using modular graph-driven architectures.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Framework-LangChain%20%2F%20LangGraph-orange)](https://github.com/langchain-ai/langgraph)
[![Vector Store](https://img.shields.io/badge/VectorStore-FAISS-green)](https://github.com/facebookresearch/faiss)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-blue)](https://www.postgresql.org/)
[![Evaluation](https://img.shields.io/badge/Evaluation-RAGAS-red)](https://github.com/explodinggradients/ragas)

RAGVerse_AI is designed as a scalable Retrieval-Augmented Generation (RAG) system supporting persistent chat sessions, modular workflows, and advanced evaluation tracking. The current version features a robust CLI-based conversational chatbot implementation.

---

## 🚀 Features

- **Conversational RAG Pipeline:** Context-aware responses driven by advanced state graphs.
- **FAISS Vector Retrieval:** Optimized local vector store indexing for rapid context fetching.
- **PostgreSQL Checkpoint Memory:** Native LangGraph checkpointers to track multi-session histories persistently.
- **LangGraph Workflow Orchestration:** Clean separation of graph state nodes, allowing frictionless extension into multi-agent loops.
- **Production Evaluation Guardrails:** Native chunked evaluation pipeline backed by RAGAS to measure quality drift safely.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM Engine** | Google Gemini (`gemini-3.1-flash-lite`) |
| **Orchestration** | LangChain + LangGraph |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **State Persistence** | PostgreSQL |
| **Embeddings** | Gemini Native Embeddings |
| **Evaluation Metrics** | RAGAS (Retrieval Augmented Generation Assessment) |

---

# Project Architecture

```text
RAGVerse_AI/
│
├── app/
│   ├── core/
│   │   └── settings.py
│   │
│   ├── database/
│   │   └── postgres.py
│   │
│   ├── graph/
│   │   ├── builder.py
│   │   ├── state.py
│   │   └── nodes/
│   │       └── chat_node.py
│   │
│   ├── llm/
│   │   ├── models.py
│   │   └── prompts.py
│   │
│   ├── retrieval/
│   │   └── retriever.py
│   │
│   └── services/
│       └── chat_service.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── run.py
├── requirements.txt
├── docker-compose.yml
├── .env.example
└── README.md
```

---

# Environment Variables

Copy:

```bash
.env.example -> .env
```

Required variables:

```env
GOOGLE_API_KEY=your_google_api_key
DATABASE_URL=postgresql://postgres:password@localhost:5432/rag_chatbot
```

Optional configuration:

```env
CHAT_MODEL_NAME=gemini-3.1-flash-lite
EMBEDDING_MODEL_NAME=gemini-embedding-001

TOP_K_RESULTS=5

CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

Optional LangSmith tracing:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=RAGVerse_AI
```

---

# Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the chatbot:

```bash
python run.py
```

A unique `thread_id` is generated for each session and persisted using PostgreSQL checkpointing.

Type:

```text
exit
```

or

```text
bye
```

to end the session.

---

# Vector Store

The retriever loads the FAISS index from:

```text
data/processed/faiss_index
```

Ensure:

- the FAISS index exists
- embeddings are compatible with the configured embedding model

---

# PostgreSQL Memory

RAGVerse_AI uses LangGraph checkpointing with PostgreSQL to persist:

- conversation history
- graph state
- thread-based sessions

This enables:

- multi-turn memory
- persistent conversations
- resumable workflows

---

# Running Tests

```bash
pytest
```

---

# Current Version

Current implementation includes:

- single-node conversational RAG workflow
- retrieval + generation pipeline
- persistent memory
- CLI interaction layer

---

# Planned Improvements

Future roadmap includes:

- Hybrid search (BM25 + vector search)
- Streaming responses
- Multi-agent workflows
- Tool calling
- Document upload pipelines
- Web interface
- FastAPI backend
- Redis caching
- Async graph execution
- Observability & monitoring
- Reranking pipelines
- Production deployment
- Authentication & multi-user support

---

# Troubleshooting

## Database Connection Issues

Verify:

- PostgreSQL is running
- `DATABASE_URL` is correct
- credentials are valid

## FAISS Loading Errors

Ensure:

- `data/processed/faiss_index/` exists
- embedding models match the generated index

## Gemini Authentication Errors

Verify:

- `GOOGLE_API_KEY` is valid
- API quota is available

---

# Why RAGVerse_AI?

This project is designed to feel like:

- a serious AI engineering project
- a scalable architecture
- a long-term platform
- a portfolio-worthy repository

---

# RAG Evaluation (RAGAS) — New Feature

This repo includes an optional evaluation pipeline using **RAGAS**.

## What it evaluates
It runs the current RAG pipeline (retrieval + Gemini generation) against a HotpotQA **validation** dataset sample and computes:
- faithfulness
- answer_relevancy
- context_precision
- context_recall

## Files
- `app/evaluation/ragas_eval.py` — entry point (load QA pairs → run pipeline → score → save results)
- `app/evaluation/pipeline_runner.py` — generates answers for each question using current retriever + LLM
- `app/evaluation/ragas_scorer.py` — computes RAGAS metrics (with checkpoint/resume)
- `app/evaluation/report_generator.py` — writes a human-readable report
- `app/evaluation/compare_scores.py` — compares multiple runs from `data/eval/baseline_summary.csv`

## Outputs
Stored under:
- `data/eval/baseline_results.csv` (per-question)
- `data/eval/baseline_summary.csv` (append per run)
- `data/eval/baseline_report.txt` (human report)
- (intermediate) `data/eval/ragas_eval_progress.csv` (checkpoint)

## Run evaluation
From repo root:

```bash
python -m app.evaluation.ragas_eval
```

## Optional environment variables
- `EVAL_NUM_PAIRS` (default: `100`) — number of sampled QA pairs
- `EVAL_SEED` (default: `42`) — sampling seed
- `EVAL_SLEEP` (default: `2.0`) — delay between questions (helps avoid rate limits)

> If validation data is missing, the loader downloads HotpotQA validation via `app/retrieval/ingest_hotpotqa.py`.

---

# License

MIT License
