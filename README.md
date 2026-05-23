# RAGVerse_AI

Enterprise-oriented conversational RAG framework built using:

- LangGraph
- Gemini
- FAISS
- PostgreSQL Memory
- LangChain

RAGVerse_AI is designed as a scalable Retrieval-Augmented Generation (RAG) system that supports:

- conversational memory
- persistent chat sessions
- modular graph workflows
- retrieval pipelines
- context-aware responses
- future agentic workflow expansion

The current version includes a CLI-based conversational RAG chatbot implementation.

---

# Features

- Conversational RAG pipeline
- FAISS vector retrieval
- PostgreSQL checkpoint-based memory
- LangGraph workflow orchestration
- Gemini-powered response generation
- Modular and scalable architecture
- Environment-based configuration
- Persistent multi-session conversations

---

# Tech Stack

| Component | Technology |
|---|---|
| LLM | Gemini |
| Framework | LangChain + LangGraph |
| Vector Store | FAISS |
| Database | PostgreSQL |
| Embeddings | Gemini Embeddings |
| Language | Python |

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

# License

MIT License