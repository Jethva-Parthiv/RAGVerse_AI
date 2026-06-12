# RAGVerse_AI Architecture

## 1. Project Overview

- **Purpose**: RAGVerse_AI is an enterprise-oriented conversational Retrieval-Augmented Generation (RAG) framework.
- **Main problem it solves**: Provide context-grounded chat responses over a local knowledge base using retrieval + LLM generation, while persisting multi-turn conversations.
- **High-level description**: 
  - A **LangGraph** workflow stores conversation state per `thread_id` using a **PostgreSQL checkpoint**.
  - A **chat node** performs advanced retrieval (multi-query expansion, dense retrieval from **FAISS**, hybrid BM25+dense merge, and cross-encoder re-ranking) and then calls a **Gemini** LLM with a context-grounding prompt.
  - An optional **evaluation pipeline** runs the current RAG system on HotpotQA validation samples and scores quality with **RAGAS**, persisting progress in CSV.

## 2. Technology Stack

| Category | Technology / Library | Used in code |
|---|---|---|
| Language | Python | Project |
| Orchestration | **LangChain** + **LangGraph** | `app/graph/*`, `app/services/*` |
| LLM provider | **Google Gemini** via `langchain_google_genai` | `app/llm/models.py`, `app/retrieval/advanced_retriever.py`, evaluation |
| Chat orchestration | `ChatPromptTemplate`, `MessagesPlaceholder` | `app/graph/nodes/chat_node.py`, `app/evaluation/pipeline_runner.py` |
| Embeddings | **HuggingFaceEmbeddings** (`langchain_huggingface`) + `BAAI/bge-base-en-v1.5` | `app/llm/embeddings.py`, `app/retrieval/retriever.py`, `advanced_retriever.py` |
| Vector store | **FAISS** (`langchain_community.vectorstores`) | `app/retrieval/retriever.py`, `app/retrieval/vectorstore.py` (if present), `app/retrieval/advanced_retriever.py`, `app/retrieval/ingest_hotpotqa.py` |
| Reranker model | `sentence-transformers` cross-encoder | `app/retrieval/advanced_retriever.py` |
| BM25 | `rank_bm25` | `app/retrieval/advanced_retriever.py` |
| Database | **PostgreSQL** (via `psycopg`) | `app/database/postgres.py` |
| Persistent memory | `langgraph.checkpoint.postgres.PostgresSaver` | `app/database/postgres.py` |
| Evaluation | **RAGAS** | `app/evaluation/*` |
| Dataset ingestion | `datasets` (HF datasets) | `app/retrieval/ingest_hotpotqa.py` |
| Prompting rules | Project-defined templates | `app/llm/prompts.py` |
| CLI UI | `rich` | `run.py` |

Notable declared dependencies in `RAGVerse_AI/requirements.txt`:
- `langchain_community`, `langchain_core`, `langchain_google_genai`, `langchain_text_splitters`, `langgraph`
- `psycopg`, `python-dotenv`, `rich`

## 3. System Architecture

### Overall architecture (component interaction)

```text
User (CLI)
  |
  | query + thread_id
  v
RAG Chat Entry (run.py)
  |
  | get_chat_response(query, thread_id)
  v
Chat Service (app/services/chat_service.py)
  |
  | invokes LangGraph workflow with persistent checkpointer
  v
LangGraph Workflow (app/graph/builder.py)
  |
  v
chat_node (app/graph/nodes/chat_node.py)
  |
  | 1) advanced retrieval
  |    AdvancedRetriever.retrieve(query)
  |    - multi-query expansion (Gemini)
  |    - dense retrieval (FAISS similarity)
  |    - hybrid merge (BM25 + dense)
  |    - cross-encoder re-ranking
  |
  | 2) context assembly
  |
  | 3) LLM generation (Gemini)
  v
Response returned to CLI
```

### Persistence & state
- The workflow state is typed as `State` (messages + query) in `app/graph/state.py`.
- A PostgreSQL-based **LangGraph checkpointer** is created once in `app/graph/builder.py` using `app/database/postgres.py`.
- The `thread_id` is passed as `configurable.thread_id` into `chain.invoke(...)`.

## 4. Project Structure

Key directories and responsibilities:

### `RAGVerse_AI/`
- **`README.md`**: project overview, environment variables, feature list, evaluation usage.
- **`run.py`**: CLI entry point that repeatedly reads user input and prints model responses.
- **`requirements.txt`**: Python dependencies.
- **`docker-compose.yml`**: likely for PostgreSQL (referenced by README).
- **`.env.example`**: environment variable template.

### `app/`

#### `app/main.py` / `run.py`
- `app/main.py` is a thin entrypoint used by `run.py`.

#### `app/api/`
- **`app/api/routes.py`**: API routing (FastAPI routes). (Not fully inspected in provided context.)

#### `app/cli/`
- **`app/cli/chat_cli.py`**: CLI utilities (not inspected in provided context).

#### `app/core/`
- **`core/settings.py`**: central configuration (model names, FAISS path, chunking params, database URL).
- **`core/config.py`**, **`core/logging.py`**: not inspected here, but likely support runtime configuration and logging.

#### `app/database/`
- **`database/postgres.py`**: creates a `PostgresSaver` checkpointer from `DATABASE_URL`.

#### `app/graph/`
- **`graph/builder.py`**: builds and compiles a LangGraph `StateGraph` with nodes and edges.
- **`graph/state.py`**: defines the workflow state type: `messages` (annotated for message accumulation) and `query`.
- **`graph/nodes/chat_node.py`**: retrieval + prompt + LLM generation for each turn.

#### `app/llm/`
- **`llm/models.py`**: Gemini chat model creation.
- **`llm/embeddings.py`**: Gemini embedding model + HuggingFace embedding model.
- **`llm/prompts.py`**: RAG grounding rules/templates.

#### `app/retrieval/`
- **`retrieval/ingest_hotpotqa.py`**: downloads HotpotQA validation/train, extracts wiki paragraphs, chunks, embeds, and builds/saves FAISS.
- **`retrieval/retriever.py`**: a simple FAISS retriever loader (not used directly by the chat node, which uses advanced retriever).
- **`retrieval/advanced_retriever.py`**: multi-query + hybrid BM25+dense + cross-encoder re-ranking.
- **`retrieval/vectorstore.py`**: not inspected in provided context.

#### `app/services/`
- **`services/chat_service.py`**: wraps the LangGraph invocation; converts state into a final string via `StrOutputParser`.

#### `app/evaluation/`
- **`evaluation/ragas_eval.py`**: evaluation entry point.
- **`evaluation/dataset_loader.py`**: loads HotpotQA validation and samples QA pairs.
- **`evaluation/pipeline_runner.py`**: runs retrieval + LLM generation per question and stores answers/contexts.
- **`evaluation/ragas_scorer.py`**: computes RAGAS metrics in batches and writes checkpoint progress CSV.
- **`evaluation/report_generator.py`**: formats a human-readable report (worst cases and recommendations).
- **`evaluation/compare_scores.py`**: compares baseline summary rows.
- **`evaluation/config.py`**: evaluation constants (batch size, checkpoint path, default number of pairs).

#### `app/utils/`
- formatting/validators/helpers (not inspected in provided context).

### `data/`
- **`data/raw/`**: raw HotpotQA datasets (e.g., `hotpotqa/train.json`, `hotpotqa/validation.json`).
- **`data/processed/`**:
  - `faiss_index/`: FAISS production index directory.
  - ingestion checkpoint directory: `data/processed/faiss_checkpoint/`.
- **`data/eval/`**: evaluation CSV outputs + report.

### `tests/`
- Unit tests for graph, prompts, and retriever (filenames listed, contents not inspected here).

## 5. RAG Pipeline Flow

### Chat-time pipeline (per user turn)

1. **CLI creates session**
   - `run.py` generates `thread_id = chat-{uuid4}`.

2. **Build initial LangGraph state** (`app/services/chat_service.py`)
   - `init_state = {"messages": [HumanMessage(content=query)], "query": query}`
   - `config = {"configurable": {"thread_id": thread_id}}`

3. **LangGraph workflow invocation** (`app/services/chat_service.py` → `app/graph/builder.py`)
   - Compiled graph contains a single node:
     - `START -> chat_node -> END`
   - Uses PostgreSQL checkpointer.

4. **Node: advanced retrieval + generation** (`app/graph/nodes/chat_node.py`)
   1) **History extraction**: `history = state["messages"][:-1]`
   2) **Advanced retrieval**: `docs = retriever.retrieve(query)` where `retriever` is from `get_advanced_retriever()`.
   3) **Context assembly**: `context = "\n\n".join(doc.page_content for doc in docs)`
   4) **Prompt + LLM call**:
      - A `ChatPromptTemplate` is created with:
        - system: `BASE_RAG_RULES_HOTPOTQA_DATASET`
        - history placeholder: `MessagesPlaceholder(variable_name="history")`
        - human message: `Context:\n{context}\n\nQuestion: {query}`
      - `chain.invoke({"history": history, "context": context, "query": query})`
   5) **Return**: `{"messages": [AIMessage(content=response.content)]}`

## 6. Advanced Retrieval Components

The advanced retrieval logic is implemented in `app/retrieval/advanced_retriever.py`.

### 6.1 Multi-query expansion (Gemini)
- **Why it exists**: Increase recall by generating alternative query formulations capturing different angles/synonyms.
- **Where**: `expand_queries(question, n=3)` and `MULTI_QUERY_PROMPT`.
- **How it interacts**: The expanded list of queries feeds dense retrieval and BM25 hybrid merging.

### 6.2 Dense retrieval (FAISS similarity)
- **Why it exists**: Retrieve semantically similar passages even when lexical overlap is low.
- **Where**:
  - FAISS loading: `_load_faiss()` (cached with `@lru_cache(maxsize=1)`)
  - Retrieval: `dense_retrieve(queries, k=DENSE_TOP_K)`
- **How it interacts**: Runs for each query variant, then deduplicates candidates by a prefix key of `doc.page_content[:200]`.

### 6.3 Hybrid BM25 + dense merge
- **Why it exists**: Improve robustness—BM25 captures keyword matches; dense captures semantic matches.
- **Where**:
  - `BM25Index` builds tokenized BM25 scores from candidate docs.
  - `hybrid_merge(dense_docs, queries, k=...)` combines dense rank scores and BM25 normalized scores.
- **How it interacts**:
  - BM25 scores are computed across query variants, normalized, then merged with dense positional rank using:
    - `DENSE_WEIGHT = 0.7`
    - `BM25_WEIGHT = 0.3`

### 6.4 Cross-encoder re-ranking
- **Why it exists**: Increase precision by directly scoring (question, passage) relevance pairs.
- **Where**:
  - `_load_cross_encoder()` loads `sentence-transformers` CrossEncoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `rerank(question, docs, top_n=RERANK_TOP_N)` sorts by cross-encoder scores.
- **How it interacts**: Takes merged candidates and returns the final list passed into the LLM context assembly.

## 7. API Layer

Current inspected code path is **CLI-first**. An `app/api/routes.py` exists, but endpoints were not fully inspected in the provided context.

### Known request flow (CLI)
- `run.py` reads user input
- `app/services/chat_service.py` invokes the compiled LangGraph and returns a string via `StrOutputParser`

## 8. Data Flow

### Sequence: user query → final answer
1. **Input**: CLI calls `get_chat_response(query, thread_id)`.
2. **Initial state**: `init_state = {"messages": [HumanMessage(content=query)], "query": query}`.
3. **Graph invocation**: `chain.invoke(init_state, config={"configurable": {"thread_id": thread_id}})`.
4. **StateGraph execution**: `START → chat_node → END`.
5. **In chat_node**:
   - `history = messages[:-1]`
   - `docs = retriever.retrieve(query)`
   - `context = "\n\n".join(doc.page_content ...)`
6. **LLM call**:
   - Prompt includes: system rules + chat history + retrieved context + current question.
7. **Output**:
   - `AIMessage(content=response.content)` returned to the chain.
   - `services/chat_service.py` extracts the message via `RunnableLambda` and returns string.

### Persistence and memory
- LangGraph checkpointer is configured in `app/graph/builder.py` using `app/database/postgres.py`.
- `thread_id` passed at invocation time enables per-session conversation persistence.

## 9. Configuration & Environment

### Environment variables
Loaded via `dotenv` and `os.getenv` in `app/core/settings.py` and used across modules.

- **Model settings**:
  - `CHAT_MODEL_NAME` (default: `gemini-3.1-flash-lite`)
  - `EMBEDDING_MODEL_NAME` (default: `gemini-embedding-001` for Gemini embeddings)
  - `HUGGINGFACE_EMBEDDING_MODEL_NAME` is also sourced from `EMBEDDING_MODEL_NAME` (see `app/core/settings.py`).

- **RAG retrieval settings**:
  - `TOP_K_RESULTS` (default: `5`)
  - `CHUNK_SIZE` (default: `1000`)
  - `CHUNK_OVERLAP` (default: `200`)

- **Storage**:
  - `FAISS_PATH` derived from `data/processed/faiss_index`
  - `DATABASE_URL` required for PostgreSQL checkpointer.

### Vector store / dataset settings
- `data/processed/faiss_index/` must exist for chat-time retrieval.

### Evaluation settings (RAGAS)
Defined in `app/evaluation/config.py`:
- `EVAL_NUM_PAIRS` (default: 100)
- `EVAL_SEED` (default: 42)
- `EVAL_SLEEP` (default: 2.0)
- `CHECKPOINT_FILE`: `data/eval/ragas_eval_progress.csv`

## 10. Evaluation Framework

### Entry point
- `app/evaluation/ragas_eval.py`

### Dataset loading
- `app/evaluation/dataset_loader.py`:
  - uses HotpotQA `data/raw/hotpotqa/validation.json`
  - if missing, imports `app/retrieval/ingest_hotpotqa.py` to download
  - samples `NUM_PAIRS` with `RANDOM_SEED`
  - returns list of `{question, ground_truth}`

### RAG execution for evaluation
- `app/evaluation/pipeline_runner.py`:
  - builds a small prompt using `BASE_RAG_RULES_HOTPOTQA_DATASET`
  - for each QA pair:
    - `docs = retriever.invoke(question)`
    - `context = "\n\n".join(doc.page_content ...)`
    - `answer = chain.invoke({"context": context, "query": question})`
    - ensures non-empty answers (`"I don't know."` if needed)
  - stores per-question output: `{question, ground_truth, answer, contexts}`

### RAGAS scoring
- `app/evaluation/ragas_scorer.py`:
  - wraps Gemini LLM and embeddings using RAGAS wrappers:
    - `LangchainLLMWrapper(get_gemini_chat_model())`
    - `LangchainEmbeddingsWrapper(get_gemini_embedding_model())`
  - metrics used:
    - `faithfulness`
    - `answer_relevancy`
    - `context_precision`
    - `context_recall`
  - evaluates in micro-batches (`BATCH_SIZE = 10`) and writes to `CHECKPOINT_FILE`.
  - supports resume by appending and using checkpoint row count as `start_idx`.
  - deletes checkpoint at the end.

### Reporting
- `app/evaluation/report_generator.py`:
  - writes:
    - `data/eval/baseline_results.csv` (per question; overwritten per run)
    - `data/eval/baseline_summary.csv` (append one row per run)
    - `data/eval/baseline_report.txt` (human-readable; worst-faithfulness and worst-recall excerpts + recommendations)

## 11. External Integrations

### LLM providers
- **Google Gemini** via `langchain_google_genai.ChatGoogleGenerativeAI` (`app/llm/models.py`).

### Embedding providers
- **Gemini embeddings**: `GoogleGenerativeAIEmbeddings` (`app/llm/embeddings.py`).
- **HuggingFace embeddings**: `HuggingFaceEmbeddings` using `BAAI/bge-base-en-v1.5` by default (`app/llm/embeddings.py`).

### Vector databases
- **FAISS** via `langchain_community.vectorstores.FAISS`.

### Third-party ML models
- **Cross-encoder re-ranker**: `sentence-transformers` model `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **BM25**: `rank_bm25.BM25Okapi`.

### Database
- **PostgreSQL** used for LangGraph checkpointer (`psycopg` + `langgraph.checkpoint.postgres.PostgresSaver`).

## 12. Design Decisions

1. **Graph-driven orchestration (LangGraph)**
   - Enables persistent, thread-based conversation state.
   - Implemented in `app/graph/builder.py` with a single `chat_node` now, but designed for multi-node expansion.

2. **Advanced retrieval instead of simple top-k**
   - `app/retrieval/advanced_retriever.py` combines multi-query expansion, hybrid BM25/dense merging, and cross-encoder re-ranking.
   - This improves both recall and precision before LLM context assembly.

3. **Local FAISS for low-latency retrieval**
   - `app/retrieval/ingest_hotpotqa.py` builds and saves a FAISS index to `data/processed/faiss_index`.
   - Chat-time retrieval loads the FAISS index via `FAISS.load_local(...)`.

4. **Evaluation built into the repository using RAGAS**
   - `app/evaluation/*` allows quality measurement and performance tracking across runs.
   - Checkpointed scoring supports long runs / rate limits.

## 13. Current Limitations

1. **Reranking & hybrid BM25 build scope**
   - BM25 index is rebuilt (lazy) from the candidate set inside `advanced_retriever` using in-memory structures.
   - Dense retrieval deduping uses a heuristic key `doc.page_content[:200]`, which may collapse distinct passages.

2. **LLM prompting is context-only**
   - The chat prompt instructs grounding rules, but no explicit structured citations or span extraction is implemented.

3. **API layer not fully documented**
   - `app/api/routes.py` exists, but endpoints weren’t inspected in detail during the document build.

4. **Potential embedding config coupling**
   - In `app/core/settings.py`, both Gemini and HuggingFace embedding model names are sourced from `EMBEDDING_MODEL_NAME` (default `gemini-embedding-001`). This can be a configuration footgun unless intentional.

## 14. Future Extension Points

Based on the existing modular structure:

1. **Graph expansion**
   - Add additional nodes in `app/graph/builder.py` (e.g., tool calling, routing, summarization nodes).

2. **Retriever variants**
   - Extend `AdvancedRetriever` in `app/retrieval/advanced_retriever.py`:
     - add MMR
     - add metadata filtering
     - swap reranker model

3. **Streaming and async**
   - Currently chat is synchronous; the LangGraph node can be adapted for streaming if the LLM wrapper supports it.

4. **API hardening / auth**
   - If `app/api/routes.py` is used, introduce authentication and per-user thread mapping.

5. **Observability**
   - Add tracing hooks (LangSmith env vars are documented in README).

## 15. Architecture Summary

RAGVerse_AI is a modular conversational RAG system built around a LangGraph workflow with PostgreSQL checkpointing. Each user turn runs `chat_node`, which uses a composite retrieval strategy (multi-query expansion → FAISS dense retrieval → hybrid BM25+dense merge → cross-encoder reranking) to assemble a grounded context, then calls a Gemini chat model using strict “answer from context only” instructions. The repository also includes a HotpotQA + RAGAS evaluation pipeline with checkpointed batch scoring and report generation.

