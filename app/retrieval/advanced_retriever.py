"""
Advanced Retriever for RAGVerse_AI
---------------------------------------------
Combines three advanced techniques:
    1. Multi-query expansion  → better recall 
    2. Cross-encoder re-ranking → better precision 
    3. Hybrid BM25 + dense    → catches keyword matches dense misses
"""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import settings
from app.core.logging import logger
from app.llm.embeddings import get_huggingface_embedding_model
from app.llm.models import get_gemini_chat_model

# ── Config ────────────────────────────────────────────────────────────────────

DENSE_TOP_K       = int(settings.rag.top_k_results * 4)   # fetch more candidates, re-rank later
MULTI_QUERY_COUNT = 3                                     # number of query variants
RERANK_TOP_N      = settings.rag.top_k_results            # final docs passed to LLM
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_WEIGHT       = 0.3   # weight for BM25 scores in hybrid merge
DENSE_WEIGHT      = 0.7   # weight for dense scores in hybrid merge


# ── Helper for doc key ────────────────────────────────────────────────────────

def _doc_key(doc: Document) -> str:
    """Generates a stable SHA-256 hash key for document deduplication."""
    return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()


# ── 1. Multi-query expansion ──────────────────────────────────────────────────

MULTI_QUERY_PROMPT = """\
You are an AI assistant helping improve document retrieval for technical documentation.
Given the user question below, generate {n} different versions of it.
Each version should rephrase the question to capture different angles,
synonyms, or sub-questions that would help find relevant documentation.

Output ONLY the {n} questions, one per line. No numbering, no explanations.

Original question: {question}
"""


def expand_queries(question: str, n: int = MULTI_QUERY_COUNT) -> list[str]:
    """
    Generates n query variants for the original question.
    Always includes the original question in the returned list.
    """
    from langchain_core.output_parsers import StrOutputParser
    parser = StrOutputParser()
    llm = get_gemini_chat_model()
    prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)

    try:
        chain = llm | parser
        response = chain.invoke(prompt)
        raw = response.content if hasattr(response, "content") else str(response)
        variants = [line.strip() for line in raw.strip().splitlines()
                    if line.strip() and len(line.strip()) > 10][:n]
    except Exception as e:
        logger.warning(f"[multi-query] Expansion failed: {e}. Using original question only.")
        variants = []

    all_queries = [question] + [v for v in variants if v.lower() != question.lower()]
    return list(dict.fromkeys(all_queries))


# ── 2. FAISS dense retriever ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_faiss() -> Optional[FAISS]:
    faiss_path = settings.vector_db.faiss_path
    faiss_dir = Path(faiss_path)
    if not faiss_dir.exists() or not (faiss_dir / "index.faiss").exists():
        logger.warning(f"[dense] FAISS index path '{faiss_path}' does not exist.")
        return None

    emb = get_huggingface_embedding_model()
    return FAISS.load_local(
        faiss_path,
        emb,
        allow_dangerous_deserialization=True,
    )


def dense_retrieve(queries: list[str], k: int = DENSE_TOP_K) -> list[Document]:
    """
    Runs each query against FAISS, deduplicates documents by SHA-256 hash,
    and returns unique matching documents.
    """
    vs = _load_faiss()
    if vs is None:
        return []

    seen: dict[str, Document] = {}

    for q in queries:
        try:
            results = vs.similarity_search(q, k=k)
            for doc in results:
                key = _doc_key(doc)
                if key not in seen:
                    seen[key] = doc
        except Exception as e:
            logger.error(f"[dense] Search failed for query '{q[:50]}': {e}")

    return list(seen.values())


# ── 3. BM25 hybrid retrieval ──────────────────────────────────────────────────

class BM25Index:
    """
    Lightweight in-memory BM25 index built from candidate documents.
    """
    def __init__(self, docs: list[Document]):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        tokenized = [self._tokenize(d.page_content) for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i], float(scores[i])) for i in top_indices]


def hybrid_merge(
    dense_docs: list[Document],
    queries: list[str],
    k: int = DENSE_TOP_K,
) -> list[Document]:
    """
    Merges dense positional ranks and BM25 scores using a weighted combination.
    """
    if not dense_docs:
        return []

    index = BM25Index(dense_docs)
    content_to_doc = {_doc_key(d): d for d in dense_docs}

    bm25_scores: dict[str, float] = {key: 0.0 for key in content_to_doc}
    for q in queries:
        results = index.search(q, k=len(dense_docs))
        max_score = max((s for _, s in results), default=1.0) or 1.0
        for doc, score in results:
            key = _doc_key(doc)
            if key in bm25_scores:
                bm25_scores[key] += score / max_score

    total = len(dense_docs)
    dense_rank_scores = {
        _doc_key(doc): (total - i) / total
        for i, doc in enumerate(dense_docs)
    }

    combined: dict[str, float] = {}
    for key in content_to_doc:
        d_score = dense_rank_scores.get(key, 0.0)
        b_score = bm25_scores.get(key, 0.0) / max(len(queries), 1)
        combined[key] = DENSE_WEIGHT * d_score + BM25_WEIGHT * b_score

    sorted_keys = sorted(combined, key=lambda k: -combined[k])[:k]
    return [content_to_doc[k] for k in sorted_keys]


# ── 4. Cross-encoder re-ranking ───────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    logger.info(f"[rerank] Loading CrossEncoder model: {RERANK_MODEL}")
    return CrossEncoder(RERANK_MODEL)


def rerank(
    question: str,
    docs: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Scores each (question, doc) pair with a cross-encoder and returns top_n docs.
    """
    if not docs:
        return docs

    model = _load_cross_encoder()
    pairs = [(question, doc.page_content[:512]) for doc in docs]

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.error(f"[rerank] Cross-encoder failed: {e}. Returning docs unsorted.")
        return docs[:top_n]

    scored = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_n]]


# ── 5. Advanced retriever — public API ───────────────────────────────────────

class AdvancedRetriever:
    """
    Combines multi-query expansion, hybrid BM25+dense retrieval, and cross-encoder reranking.
    """

    def __init__(
        self,
        use_multi_query: bool = True,
        use_hybrid: bool = True,
        use_reranker: bool = True,
    ):
        self.use_multi_query = use_multi_query
        self.use_hybrid      = use_hybrid
        self.use_reranker    = use_reranker

    def retrieve(self, question: str) -> list[Document]:
        # Step 1: Query expansion
        if self.use_multi_query:
            queries = expand_queries(question)
            logger.info(f"[advanced] Expanded question into {len(queries)} variants.")
        else:
            queries = [question]

        # Step 2: Dense retrieval
        dense_docs = dense_retrieve(queries, k=DENSE_TOP_K)
        logger.info(f"[advanced] Dense retrieval fetched {len(dense_docs)} candidate documents.")

        if not dense_docs:
            return []

        # Step 3: Hybrid BM25 merge
        if self.use_hybrid and len(dense_docs) > 1:
            merged_docs = hybrid_merge(dense_docs, queries, k=DENSE_TOP_K)
            logger.info(f"[advanced] Hybrid merge produced {len(merged_docs)} documents.")
        else:
            merged_docs = dense_docs

        # Step 4: Cross-encoder re-rank
        if self.use_reranker:
            final_docs = rerank(question, merged_docs, top_n=RERANK_TOP_N)
            logger.info(f"[advanced] Re-ranked to final top {len(final_docs)} documents.")
        else:
            final_docs = merged_docs[:RERANK_TOP_N]

        return final_docs

    def invoke(self, question: str) -> list[Document]:
        """LangChain-compatible interface shim."""
        return self.retrieve(question)


@lru_cache(maxsize=1)
def get_advanced_retriever() -> AdvancedRetriever:
    return AdvancedRetriever()