"""
Advanced Retriever for RAGVerse_AI
---------------------------------------------
Add three techniques in one call:

    1. Multi-query expansion  → better recall 
    2. Cross-encoder re-ranking → better precision 
    3. Hybrid BM25 + dense    → catches keyword matches dense misses

"""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.settings import *
from app.llm.embeddings import get_huggingface_embedding_model
from app.llm.models import get_gemini_chat_model

# ── Config ────────────────────────────────────────────────────────────────────

DENSE_TOP_K       = int(TOP_K_RESULTS * 4)   # fetch more, re-rank later
MULTI_QUERY_COUNT = 3                         # number of query variants
RERANK_TOP_N      = TOP_K_RESULTS             # final docs passed to LLM
RERANK_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # tiny + fast
BM25_WEIGHT       = 0.3   # weight for BM25 scores in hybrid merge
DENSE_WEIGHT      = 0.7   # weight for dense scores in hybrid merge


# ── 1. Multi-query expansion ──────────────────────────────────────────────────

MULTI_QUERY_PROMPT = """\
You are an AI assistant helping improve document retrieval.
Given the user question below, generate {n} different versions of it.
Each version should rephrase the question to capture different angles,
synonyms, or sub-questions that would help find relevant documents.

Output ONLY the {n} questions, one per line. No numbering, no explanations.

Original question: {question}
"""


def expand_queries(question: str, n: int = 3) -> list[str]:
    """
    Uses Gemini to generate n query variants for the original question.
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
        print(f"[multi-query] Expansion failed: {e}. Using original only.")
        variants = []

    # Always keep original; deduplicate
    all_queries = [question] + [v for v in variants if v.lower() != question.lower()]
    return list(dict.fromkeys(all_queries))   # preserves order, removes duplicates


# ── 2. FAISS dense retriever ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_faiss() -> FAISS:
    emb = get_huggingface_embedding_model()
    return FAISS.load_local(
        FAISS_PATH,
        emb,
        allow_dangerous_deserialization=True,
    )


def dense_retrieve(queries: list[str], k: int = DENSE_TOP_K) -> list[Document]:
    """
    Runs each query against FAISS, deduplicates by page_content,
    returns up to k * len(queries) unique docs.
    """
    vs = _load_faiss()
    seen: dict[str, Document] = {}

    for q in queries:
        try:
            results = vs.similarity_search(q, k=k)
            for doc in results:
                key = doc.page_content[:200]   # dedup key
                if key not in seen:
                    seen[key] = doc
        except Exception as e:
            print(f"[dense] Search failed for query '{q[:50]}': {e}")

    return list(seen.values())


# ── 3. BM25 hybrid retrieval ──────────────────────────────────────────────────

class BM25Index:
    """
    Lightweight BM25 index built from a list of Documents.
    Built lazily so startup is not blocked.
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


_bm25_index: Optional[BM25Index] = None


def _get_bm25_index(docs: list[Document]) -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        print("[hybrid] Building BM25 index from candidate docs ...")
        _bm25_index = BM25Index(docs)
    return _bm25_index


def hybrid_merge(
    dense_docs: list[Document],
    queries: list[str],
    k: int = DENSE_TOP_K,
) -> list[Document]:
    """
    Merges dense and BM25 scores using weighted combination.
    BM25 scores are normalized to [0, 1] before merging.
    """
    index = _get_bm25_index(dense_docs)
    content_to_doc = {d.page_content[:200]: d for d in dense_docs}

    # Accumulate BM25 scores across all queries
    bm25_scores: dict[str, float] = {key: 0.0 for key in content_to_doc}
    for q in queries:
        results = index.search(q, k=len(dense_docs))
        max_score = max((s for _, s in results), default=1.0) or 1.0
        for doc, score in results:
            key = doc.page_content[:200]
            if key in bm25_scores:
                bm25_scores[key] += score / max_score   # normalize

    # Dense docs get a uniform descending score (positional rank)
    total = len(dense_docs)
    dense_rank_scores = {
        doc.page_content[:200]: (total - i) / total
        for i, doc in enumerate(dense_docs)
    }

    # Combine
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
    print(f"[rerank] Loading cross-encoder: {RERANK_MODEL} ...")
    return CrossEncoder(RERANK_MODEL)


def rerank(
    question: str,
    docs: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Scores each (question, doc) pair with a cross-encoder.
    Returns top_n docs sorted by relevance score descending.
    """
    if not docs:
        return docs

    model = _load_cross_encoder()
    pairs = [(question, doc.page_content[:512]) for doc in docs]

    try:
        scores = model.predict(pairs)
    except Exception as e:
        print(f"[rerank] Cross-encoder failed: {e}. Returning docs unsorted.")
        return docs[:top_n]

    scored = sorted(zip(scores, docs), key=lambda x: -x[0])
    return [doc for _, doc in scored[:top_n]]


# ── 5. Advanced retriever — public API ───────────────────────────────────────

class AdvancedRetriever:
    """
    Drop-in replacement for LangChain BaseRetriever.
    Combines multi-query + hybrid BM25+dense + cross-encoder re-ranking.

    Usage:
        retriever = AdvancedRetriever()
        docs = retriever.retrieve("your question here")
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
        # Step 1: expand queries
        if self.use_multi_query:
            queries = expand_queries(question)
            print(f"[advanced] Multi-query expanded to {len(queries)} queries")
        else:
            queries = [question]

        # Step 2: dense retrieval across all queries
        dense_docs = dense_retrieve(queries, k=DENSE_TOP_K)
        print(f"[advanced] Dense retrieved {len(dense_docs)} unique docs")

        # Step 3: hybrid merge with BM25
        if self.use_hybrid and len(dense_docs) > 1:
            merged_docs = hybrid_merge(dense_docs, queries, k=DENSE_TOP_K)
            print(f"[advanced] Hybrid merged to {len(merged_docs)} docs")
        else:
            merged_docs = dense_docs

        # Step 4: re-rank with cross-encoder
        if self.use_reranker:
            final_docs = rerank(question, merged_docs, top_n=RERANK_TOP_N)
            print(f"[advanced] Re-ranked to final {len(final_docs)} docs")
        else:
            final_docs = merged_docs[:RERANK_TOP_N]

        return final_docs

    # LangChain-compatible shim so it also works with .invoke()
    def invoke(self, question: str) -> list[Document]:
        return self.retrieve(question)


@lru_cache(maxsize=1)
def get_advanced_retriever() -> AdvancedRetriever:
    return AdvancedRetriever()