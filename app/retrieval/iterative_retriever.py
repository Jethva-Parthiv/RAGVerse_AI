"""
Phase 2.5 — Iterative Decomposed Retriever for RAGVerse_AI
------------------------------------------------------------
Solves the core HotpotQA multi-hop problem:
  "47/100 questions return cannot answer"

How it works:
    Step 1 → decompose_query()
             Ask Gemini: is this multi-hop? If yes, return ordered sub-questions.
             If no, return [original_question] (single hop, normal path).

    Step 2 → iterative_retrieve()
             For each sub-question:
               a. Retrieve docs using AdvancedRetriever (Phase 2)
               b. Extract the "bridge entity" — the key fact/name/year
                  that connects this hop to the next
               c. Inject the bridge entity into the NEXT sub-question
                  before retrieving it
             This means hop-2 query is dynamically constructed from
             what hop-1 actually found — not a static rewrite.

    Step 3 → Returns merged, deduplicated docs from ALL hops
             so the LLM has the full reasoning chain as context.

Usage (drop-in for advanced_retriever.py):
    from app.retrieval.iterative_retriever import get_iterative_retriever
    retriever = get_iterative_retriever()
    docs = retriever.retrieve("Where was the director of Titanic born?")

Example trace:
    Original: "Where was the director of Titanic born?"
    Sub-Q 1 : "Who directed Titanic?"
    Retrieve → "Titanic was directed by James Cameron"
    Bridge   : "James Cameron"
    Sub-Q 2 : "Where was James Cameron born?"  ← injected bridge
    Retrieve → "James Cameron was born in Kapuskasing, Ontario"
    Final docs: [hop-1 chunks] + [hop-2 chunks] → passed to LLM
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

from langchain_core.documents import Document

from app.llm.models import get_gemini_chat_model
from app.retrieval.advanced_retriever import get_advanced_retriever

# ── Config ────────────────────────────────────────────────────────────────────

MAX_HOPS        = 3     # safety cap — HotpotQA is max 2-hop
MIN_BRIDGE_LEN  = 2     # ignore bridge entities shorter than this
SINGLE_HOP_K    = 5     # docs to return for single-hop questions
MULTI_HOP_K     = 3     # docs per hop for multi-hop (merged across hops)


# ── Step 1: Query decomposition ───────────────────────────────────────────────

_DECOMPOSE_SYSTEM = """You are an expert at analyzing questions for a RAG retrieval system.

Your job: decide if a question requires multi-hop reasoning (needs information from
2+ separate sources that must be chained together), then decompose it if needed.

OUTPUT FORMAT — respond with ONLY valid JSON, no markdown, no explanation:

For a SINGLE-HOP question (answer in one source):
{
  "is_multihop": false,
  "subquestions": []
}

For a MULTI-HOP question (answer needs 2+ chained sources):
{
  "is_multihop": true,
  "subquestions": [
    "first sub-question that finds the bridge entity",
    "second sub-question with {bridge} placeholder where the bridge entity goes"
  ]
}

RULES for subquestions:
- Always put simpler/foundational questions FIRST
- Use {bridge} as a placeholder in later questions where the answer from the
  previous step should be inserted
- Maximum 3 subquestions
- Make each subquestion self-contained and searchable

EXAMPLES:

Q: "Where was the director of Titanic born?"
{
  "is_multihop": true,
  "subquestions": [
    "Who directed the film Titanic?",
    "Where was {bridge} born?"
  ]
}

Q: "Which film was released first, Inception or The Dark Knight?"
{
  "is_multihop": true,
  "subquestions": [
    "What year was Inception released?",
    "What year was The Dark Knight released?"
  ]
}

Q: "What is the capital of France?"
{
  "is_multihop": false,
  "subquestions": []
}

Q: "Which university did the founder of SpaceX attend?"
{
  "is_multihop": true,
  "subquestions": [
    "Who founded SpaceX?",
    "Which university did {bridge} attend?"
  ]
}"""


def decompose_query(question: str) -> dict:
    """
    Asks Gemini to classify and decompose the question.

    Returns:
        {
            "is_multihop": bool,
            "subquestions": list[str]   # empty if single-hop
        }
    """
    llm = get_gemini_chat_model()

    try:
        response = llm.invoke([
            {"role": "system", "content": _DECOMPOSE_SYSTEM},
            {"role": "user",   "content": f"Question: {question}"},
        ])
 
        # Generate — Gemini may return list or string in response.content
        content  = response.content
        if isinstance(content, list):
            raw = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            raw = str(content).strip()
  
        # Strip markdown fences if Gemini adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        # Validate structure
        assert "is_multihop"  in result
        assert "subquestions" in result
        assert isinstance(result["subquestions"], list)

        # Enforce cap
        result["subquestions"] = result["subquestions"][:MAX_HOPS]

        return result

    except Exception as e:
        print(f"[decompose] Failed to parse decomposition: {e}. Treating as single-hop.")
        return {"is_multihop": False, "subquestions": []}


# ── Step 2: Bridge entity extraction ─────────────────────────────────────────

_BRIDGE_SYSTEM = """You are extracting a KEY ENTITY from a document to use in
the next retrieval query.

Given a sub-question and the retrieved document chunks, extract the single most
important entity (person name, place, organization, year, title) that answers
the sub-question and will be needed for the follow-up query.

OUTPUT: respond with ONLY the entity string — no punctuation, no explanation,
no full sentence. Just the entity itself.

Examples:
  Sub-question: "Who directed Titanic?"
  Chunks mention: "Titanic was directed by James Cameron in 1997"
  Output: James Cameron

  Sub-question: "What company did Elon Musk found before Tesla?"
  Chunks mention: "Musk co-founded Zip2 Corporation in 1995"
  Output: Zip2 Corporation

If you cannot find the entity, output: UNKNOWN"""


def extract_bridge_entity(subquestion: str, docs: list[Document]) -> Optional[str]:
    """
    Extracts the key bridge entity from retrieved docs for a given sub-question.
    Returns None if extraction fails or entity is too short to be useful.
    """
    if not docs:
        return None

    llm     = get_gemini_chat_model()
    context = "\n\n".join(d.page_content for d in docs)

    try:
        response = llm.invoke([
            {"role": "system", "content": _BRIDGE_SYSTEM},
            {"role": "user",   "content": (
                f"Sub-question: {subquestion}\n\n"
                f"Retrieved chunks:\n{context}"
            )},
        ])

        content  = response.content
        if isinstance(content, list):
            entity = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        else:
            entity = str(content).strip()

        entity = entity.strip().strip('"').strip("'")

        if entity == "UNKNOWN" or len(entity) < MIN_BRIDGE_LEN:
            print("[DEBUG] entity : ", entity)
            print(f"[bridge] No usable entity found for: '{subquestion[:50]}'")
            return None

        print(f"[bridge] Extracted: '{entity}'")
        return entity

    except Exception as e:
        print(f"[bridge] Extraction failed: {e}")
        return None


# ── Step 3: Inject bridge into next sub-question ──────────────────────────────

def inject_bridge(subquestion: str, bridge_entity: str) -> str:
    """
    Replaces {bridge} placeholder with the extracted entity.
    If no placeholder exists, appends the entity as context.

    Examples:
        inject_bridge("Where was {bridge} born?", "James Cameron")
        → "Where was James Cameron born?"

        inject_bridge("What films did they direct?", "James Cameron")
        → "What films did James Cameron direct?"  (appended as context)
    """
    if "{bridge}" in subquestion:
        result = subquestion.replace("{bridge}", bridge_entity)
    else:
        # No placeholder — prepend entity as context
        result = f"{bridge_entity} — {subquestion}"

    print(f"[inject] '{subquestion}' → '{result}'")
    return result


# ── Step 4: Full iterative retrieval ─────────────────────────────────────────

def iterative_retrieve(
    question:     str,
    subquestions: list[str],
    k_per_hop:    int = MULTI_HOP_K,
) -> list[Document]:
    """
    Executes multi-hop retrieval:
      For each sub-question:
        1. Retrieve docs (using AdvancedRetriever from Phase 2)
        2. Extract bridge entity from results
        3. Inject bridge into the NEXT sub-question

    Returns merged, deduplicated docs from all hops.
    """
    retriever   = get_advanced_retriever()
    all_docs:    dict[str, Document] = {}   # keyed by content[:200] for dedup
    bridge:      Optional[str]       = None
    active_subs: list[str]           = list(subquestions)

    for hop_idx, subq in enumerate(active_subs):
        # Inject bridge from previous hop
        if bridge and "{bridge}" in subq:
            subq = inject_bridge(subq, bridge)
        elif bridge and hop_idx > 0:
            subq = inject_bridge(subq, bridge)

        print(f"[iterative] Hop {hop_idx + 1}/{len(active_subs)}: '{subq[:65]}'")

        # Retrieve for this hop
        hop_docs = retriever.retrieve(subq)[:k_per_hop * 2]

        # Add to merged pool
        for doc in hop_docs:
            key = doc.page_content[:200]
            if key not in all_docs:
                all_docs[key] = doc

        # Extract bridge for next hop (skip on last hop)
        if hop_idx < len(active_subs) - 1:
            bridge = extract_bridge_entity(subq, hop_docs)

            # If bridge extraction failed, note it but continue
            if bridge is None:
                print(f"[iterative] Warning: bridge extraction failed at hop {hop_idx + 1}. "
                      f"Next hop will use original sub-question.")

    merged = list(all_docs.values())
    print(f"[iterative] Merged {len(merged)} unique docs across {len(active_subs)} hops")
    return merged


# ── Step 5: Public API — IterativeRetriever ───────────────────────────────────

class IterativeRetriever:
    """
    Drop-in replacement for AdvancedRetriever.
    Routes single-hop questions to AdvancedRetriever unchanged.
    Routes multi-hop questions through iterative decomposed retrieval.

    Usage:
        retriever = IterativeRetriever()
        docs = retriever.retrieve("your question here")
        docs = retriever.invoke("also works")   # LangChain-compatible
    """

    def __init__(self):
        self._advanced = get_advanced_retriever()

    def retrieve(self, question: str) -> list[Document]:
        print(f"\n[iterative] Question: '{question[:70]}'")

        # Step 1: Decompose
        decomp = decompose_query(question)

        if not decomp["is_multihop"] or not decomp["subquestions"]:
            # Single-hop: use Phase 2 advanced retriever as-is
            print("[iterative] → Single-hop. Using AdvancedRetriever directly.")
            return self._advanced.retrieve(question)

        print(f"[iterative] → Multi-hop ({len(decomp['subquestions'])} hops). "
              f"Subquestions: {decomp['subquestions']}")

        # Step 2: Iterative multi-hop retrieval
        docs = iterative_retrieve(question, decomp["subquestions"])

        # Step 3: Fallback — if iterative found nothing, try direct retrieval
        if len(docs) < 2:
            print("[iterative] Warning: iterative retrieval found <2 docs. "
                  "Falling back to direct retrieval.")
            docs = self._advanced.retrieve(question)

        return docs

    def invoke(self, question: str) -> list[Document]:
        """LangChain-compatible shim."""
        return self.retrieve(question)

    def get_decomposition(self, question: str) -> dict:
        """
        Utility: inspect how a question gets decomposed without full retrieval.
        Useful for debugging.
        """
        return decompose_query(question)


@lru_cache(maxsize=1)
def get_iterative_retriever() -> IterativeRetriever:
    return IterativeRetriever()