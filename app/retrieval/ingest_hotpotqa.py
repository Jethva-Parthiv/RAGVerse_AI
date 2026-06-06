"""
RAGVerse_AI — HotpotQA ingestion pipeline
------------------------------------------------------
Downloads HotpotQA, extracts Wikipedia context paragraphs,
chunks them, embeds with HuggingFace(Sentence Transformer), and saves a FAISS index.

Usage:
    python -m app.retrieval.ingest_hotpotqa

Optional env vars (override settings.py defaults):
    INGEST_SPLIT      — "train" | "validation" (default: "train")
    INGEST_MAX_DOCS   — int, cap total docs for quick tests (default: all)
    INGEST_BATCH_SIZE — embedding batch size (default: 256)
    INGEST_SAVE_EVERY — save FAISS checkpoint every N batches (default: 50)
"""

import json
import os
import time
from datasets import load_dataset
from pathlib import Path
from typing import Generator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.core.settings import *
from app.llm.embeddings import get_huggingface_embedding_model


RAW_DIR       = Path("data/raw/hotpotqa")
PROCESSED_DIR = Path("data/processed")
CHECKPOINT_PATH = PROCESSED_DIR / "faiss_checkpoint"

INGEST_SPLIT      = os.getenv("INGEST_SPLIT", "train")
INGEST_MAX_DOCS   = int(os.getenv("INGEST_MAX_DOCS", 0))   # 0 = no cap
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", 256))
INGEST_SAVE_EVERY = int(os.getenv("INGEST_SAVE_EVERY", 50))


# ── Step 1: Download raw JSON ─────────────────────────────────────────────────


def download_hotpotqa(split: str = INGEST_SPLIT) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"{split}.json"

    if dest.exists():
        print(f"[download] Already exists: {dest}")
        return dest

    print(f"[download] Fetching HotpotQA '{split}' split via Hugging Face datasets...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split=split)

    records = [dict(row) for row in dataset]
    with open(dest, "w") as f:
        json.dump(records, f)

    print(f"[download] Saved to {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest



def extract_documents(raw_path, max_docs: int = 0) -> list[dict]:
    """
    Each HotpotQA item contains a 'context' field:
        {
            "sentences": [["Sent 1", ...], ["Sent 2", ...]],
            "title": ["Title1", "Title2"]
        }

    We join each article's sentences into one document and deduplicate by title.
    Returns list of {"page_content": str, "metadata": {"title": str, "source": "hotpotqa"}}
    """
    print(f"[extract] Loading {raw_path} ...")
    with open(raw_path, encoding="utf-8") as f:
        data = json.load(f)

    seen_titles: set[str] = set()
    docs: list[dict] = []

    for item in data:
        context = item.get("context", {})
        titles = context.get("title", [])
        sentences_list = context.get("sentences", [])

        for title, sentences in zip(titles, sentences_list):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            text = " ".join(sentences).strip()
            if not text:
                continue
            docs.append({
                "page_content": text,
                "metadata": {"title": title, "source": "hotpotqa"},
            })
            if max_docs and len(docs) >= max_docs:
                break
        if max_docs and len(docs) >= max_docs:
            break

    print(f"[extract] Unique articles extracted: {len(docs):,}")
    return docs



# ── Step 3: Chunk documents ───────────────────────────────────────────────────

def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in docs:
        splits = splitter.split_text(doc["page_content"])
        for i, text in enumerate(splits):
            chunks.append({
                "page_content": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })

    print(f"[chunk] Total chunks: {len(chunks):,}  "
          f"(avg {len(chunks) // max(len(docs), 1)} per doc)")
    return chunks


# ── Step 4: Batch embed + build FAISS (with checkpointing) ───────────────────

def _batches(lst: list, size: int) -> Generator[list, None, None]:
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def build_faiss_index(chunks: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    embedding_model = get_huggingface_embedding_model()

    # Load checkpoint if it exists
    vectorstore: FAISS | None = None
    start_batch = 0

    state_file = CHECKPOINT_PATH / "progress.json"
    if CHECKPOINT_PATH.exists() and state_file.exists():
        with open(state_file) as f:
            state = json.load(f)
        start_batch = state.get("next_batch", 0)
        print(f"[embed] Resuming from batch {start_batch}")
        vectorstore = FAISS.load_local(
            str(CHECKPOINT_PATH),
            embedding_model,
            allow_dangerous_deserialization=True,
        )


    all_batches = list(_batches(chunks, INGEST_BATCH_SIZE))
    total = len(all_batches)

    for batch_idx, batch in enumerate(all_batches):
        if batch_idx < start_batch:
            continue

        docs = [Document(page_content=c["page_content"], metadata=c["metadata"])
                for c in batch]

        # Retry loop for Gemini rate limits
        for attempt in range(5):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(docs, embedding_model)
                else:
                    vectorstore.add_documents(docs)
                # time.sleep(65)
                break
            except Exception as e:
                wait = 3 ** attempt
                print(f"[embed] Batch {batch_idx} attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait}s ...")
                time.sleep(wait)
        else:
            print(f"[embed] Batch {batch_idx} permanently failed. Saving and exiting.")
            _save_checkpoint(vectorstore, batch_idx, state_file)
            raise RuntimeError(f"Embedding failed at batch {batch_idx}")

        done = batch_idx + 1
        print(f"[embed] {done}/{total} batches done "
              f"({done * INGEST_BATCH_SIZE:,} chunks embedded)", end="\r")

        # Periodic checkpoint
        if done % INGEST_SAVE_EVERY == 0:
            _save_checkpoint(vectorstore, done, state_file)

    print()  # newline after \r

    # Final save to production FAISS_PATH
    final_path = FAISS_PATH
    Path(final_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(final_path)
    print(f"[embed] ✓ FAISS index saved to {final_path}")
    print(f"[embed] ✓ Total vectors: {vectorstore.index.ntotal:,}")

    # Clean up checkpoint
    # import shutil
    # if CHECKPOINT_PATH.exists():
    #     shutil.rmtree(CHECKPOINT_PATH)
    #     print("[embed] Checkpoint cleaned up.")


def _save_checkpoint(vectorstore: FAISS, next_batch: int, state_file: Path) -> None:
    CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(CHECKPOINT_PATH))
    with open(state_file, "w") as f:
        json.dump({"next_batch": next_batch}, f)
    print(f"\n[checkpoint] Saved at batch {next_batch}")


# ── Step 5: Stats reporter ────────────────────────────────────────────────────

def print_index_stats() -> None:
    """Load the saved index and print a summary."""
    from app.retrieval.retriever import get_retriever
    retriever = get_retriever()
    vs = retriever.vectorstore
    print("\n── Index stats ─────────────────────────────")
    print(f"  Vectors stored : {vs.index.ntotal:,}")
    print(f"  Embedding dim  : {vs.index.d}")
    print(f"  FAISS path     : {FAISS_PATH}")
    print("────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print("  RAGVerse — HotpotQA ingestion pipeline")
    print("=" * 55)
    print(f"  Split      : {INGEST_SPLIT}")
    print(f"  Max docs   : {INGEST_MAX_DOCS or 'all'}")
    print(f"  Batch size : {INGEST_BATCH_SIZE}")
    print(f"  Chunk size : {CHUNK_SIZE} / overlap {CHUNK_OVERLAP}")
    print(f"  FAISS path : {FAISS_PATH}")
    print("=" * 55)

    raw_path = download_hotpotqa(INGEST_SPLIT)
    docs     = extract_documents(raw_path, INGEST_MAX_DOCS)
    chunks   = chunk_documents(docs)
    build_faiss_index(chunks)
    print_index_stats()
    print("\n✓ Ingestion complete. Your FAISS index is ready.\n")


if __name__ == "__main__":
    main()