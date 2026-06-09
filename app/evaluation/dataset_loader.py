# ── Step 1: Load & sample QA pairs ───────────────────────────────────────────

from app.evaluation.config import *
import json

def load_qa_pairs(num_pairs: int = NUM_PAIRS) -> list[dict]:
    if not RAW_VAL_PATH.exists():
        print("[eval] Validation data not found. Downloading ...")
        from app.retrieval.ingest_hotpotqa import download_hotpotqa
        download_hotpotqa("validation")

    print(f"[eval] Loading from {RAW_VAL_PATH} ...")
    with open(RAW_VAL_PATH, encoding="utf-8") as f:
        data = json.load(f)

    import random
    random.seed(RANDOM_SEED)
    sampled = random.sample(data, min(num_pairs, len(data)))

    pairs = []
    for item in sampled:
        question = item.get("question", "").strip()
        answer   = item.get("answer", "").strip()
        if question and answer:
            pairs.append({"question": question, "ground_truth": answer})

    print(f"[eval] {len(pairs)} QA pairs sampled (seed={RANDOM_SEED})")
    return pairs
