# ── Step 1: Load & sample QA pairs ───────────────────────────────────────────

from app.evaluation.config import *
import json

def load_qa_pairs(num_pairs: int = NUM_PAIRS) -> list[dict]:
    if not RAW_VAL_PATH.exists():
        print("[eval] Validation data file not found locally. Downloading via HuggingFace datasets...")
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "distractor", split="validation")
        RAW_VAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw_list = [{"question": row["question"], "answer": row["answer"]} for row in ds]
        with open(RAW_VAL_PATH, "w", encoding="utf-8") as f:
            json.dump(raw_list, f, indent=2)

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
