"""
Score comparison utility — RAGVerse_AI
---------------------------------------

Usage:
    python -m app.evaluation.compare_scores

It reads all rows from data/eval/baseline_summary.csv
and prints a delta table showing what improved or regressed.
"""

import csv
from pathlib import Path

SUMMARY_CSV = Path("data/eval/baseline_summary.csv")


def load_all_runs() -> list[dict]:
    if not SUMMARY_CSV.exists():
        print("[compare] No summary CSV found. Run ragas_eval.py first.")
        return []
    with open(SUMMARY_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare() -> None:
    runs = load_all_runs()
    if len(runs) < 2:
        print("[compare] Need at least 2 eval runs to compare.")
        print(f"          Current runs: {len(runs)}")
        if runs:
            _print_single(runs[0])
        return

    baseline = runs[0]
    latest   = runs[-1]
    metrics  = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    print("\n" + "=" * 65)
    print("  RAGVerse — Score Comparison")
    print("=" * 65)
    print(f"  Baseline : {baseline['timestamp']}  (n={baseline['num_pairs']})")
    print(f"  Latest   : {latest['timestamp']}    (n={latest['num_pairs']})")
    print("-" * 65)
    print(f"  {'Metric':<22} {'Baseline':>10} {'Latest':>10} {'Delta':>10}  {'Trend'}")
    print("-" * 65)

    for m in metrics:
        b = float(baseline[m])
        l = float(latest[m])
        d = l - b
        trend = "✅ +" if d > 0.01 else ("⚠️  " if d < -0.01 else "➡️  ~")
        print(f"  {m:<22} {b:>10.4f} {l:>10.4f} {d:>+10.4f}  {trend}")

    print("=" * 65)

    # Show all historical runs if more than 2
    if len(runs) > 2:
        print("\n  All runs:")
        print(f"  {'Timestamp':<22} {'faith':>6} {'relev':>6} {'prec':>6} {'recall':>6}")
        print("  " + "-" * 48)
        for r in runs:
            print(f"  {r['timestamp']:<22} "
                  f"{float(r['faithfulness']):>6.3f} "
                  f"{float(r['answer_relevancy']):>6.3f} "
                  f"{float(r['context_precision']):>6.3f} "
                  f"{float(r['context_recall']):>6.3f}")


def _print_single(run: dict) -> None:
    print("\n── Only one run found (baseline) ─────────────────────────")
    for k, v in run.items():
        print(f"  {k:<22} : {v}")


def log_new_run(label: str, scores: dict, num_pairs: int) -> None:
    """
    Call this from Phase 2 scripts to append a new score row.
    scores = {"faithfulness": 0.xx, "answer_relevancy": 0.xx, ...}
    label  = short description e.g. "reranker+multiquery"
    """
    from datetime import datetime
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    timestamp = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{label}]"

    fieldnames = ["timestamp", "num_pairs",
                  "faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]

    file_exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"timestamp": timestamp, "num_pairs": num_pairs, **scores})

    print(f"[compare] Logged new run: {timestamp}")


if __name__ == "__main__":
    compare()