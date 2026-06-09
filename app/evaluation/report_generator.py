
# ── Step 4: Save results ──────────────────────────────────────────────────────

import csv
from datetime import datetime
from app.evaluation.config import *


def save_results(summary: dict, per_question: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Per-question CSV (overwrite each run)
    fields = ["question", "ground_truth", "answer",
              "faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(per_question)
    print(f"\n[eval] Per-question CSV  → {RESULTS_CSV}")

    # Summary CSV (append — one row per run, used by compare_scores.py)
    sum_fields = ["timestamp", "num_pairs", "faithfulness", "answer_relevancy",
                  "context_precision", "context_recall"]
    exists = SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields)
        if not exists:
            w.writeheader()
        w.writerow({"timestamp": timestamp, "num_pairs": len(per_question), **summary})
    print(f"[eval] Summary CSV       → {SUMMARY_CSV}")

    _write_report(summary, per_question, timestamp)
    print(f"[eval] Human report      → {REPORT_TXT}")


def _write_report(summary: dict, per_question: list[dict], timestamp: str) -> None:
    worst_faith  = sorted(per_question, key=lambda x: x["faithfulness"])[:5]
    worst_recall = sorted(per_question, key=lambda x: x["context_recall"])[:5]

    def band(v):
        if v < 0.40: return "Poor"
        if v < 0.65: return "Fair"
        if v < 0.80: return "Good"
        return "Strong"

    lines = [
        "=" * 62,
        "  RAGVerse Phase 1 — RAGAS Baseline Report",
        "=" * 62,
        f"  Generated  : {timestamp}",
        f"  QA pairs   : {len(per_question)}",
        "",
        "── Aggregate Scores ────────────────────────────────────────",
        f"  Faithfulness        : {summary['faithfulness']:.4f}  ({band(summary['faithfulness'])})",
        f"  Answer Relevancy    : {summary['answer_relevancy']:.4f}  ({band(summary['answer_relevancy'])})",
        f"  Context Precision   : {summary['context_precision']:.4f}  ({band(summary['context_precision'])})",
        f"  Context Recall      : {summary['context_recall']:.4f}  ({band(summary['context_recall'])})",
        "",
        "── 5 Worst Faithfulness (hallucination risk) ───────────────",
    ]
    for q in worst_faith:
        lines.append(f"  [{q['faithfulness']:.2f}] {q['question'][:72]}")
    lines += [
        "",
        "── 5 Worst Context Recall (retrieval gaps) ─────────────────",
    ]
    for q in worst_recall:
        lines.append(f"  [{q['context_recall']:.2f}] {q['question'][:72]}")
    lines += [
        "",
        "── Phase 2 Recommendations ─────────────────────────────────",
        "  Low faithfulness  → re-ranking, reduce TOP_K",
        "  Low recall        → multi-query, hybrid BM25+dense",
        "  Low precision     → tune CHUNK_SIZE, add MMR",
        "  Low relevancy     → improve BASE_RAG_RULES prompt",
        "=" * 62,
    ]

    report = "\n".join(lines)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
