import os
# ragas 0.1.21 — metrics are module-level singletons;
# llm/embeddings are passed to evaluate(), NOT to metric constructors
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset

from app.core.settings import *
from app.evaluation.config import CHECKPOINT_FILE,BATCH_SIZE
from app.llm.models import get_gemini_chat_model
from app.llm.embeddings import get_gemini_embedding_model
import pandas as pd
from datasets import Dataset


def _safe(row, col: str) -> float:
    val = row.get(col, float("nan"))
    try:
        f = float(val)
        return round(f if f == f else 0.0, 4)   # nan check
    except (TypeError, ValueError):
        return 0.0


def _col_mean(df, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return round(float(df[col].dropna().mean()), 4)



def score_with_ragas_in_batches(results: list[dict], batch_size: int = BATCH_SIZE, checkpoint_file: str = CHECKPOINT_FILE):
    """
    Evaluates the results in micro-batches and saves progress to a CSV file continuously.
    If a crash occurs, you can rerun the function and it will resume from where it left off.
    """
    
    # 1. Determine starting index if a checkpoint file already exists
    start_idx = 0
    if os.path.exists(checkpoint_file):
        try:
            existing_df = pd.read_csv(checkpoint_file)
            start_idx = len(existing_df)
            print(f"[eval] Found existing checkpoint with {start_idx} evaluated rows. Resuming...")
        except Exception:
            print("[eval] Checkpoint file corrupted or empty. Starting fresh.")
            start_idx = 0

    total_records = len(results)
    if start_idx >= total_records:
        print("[eval] All records have already been evaluated.")
        return load_final_results(checkpoint_file, results)

    # 2. Re-initialize model wrappers
    print("[eval] Wrapping Gemini for RAGAS ...")
    ragas_llm = LangchainLLMWrapper(get_gemini_chat_model())
    ragas_emb = LangchainEmbeddingsWrapper(get_gemini_embedding_model())
    
    # Keep the metric fix from earlier to avoid 400 candidate errors
    answer_relevancy.strictness = 1
    for _m in [faithfulness, answer_relevancy, context_precision, context_recall]:
        _m.reproducibility = 1

    run_config = RunConfig(
        max_workers=1,
        max_retries=2,
        max_wait=65,
        log_tenacity=False,
    )

    # 3. Loop through the dataset in safe increments
    for i in range(start_idx, total_records, batch_size):
        chunk = results[i : i + batch_size]
        print(f"\n[eval] Processing rows {i} to {min(i + batch_size, total_records)} / {total_records}...")
        
        # Build HuggingFace Dataset for just this slice
        chunk_dataset = Dataset.from_list([
            {
                "question":     r["question"],
                "answer":       r["answer"],
                "contexts":     r["contexts"],
                "ground_truth": r["ground_truth"],
            }
            for r in chunk
        ])
        
        # Evaluate current batch slice
        scores = evaluate(
            dataset=chunk_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_emb,
            run_config=run_config,
            raise_exceptions=False,  # Still logs warnings inside the slice instead of crashing
        )
        
        # Convert slice score to a Pandas DataFrame
        chunk_df = scores.to_pandas()
        
        # Append slice result immediately to CSV
        if not os.path.exists(checkpoint_file):
            chunk_df.to_csv(checkpoint_file, index=False)
        else:
            chunk_df.to_csv(checkpoint_file, mode='a', header=False, index=False)
            
        print(f"[eval] Successfully saved checkpoint to {checkpoint_file}")

    return load_final_results(checkpoint_file, results)


def load_final_results(checkpoint_file: str, original_results: list[dict]) -> tuple[dict, list[dict]]:
    """Helper to read completed CSV file, assemble the final dictionary outputs, and delete the checkpoint file."""
    scores_df = pd.read_csv(checkpoint_file)
    
    # Rebuild per-question output format expected by your application
    per_question = []
    for i, row in scores_df.iterrows():
        per_question.append({
            "question":          original_results[i]["question"],
            "ground_truth":      original_results[i]["ground_truth"],
            "answer":            original_results[i]["answer"],
            "faithfulness":      _safe(row, "faithfulness"),
            "answer_relevancy":  _safe(row, "answer_relevancy"),
            "context_precision": _safe(row, "context_precision"),
            "context_recall":    _safe(row, "context_recall"),
        })

    summary = {
        "faithfulness":      _col_mean(scores_df, "faithfulness"),
        "answer_relevancy":  _col_mean(scores_df, "answer_relevancy"),
        "context_precision": _col_mean(scores_df, "context_precision"),
        "context_recall":    _col_mean(scores_df, "context_recall"),
    }

    print(f"\n[eval] Done! Scored {len(scores_df)} samples total.")
    
    # ── NEW CODE: Safely delete the checkpoint file ──
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print(f"[eval] Successfully cleaned up and deleted checkpoint file: {checkpoint_file}")
        except Exception as e:
            print(f"[eval] Warning: Could not delete checkpoint file due to error: {e}")
            
    return summary, per_question