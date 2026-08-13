from pathlib import Path
import os
from app.core.config import settings

EVAL_DIR = settings.paths.eval_data_dir
RAW_VAL_PATH = settings.paths.data_dir / "raw" / "hotpotqa" / "validation.json"

RESULTS_CSV = EVAL_DIR / "baseline_results.csv"
SUMMARY_CSV = EVAL_DIR / "baseline_summary.csv"
REPORT_TXT = EVAL_DIR / "baseline_report.txt"

NUM_PAIRS = int(os.getenv("EVAL_NUM_PAIRS", 100))
RANDOM_SEED = int(os.getenv("EVAL_SEED", 42))
SLEEP_BETWEEN = float(os.getenv("EVAL_SLEEP", 2.0))

CHECKPOINT_FILE = EVAL_DIR / "ragas_eval_progress.csv"

BATCH_SIZE = 10

