from app.evaluation.config import NUM_PAIRS
from app.evaluation.dataset_loader import load_qa_pairs
from app.evaluation.pipeline_runner import run_rag_pipeline
from app.evaluation.ragas_scorer import score_with_ragas_in_batches
from app.evaluation.report_generator import save_results


def main():
    pairs = load_qa_pairs(NUM_PAIRS)

    results = run_rag_pipeline(pairs)

    summary, per_question = score_with_ragas_in_batches(results)

    save_results(summary, per_question)


if __name__ == "__main__":
    main()