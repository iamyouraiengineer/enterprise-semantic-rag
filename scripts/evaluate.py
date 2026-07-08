"""
CLI entrypoint for running RAG evaluation on a golden dataset.

Usage:
    python scripts/evaluate.py --dataset data/golden_dataset.json
"""

import argparse
import json

from config.log_config import configure_logging
from src.embedding.embedder import Embedder
from src.evaluation.metrics import RAGEvaluator


def load_golden_dataset(path: str) -> list:
    """Load evaluation dataset from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--dataset", type=str, default="data/golden_dataset.json", help="Path to golden dataset")
    args = parser.parse_args()

    embedder = Embedder()
    evaluator = RAGEvaluator(embedder=embedder)

    dataset = load_golden_dataset(args.dataset)

    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(dataset)} questions")
    print(f"{'='*60}\n")

    all_scores = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]
        retrieved = item["retrieved_chunks"]
        ground_truth = item.get("ground_truth_chunks")

        scores = evaluator.evaluate(question, answer, retrieved, ground_truth)
        all_scores.append(scores)

        print(f"Q: {question}")
        for metric, score in scores.items():
            print(f"  {metric}: {score:.3f}")
        print()

    # Aggregate
    print(f"{'='*60}")
    print("AGGREGATE SCORES")
    print(f"{'='*60}")
    metrics = all_scores[0].keys()
    for metric in metrics:
        avg = sum(s[metric] for s in all_scores) / len(all_scores)
        print(f"  {metric}: {avg:.3f}")
    print()


if __name__ == "__main__":
    main()