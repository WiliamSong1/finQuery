
import csv
import json
import math
import statistics
import time

from pathlib import Path

from app.database.vector_store import search_chunks
from app.retrieval.bm25_search import bm25_search
from app.retrieval.hybrid_search import (
    hybrid_search,
    reranked_hybrid_search
)


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "benchmark_questions.json"
RESULTS_DIR = ROOT / "results"

TOP_K = 5


SEARCH_METHODS = {
    "dense": search_chunks,
    "bm25": bm25_search,
    "hybrid": hybrid_search,
    "reranked": reranked_hybrid_search
}


def load_questions():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize(text):
    return " ".join(text.lower().split())


def matches_evidence(result, evidence):
    return (
        result.get("document", "").lower()
        == evidence["document"].lower()
        and result.get("page") == evidence["page"]
        and normalize(evidence["contains"])
        in normalize(result.get("text", ""))
    )


def evaluate_results(results, relevant_evidence):
    matched_labels = set()
    first_relevant_rank = None

    for rank, result in enumerate(results, start=1):
        for index, evidence in enumerate(relevant_evidence):
            if matches_evidence(result, evidence):
                matched_labels.add(index)

                if first_relevant_rank is None:
                    first_relevant_rank = rank

    recall = len(matched_labels) / len(relevant_evidence)

    hit = 1.0 if matched_labels else 0.0

    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank is not None
        else 0.0
    )

    return {
        "recall_at_5": recall,
        "hit_at_5": hit,
        "mrr_at_5": reciprocal_rank
    }


def percentile(values, percent):
    if not values:
        return 0.0

    ordered = sorted(values)

    position = math.ceil(
        percent / 100 * len(ordered)
    ) - 1

    return ordered[max(0, position)]


def run_benchmark():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    questions = load_questions()

    if not questions:
        raise ValueError("Benchmark dataset is empty")

    detailed_results = []
    summary_results = []

    for method_name, search_function in SEARCH_METHODS.items():
        print(f"\nEvaluating {method_name.upper()}")

        # Warm up embedding models and reranker.
        search_function(
            questions[0]["question"],
            limit=TOP_K
        )

        method_results = []

        for question in questions:
            start_time = time.perf_counter()

            results = search_function(
                question["question"],
                limit=TOP_K
            )

            latency_ms = (
                time.perf_counter() - start_time
            ) * 1000

            metrics = evaluate_results(
                results,
                question["relevant_evidence"]
            )

            record = {
                "method": method_name,
                "question_id": question["id"],
                "question": question["question"],
                "latency_ms": round(latency_ms, 2),
                **metrics,
                "retrieved_chunks": [
                    {
                        "chunk_id": item.get("chunk_id"),
                        "document": item.get("document"),
                        "page": item.get("page")
                    }
                    for item in results
                ]
            }

            method_results.append(record)
            detailed_results.append(record)

            print(
                f"{question['id']} | "
                f"Recall@5: {metrics['recall_at_5']:.2f} | "
                f"MRR@5: {metrics['mrr_at_5']:.2f} | "
                f"Latency: {latency_ms:.0f} ms"
            )

        latencies = [
            item["latency_ms"]
            for item in method_results
        ]

        summary_results.append({
            "method": method_name,
            "questions": len(method_results),
            "recall_at_5": round(
                statistics.mean(
                    item["recall_at_5"]
                    for item in method_results
                ), 4
            ),
            "hit_at_5": round(
                statistics.mean(
                    item["hit_at_5"]
                    for item in method_results
                ), 4
            ),
            "mrr_at_5": round(
                statistics.mean(
                    item["mrr_at_5"]
                    for item in method_results
                ), 4
            ),
            "p50_latency_ms": round(
                statistics.median(latencies), 2
            ),
            "p95_latency_ms": round(
                percentile(latencies, 95), 2
            )
        })

    with open(
        RESULTS_DIR / "retrieval_details.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(detailed_results, file, indent=2)

    with open(
        RESULTS_DIR / "retrieval_summary.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_results[0].keys())
        )

        writer.writeheader()
        writer.writerows(summary_results)

    print("\n=== BENCHMARK SUMMARY ===")

    for row in summary_results:
        print(row)

    print("\nResults saved to evaluation/results/")


if __name__ == "__main__":
    run_benchmark()
