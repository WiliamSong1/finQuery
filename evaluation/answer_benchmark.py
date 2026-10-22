
import asyncio
import json
import math
import os
import statistics
import time

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from app.database.vector_store import get_client
from app.retrieval.hybrid_search import reranked_hybrid_search
from app.generation.answer import generate_answer


load_dotenv()

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-20b"
)


TEST_CASES = [
    {
        "id": "rd_growth_2024",
        "question": (
            "Why did Apple's research and development "
            "expenses increase in 2024 compared with 2023?"
        )
    },
    {
        "id": "rd_growth_2025",
        "question": (
            "Why did Apple's research and development "
            "expenses increase in 2025 compared with 2024?"
        )
    },
    {
        "id": "operating_expenses_2025",
        "question": (
            "What were Apple's total operating "
            "expenses in fiscal 2025?"
        )
    }
]


def collect_finquery_outputs():
    """
    Execute real retrieval and answer generation.

    Use the same retrieved chunks for generation and
    evaluation, avoiding a second retrieval operation.
    """

    records = []

    for case in TEST_CASES:
        print(f"\nRunning FinQuery: {case['id']}", flush=True)

        record = {
            "id": case["id"],
            "question": case["question"],
            "status": "pending",
            "faithfulness": None
        }

        try:
            start = time.perf_counter()

            chunks = reranked_hybrid_search(
                query=case["question"],
                limit=5,
                candidate_limit=20
            )

            retrieval_ms = (
                time.perf_counter() - start
            ) * 1000

            generation_start = time.perf_counter()

            result = generate_answer(
                query=case["question"],
                limit=5,
                chunks=chunks
            )

            generation_ms = (
                time.perf_counter() - generation_start
            ) * 1000

            record.update({
                "status": result.get("status"),
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "internal_verification": result.get(
                    "verification"
                ),
                "retrieved_contexts": [
                    chunk["text"]
                    for chunk in chunks
                ],
                "retrieved_chunks": [
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "document": chunk.get("document"),
                        "page": chunk.get("page"),
                        "rerank_score": chunk.get(
                            "rerank_score"
                        )
                    }
                    for chunk in chunks
                ],
                "retrieval_latency_ms": round(
                    retrieval_ms, 2
                ),
                "generation_latency_ms": round(
                    generation_ms, 2
                )
            })

        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)

        print(f"Status: {record['status']}", flush=True)

        records.append(record)

    return records


async def evaluate_faithfulness(records):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    async with AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=90.0,
        max_retries=1
    ) as client:

        # This is the configuration that worked
        # in our previous Ragas smoke test.
        llm = llm_factory(
            MODEL_NAME,
            provider="openai",
            client=client,
            adapter="instructor",
            max_tokens=2048
        )

        metric = Faithfulness(llm=llm)

        for record in records:
            if record["status"] != "answered":
                record["evaluation_status"] = "skipped"
                record["evaluation_reason"] = (
                    "FinQuery did not return an "
                    "accepted answer"
                )
                continue

            if not record.get("retrieved_contexts"):
                record["evaluation_status"] = "skipped"
                record["evaluation_reason"] = (
                    "No retrieved evidence"
                )
                continue

            print(
                f"Scoring: {record['id']}",
                flush=True
            )

            try:
                start = time.perf_counter()

                result = await metric.ascore(
                    user_input=record["question"],
                    response=record["answer"],
                    retrieved_contexts=record[
                        "retrieved_contexts"
                    ]
                )

                score = float(result.value)

                if not math.isfinite(score):
                    raise ValueError(
                        "Non-finite faithfulness score"
                    )

                record["faithfulness"] = round(
                    score, 4
                )
                record["evaluation_status"] = "scored"

                record["evaluation_latency_ms"] = round(
                    (time.perf_counter() - start) * 1000,
                    2
                )

            except Exception as exc:
                record["evaluation_status"] = "error"
                record["evaluation_error"] = str(exc)


def summarize(records):
    scores = [
        record["faithfulness"]
        for record in records
        if record.get("evaluation_status") == "scored"
    ]

    answered = sum(
        record.get("status") == "answered"
        for record in records
    )

    return {
        "total_questions": len(records),
        "accepted_answers": answered,
        "answer_rate": round(
            answered / len(records), 4
        ) if records else None,
        "faithfulness_scored": len(scores),
        "mean_faithfulness": (
            round(statistics.mean(scores), 4)
            if scores else None
        ),
        "evaluation_errors": sum(
            record.get("evaluation_status") == "error"
            for record in records
        )
    }


async def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        records = collect_finquery_outputs()
    finally:
        # Release local Qdrant resources.
        get_client().close()

    await evaluate_faithfulness(records)

    output = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
        "generator_model": MODEL_NAME,
        "judge_model": MODEL_NAME,
        "summary": summarize(records),
        "results": records
    }

    output_path = (
        RESULTS_DIR / "answer_benchmark.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(output, file, indent=2)

    print("\n=== ANSWER BENCHMARK SUMMARY ===")
    print(json.dumps(output["summary"], indent=2))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
