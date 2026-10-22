
import asyncio
import json
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


load_dotenv()

RESULTS_DIR = Path("evaluation/results")


TEST_CASES = [
    {
        "id": "supported_rd_growth",
        "question": (
            "Why did Apple's R&D expenses increase in 2025?"
        ),
        "answer": (
            "Apple's R&D expenses increased primarily "
            "because of higher headcount-related expenses "
            "and infrastructure-related costs."
        ),
        "contexts": [
            (
                "The growth in R&D expense during 2025 "
                "compared to 2024 was primarily driven "
                "by increases in headcount-related expenses "
                "and infrastructure-related costs."
            )
        ],
        "expected": "supported"
    },
    {
        "id": "unsupported_engineers",
        "question": (
            "Why did Apple's R&D expenses increase in 2025?"
        ),
        "answer": (
            "Apple's R&D expenses increased because "
            "the company hired exactly 10,000 engineers."
        ),
        "contexts": [
            (
                "The growth in R&D expense during 2025 "
                "compared to 2024 was primarily driven "
                "by increases in headcount-related expenses "
                "and infrastructure-related costs."
            )
        ],
        "expected": "unsupported"
    }
]


async def main():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing")

    model_name = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=90.0,
        max_retries=1
    ) as client:

        llm = llm_factory(
            model_name,
            provider="openai",
            client=client,
            adapter="instructor",
            max_tokens=2048
        )

        metric = Faithfulness(llm=llm)

        results = []

        for case in TEST_CASES:
            print(f"\nEvaluating: {case['id']}")

            try:
                score_result = await metric.ascore(
                    user_input=case["question"],
                    response=case["answer"],
                    retrieved_contexts=case["contexts"]
                )

                score = float(score_result.value)

                if not math.isfinite(score):
                    raise ValueError(
                        "Ragas returned a non-finite score"
                    )

                record = {
                    "id": case["id"],
                    "expected": case["expected"],
                    "score": round(score, 4),
                    "status": "scored"
                }

            except Exception as exc:
                traceback.print_exc()
                record = {
                    "id": case["id"],
                    "expected": case["expected"],
                    "score": None,
                    "status": "error",
                    "error": str(exc)
                }

            print(record)
            results.append(record)

    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "judge_model": model_name,
        "results": results
    }

    output_path = (
        RESULTS_DIR / "faithfulness_smoke_test.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
