
from functools import lru_cache
from typing import TypedDict
import re
from decimal import Decimal
from app.tools.sec_comparison import compare_sec_financials
from langgraph.graph import StateGraph, START, END
from app.agents.planner import plan_query, document_year
from app.retrieval.bm25_search import get_all_chunks
from app.retrieval.hybrid_search import hybrid_search
from app.retrieval.reranker import rerank_chunks
from app.generation.answer import generate_answer


class AgentState(TypedDict):
    query: str
    limit: int
    attempts: int
    plan: dict
    chunks: list[dict]
    result: dict


def plan_node(state: AgentState) -> dict:
    plan = plan_query(state["query"])

    print("[LangGraph] Required years:",
          plan["required_years"])

    return {"plan": plan}


def route_after_plan(state: AgentState) -> str:
    query = state["query"].lower()
    years = state["plan"]["required_years"]

    is_apple = bool(re.search(r"\bapple\b", query))

    is_operating_expenses = bool(
        re.search(r"\boperating expenses\b", query)
    )

    is_comparison = (
        len(years) >= 2
        and any(
            word in query
            for word in [
                "change",
                "compare",
                "increase",
                "decrease",
                "difference"
            ]
        )
    )

    if (
        is_apple
        and is_operating_expenses
        and is_comparison
    ):
        return "financial"

    return "retrieve"

def retrieve_node(state: AgentState) -> dict:
    attempt = state["attempts"] + 1

    print(f"[LangGraph] Retrieval attempt {attempt}")

    plan = state["plan"]
    required_years = plan["required_years"]

    # Determine which years are actually indexed
    indexed_chunks = get_all_chunks()
    available_years = {
        document_year(chunk["document"])
        for chunk in indexed_chunks
    }

    missing_years = [
        year for year in required_years
        if year not in available_years
    ]

    if missing_years:
        return {
            "attempts": attempt,
            "chunks": [],
            "result": {
                "answer": (
                    "Cannot complete the comparison. "
                    "Missing indexed filings for: "
                    + ", ".join(map(str, missing_years))
                ),
                "sources": [],
                "status": "missing_documents",
                "missing_years": missing_years
            }
        }

    all_chunks = []
    seen_ids = set()
    years_without_results = []

    # Retrieve evidence for each planned reporting year
    for task in plan["subqueries"]:
        year = task["year"]

        candidates = hybrid_search(
            query=task["query"],
            limit=100
        )

        # Prevent using evidence from the wrong filing year
        if year is not None:
            candidates = [
                chunk for chunk in candidates
                if document_year(chunk["document"]) == year
            ]

        if not candidates:
            years_without_results.append(year)
            continue

        # Rerank only candidates for this task
        ranked = rerank_chunks(
            query=task["query"],
            chunks=candidates[:30],
            limit=min(
                state["limit"],
                len(candidates)
            )
        )

        for chunk in ranked:
            chunk_id = chunk["chunk_id"]

            if chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk_id)

    if years_without_results or not all_chunks:
        return {
            "attempts": attempt,
            "chunks": [],
            "result": {
                "answer": (
                    "The required filings may be indexed, "
                    "but sufficient passages could not be "
                    "retrieved for every requested year."
                ),
                "sources": [],
                "status": "insufficient_evidence",
                "missing_evidence_years": years_without_results
            }
        }

    # The second attempt includes more candidates
    max_chunks = min(
        len(all_chunks),
        max(10, len(plan["subqueries"]) * state["limit"])
    )

    return {
        "attempts": attempt,
        "chunks": all_chunks[:max_chunks],
        "result": {}
    }


def generate_node(state: AgentState) -> dict:
    print("[LangGraph] Generating and verifying answer")

    result = generate_answer(
        query=state["query"],
        limit=state["limit"],
        chunks=state["chunks"]
    )

    return {"result": result}


def route_after_retrieval(state: AgentState) -> str:
    if not state["chunks"]:
        return "finish"

    return "generate"


def route_after_generation(state: AgentState) -> str:
    status = state["result"].get("status")

    if status == "answered":
        return "finish"

    if state["attempts"] >= 2:
        return "finish"

    if status in {
        "insufficient_evidence",
        "verification_failed",
        "citation_validation_failed"
    }:
        return "retry"

    return "finish"



@lru_cache(maxsize=1)
def get_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("plan", plan_node)
    workflow.add_node("financial", financial_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "plan")

    workflow.add_conditional_edges(
        "plan",
        route_after_plan,
        {
            "financial": "financial",
            "retrieve": "retrieve"
        }
    )

    workflow.add_edge("financial", END)

    workflow.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "finish": END
        }
    )

    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "retry": "retrieve",
            "finish": END
        }
    )

    return workflow.compile()



def run_agent(query: str, limit: int = 5) -> dict:
    if not query.strip():
        raise ValueError("Question cannot be empty")

    if not 1 <= limit <= 10:
        raise ValueError("Limit must be between 1 and 10")

    initial_state = {
        "query": query,
        "limit": limit,
        "attempts": 0,
        "plan": {},
        "chunks": [],
        "result": {}
    }

    final_state = get_graph().invoke(
        initial_state,
        config={"recursion_limit": 10}
    )

    return {
        "query": query,
        "attempts": final_state["attempts"],
        "required_years": (
            final_state["plan"]["required_years"]
        ),
        **final_state["result"]
    }

def financial_node(state: AgentState) -> dict:
    print("[LangGraph] Running SEC financial comparison")

    years = state["plan"]["required_years"]

    start_year = min(years)
    end_year = max(years)

    comparison = compare_sec_financials(
        cik="0000320193",
        metric="operating_expenses",
        start_year=start_year,
        end_year=end_year
    )

    million = Decimal("1000000")

    start_value = Decimal(comparison["start_value"])
    end_value = Decimal(comparison["end_value"])
    change = Decimal(comparison["absolute_change"])

    start_millions = start_value / million
    end_millions = end_value / million
    change_millions = change / million

    percentage = comparison["percentage_change"]

    if change > 0:
        direction = "increased"
    elif change < 0:
        direction = "decreased"
    else:
        direction = "remained unchanged"

    answer = (
        f"Apple's operating expenses {direction} "
        f"from ${start_millions:,.0f} million "
        f"in {start_year} [1] "
        f"to ${end_millions:,.0f} million "
        f"in {end_year} [2]. "
        f"The absolute change was "
        f"${abs(change_millions):,.0f} million."
    )

    if percentage is not None:
        answer += (
            f" This represents a "
            f"{abs(Decimal(percentage)):.2f}% "
            f"{'increase' if change > 0 else 'decrease' if change < 0 else 'change'}."
        )

    return {
        "result": {
            "answer": answer,
            "status": "answered",
            "calculation": comparison,
            "sources": [
                {
                    "citation_id": str(i),
                    **source
                }
                for i, source in enumerate(
                    comparison["sources"], start=1
                )
            ],
            "answer_type": "deterministic_financial_comparison"
        }
    }
