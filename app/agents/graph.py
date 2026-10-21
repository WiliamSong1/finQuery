
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.retrieval.hybrid_search import reranked_hybrid_search
from app.generation.answer import generate_answer


class AgentState(TypedDict):
    query: str
    limit: int
    attempts: int
    chunks: list[dict]
    result: dict


def retrieve_node(state: AgentState) -> dict:
    attempt = state["attempts"] + 1

    # Broaden retrieval on the second attempt
    limit = (
        state["limit"]
        if attempt == 1
        else min(state["limit"] * 2, 10)
    )

    print(f"[LangGraph] Retrieval attempt {attempt}")

    chunks = reranked_hybrid_search(
        query=state["query"],
        limit=limit,
        candidate_limit=max(20, limit * 4)
    )

    return {
        "chunks": chunks,
        "attempts": attempt
    }


def generate_node(state: AgentState) -> dict:
    print("[LangGraph] Generating and verifying answer")

    result = generate_answer(
        query=state["query"],
        limit=state["limit"],
        chunks=state["chunks"]
    )

    return {"result": result}


def route_answer(state: AgentState) -> str:
    result = state["result"]

    if result.get("status") == "answered":
        return "finish"

    if state["attempts"] >= 2:
        return "finish"

    if result.get("status") in {
        "insufficient_evidence",
        "verification_failed",
        "citation_validation_failed"
    }:
        return "retry"

    return "finish"


@lru_cache(maxsize=1)
def get_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")

    workflow.add_conditional_edges(
        "generate",
        route_answer,
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
        **final_state["result"]
    }
