"""LangGraph agent: rewrite -> retrieve -> grade -> (retry once) -> answer with citations."""
import os
import time
from typing import List, TypedDict

from langgraph.graph import END, StateGraph
from openai import APIError, OpenAI, RateLimitError

from ingest import Chunk, Index


def _make_client(prefix: str = ""):
    base = os.environ.get(f"{prefix}LLM_BASE_URL")
    key = os.environ.get(f"{prefix}LLM_API_KEY")
    model = os.environ.get(f"{prefix}LLM_MODEL")
    if not (base and key and model):
        return None
    return OpenAI(base_url=base, api_key=key), model


def llm(messages, max_tokens: int = 700, temperature: float = 0.0) -> str:
    """Call the primary provider, retry on rate limits, then fall back to FALLBACK_* if configured."""
    providers = [p for p in (_make_client(""), _make_client("FALLBACK_")) if p]
    if not providers:
        raise RuntimeError("Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL.")
    last = None
    for client, model in providers:
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
                )
                return (r.choices[0].message.content or "").strip()
            except RateLimitError as e:
                last = e
                time.sleep(2**attempt)
            except APIError as e:
                last = e
                break
    raise RuntimeError(f"All LLM providers failed: {last}")


class State(TypedDict, total=False):
    question: str
    history: list
    query: str
    attempts: int
    chunks: List[Chunk]
    relevant: bool
    answer: str
    sources: List[str]


def _fmt_context(chunks: List[Chunk]) -> str:
    return "\n\n".join(f"[{c.source}, p.{c.page}]\n{c.text}" for c in chunks)


def build_graph(index: Index):
    def rewrite(state: State):
        attempts = state.get("attempts", 0)
        if attempts == 0:
            hist = "\n".join(f"{m['role']}: {m['content']}" for m in (state.get("history") or [])[-6:])
            prompt = (
                "Rewrite the user's last question as a standalone search query using the chat history "
                "for context. Output only the query.\n\n"
                f"History:\n{hist or '(none)'}\n\nQuestion: {state['question']}"
            )
        else:
            prompt = (
                "A document search for the query below returned irrelevant results. Write a different "
                "query using alternative keywords or synonyms. Output only the query.\n\n"
                f"Original question: {state['question']}\nFailed query: {state['query']}"
            )
        return {"query": llm([{"role": "user", "content": prompt}], max_tokens=60), "attempts": attempts + 1}

    def retrieve(state: State):
        return {"chunks": index.search(state["query"], k=6)}

    def grade(state: State):
        if not state["chunks"]:
            return {"relevant": False}
        prompt = (
            "Do the excerpts contain enough information to answer the question? Reply with exactly YES or NO.\n\n"
            f"Question: {state['question']}\n\nExcerpts:\n{_fmt_context(state['chunks'])}"
        )
        verdict = llm([{"role": "user", "content": prompt}], max_tokens=5)
        return {"relevant": verdict.upper().startswith("YES")}

    def route(state: State):
        if state["relevant"]:
            return "answer"
        return "retry" if state["attempts"] < 2 else "none"

    def answer(state: State):
        system = (
            "Answer the question using ONLY the excerpts. Cite sources inline as [file, p.N]. "
            "If the excerpts do not contain the answer, say you could not find it in the documents."
        )
        user = f"Excerpts:\n{_fmt_context(state['chunks'])}\n\nQuestion: {state['question']}"
        text = llm([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=800)
        srcs = sorted({f"{c.source} (p.{c.page})" for c in state["chunks"]})
        return {"answer": text, "sources": srcs}

    def not_found(state: State):
        return {
            "answer": "I couldn't find this in the uploaded documents. Try rephrasing, or upload a file that covers it.",
            "sources": [],
        }

    g = StateGraph(State)
    g.add_node("rewrite", rewrite)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("answer", answer)
    g.add_node("not_found", not_found)
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route, {"retry": "rewrite", "answer": "answer", "none": "not_found"})
    g.add_edge("answer", END)
    g.add_edge("not_found", END)
    return g.compile()
