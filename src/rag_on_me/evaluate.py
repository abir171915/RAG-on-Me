"""Measure retrieval quality: do the expected sources show up in the top k?"""
from pathlib import Path
from typing import Any, Callable, Dict, List, Set

from dotenv import load_dotenv
from rag_on_me.embedding import EmbeddingPipeline
from rag_on_me.vector_store import VectorStore
from rag_on_me.search import Retriever
from rag_on_me.query_rewriter import QueryRewriter
from rag_on_me.eval_set import EVAL_SET

load_dotenv()

TOP_K = 5


def source_name(metadata: Dict[str, Any]) -> str:
    """Same naming the generator uses: repo name for GitHub, file name for PDFs."""
    return metadata.get("repo") or Path(metadata.get("source", "")).name


def known_sources(store: VectorStore) -> Set[str]:
    metadatas = store.collection.get(include=["metadatas"])["metadatas"]
    return {source_name(m) for m in metadatas}


def score(retrieved: List[str], expected: List[str]) -> Dict[str, float]:
    """hit: at least one expected source found. recall: fraction of expected sources found."""
    found = set(retrieved) & set(expected)
    return {"hit": float(bool(found)), "recall": len(found) / len(expected)}


def evaluate(name: str, retrieve: Callable[[str], List[Dict[str, Any]]]) -> Dict[str, float]:
    print(f"\n===== {name} (top {TOP_K})")
    totals = {"hit": 0.0, "recall": 0.0}

    for case in EVAL_SET:
        retrieved = [source_name(h["metadata"]) for h in retrieve(case["question"])]
        s = score(retrieved, case["expected"])
        totals["hit"] += s["hit"]
        totals["recall"] += s["recall"]
        print(f"  hit={s['hit']:.0f} recall={s['recall']:.2f} | {case['question']}")

    n = len(EVAL_SET)
    averages = {k: v / n for k, v in totals.items()}
    print(f"  --> hit rate {averages['hit']:.2f} | mean recall {averages['recall']:.2f}")
    return averages


if __name__ == "__main__":
    store = VectorStore()

    # Catch typos in expected source names before they silently score as misses
    known = known_sources(store)
    unknown = {s for case in EVAL_SET for s in case["expected"]} - known
    if unknown:
        raise SystemExit(f"Unknown sources in EVAL_SET: {unknown}\nKnown sources: {sorted(known)}")

    retriever = Retriever(EmbeddingPipeline(), store)
    rewriter = QueryRewriter()

    baseline = evaluate("baseline: original question only",
                        lambda q: retriever.search(q, top_k=TOP_K))
    rewritten = evaluate("rewriter: rewritten query",
                         lambda q: retriever.search_many(rewriter.rewrite(q), top_k=TOP_K))

    print(f"\nhit rate:    {baseline['hit']:.2f} -> {rewritten['hit']:.2f}")
    print(f"mean recall: {baseline['recall']:.2f} -> {rewritten['recall']:.2f}")
