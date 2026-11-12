"""Tiny evaluation harness. Usage: python eval.py eval_set.json doc1.pdf [doc2.txt ...]

eval_set.json: [{"q": "question", "expect": "keyword that must appear in the answer"}, ...]
"""
import json
import sys

from agent import build_graph
from ingest import Index

if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    idx = Index()
    idx.add_files(sys.argv[2:])
    graph = build_graph(idx)
    passed = 0
    for c in cases:
        out = graph.invoke({"question": c["q"], "history": [], "attempts": 0})
        ok = c["expect"].lower() in out["answer"].lower()
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['q']}")
    print(f"\n{passed}/{len(cases)} passed")
