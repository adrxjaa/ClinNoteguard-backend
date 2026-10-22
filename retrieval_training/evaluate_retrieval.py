"""
Evaluate BM25 vs Dense (base) vs Dense (fine-tuned) vs Hybrid RRF on silver-labelled claims.

Metrics (per claim, patient-scoped candidate set): Recall@1/3/5, MRR, nDCG@5.
NOTE: labels are SILVER (rule-based), so absolute numbers are optimistic for lexical methods;
use them to compare configurations, not as final accuracy.

Usage: ./venv/bin/python retrieval_training/evaluate_retrieval.py [--splits valid test1 test2 test3]
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from api.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from api.retrieval.dense_retriever import DEFAULT_DENSE_MODEL, FINETUNED_DIR, DenseRetriever  # noqa: E402
from api.retrieval.ehr_store import EHRStore  # noqa: E402
from api.retrieval.hybrid import HybridRetriever  # noqa: E402

DATA_DIR = os.path.join(ROOT, "retrieval_data")


def metrics(ranked_ids, positives):
    pos = set(positives)
    out = {}
    for k in (1, 3, 5):
        out[f"R@{k}"] = len(pos & set(ranked_ids[:k])) / len(pos)
    rr = 0.0
    for i, d in enumerate(ranked_ids, 1):
        if d in pos:
            rr = 1.0 / i
            break
    out["MRR"] = rr
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked_ids[:5], 1) if d in pos)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(pos), 5) + 1))
    out["nDCG@5"] = dcg / idcg
    return out


def evaluate(split, systems):
    store = EHRStore.from_csv([split])
    with open(os.path.join(DATA_DIR, f"{split}_claims.jsonl")) as f:
        rows = [json.loads(l) for l in f]
    if not rows:
        return None, 0
    agg = {name: {} for name in systems}
    for name, make in systems.items():
        search = make(store)
        for r in rows:
            ids = [e.doc_id for e in search(r["claim_text"], r["patient_id"])]
            for k, v in metrics(ids, r["positives"]).items():
                agg[name][k] = agg[name].get(k, 0.0) + v
        for k in agg[name]:
            agg[name][k] /= len(rows)
    return agg, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["valid", "test1", "test2", "test3"])
    args = ap.parse_args()

    def bm25(store):
        r = BM25Retriever(store)
        return lambda c, p: [e for e, _ in r.search(c, p, 20)]

    def dense_factory(path):
        def make(store):
            r = DenseRetriever(store, path)
            return lambda c, p: [e for e, _ in r.search(c, p, 20)]
        return make

    def hybrid_factory(path):
        def make(store):
            h = HybridRetriever(BM25Retriever(store), DenseRetriever(store, path))
            return lambda c, p: [r[0] for r in h.search_events(c, p, 20)]
        return make

    systems = {"BM25": bm25, "Dense (base)": dense_factory(DEFAULT_DENSE_MODEL)}
    if os.path.isdir(FINETUNED_DIR):
        systems["Dense (fine-tuned)"] = dense_factory(FINETUNED_DIR)
        systems["Hybrid RRF (BM25+FT)"] = hybrid_factory(FINETUNED_DIR)
    else:
        systems["Hybrid RRF (BM25+base)"] = hybrid_factory(DEFAULT_DENSE_MODEL)

    for split in args.splits:
        agg, n = evaluate(split, systems)
        print(f"\n=== {split}  ({n} claims with EHR evidence) ===")
        if not agg:
            continue
        cols = ["R@1", "R@3", "R@5", "MRR", "nDCG@5"]
        print(f"{'system':<26}" + "".join(f"{c:>9}" for c in cols))
        for name, m in agg.items():
            print(f"{name:<26}" + "".join(f"{m[c]:>9.3f}" for c in cols))


if __name__ == "__main__":
    main()
