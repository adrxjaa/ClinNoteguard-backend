"""
Demo: claim -> hybrid (BM25 + Dense + RRF) evidence retrieval from a patient's EHR history.

Usage: ./venv/bin/python retrieval_training/demo_retrieval.py [encounter_id=D2N001] [split=train]
Shows each EHR-supported claim of that encounter with BM25 / Dense / fused ranks.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from api.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from api.retrieval.dense_retriever import DenseRetriever  # noqa: E402
from api.retrieval.ehr_store import EHRStore  # noqa: E402
from api.retrieval.hybrid import HybridRetriever  # noqa: E402


def main():
    enc = sys.argv[1] if len(sys.argv) > 1 else "D2N001"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    store = EHRStore.from_csv([split])
    pid = EHRStore.patient_id_for_encounter(enc)
    print(f"Patient {pid}: {len(store.events(pid))} EHR events")
    for e in store.events(pid):
        print(f"   - [{e.relative_time}] {e.description}")

    dense = DenseRetriever(store)
    print(f"\nDense encoder: {dense.model_name}")
    hybrid = HybridRetriever(BM25Retriever(store), dense)

    with open(os.path.join(ROOT, "retrieval_data", f"{split}_claims.jsonl")) as f:
        claims = [json.loads(l) for l in f if json.loads(l)["encounter_id"] == enc][:5]
    out = []
    for c in claims:
        res = hybrid.search(c["claim_text"], pid, top_k=3)
        print(f"\nCLAIM {c['claim_id']} [{c['section']}]: {c['claim_text'][:160]}")
        for r in res:
            print(f"   rrf={r['rrf_score']:.4f} bm25#{r['bm25_rank']} dense#{r['dense_rank']}  {r['text']}")
        out.append({"claim": c["claim_text"], "candidates": res})
    with open(os.path.join(ROOT, "demo_retrieval_output.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
