"""
Benchmark Runner for Synthetic Complex EHR Retrieval
Tests BM25, Dense (base & fine-tuned), and Hybrid RRF across patients with 20 historical EHR claims.
"""

import os
import sys
import json
from typing import Dict, Any, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from api.retrieval.ehr_store import EHRStore
from api.retrieval.bm25_retriever import BM25Retriever
from api.retrieval.dense_retriever import DenseRetriever, DEFAULT_DENSE_MODEL, FINETUNED_DIR
from api.retrieval.hybrid import HybridRetriever

BENCHMARK_FILE = os.path.join(ROOT, "retrieval_data", "synthetic_complex_benchmark.json")
OUTPUT_REPORT = os.path.join(ROOT, "retrieval_data", "benchmark_results.json")


def run_benchmark():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"Error: {BENCHMARK_FILE} not found.")
        return

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Initialize EHRStore from the structured claims JSON
    store = EHRStore.from_claims_json(BENCHMARK_FILE)
    print("=" * 80)
    print(f"RUNNING: {data.get('benchmark_name')}")
    print("=" * 80)

    # Validate that every patient has 20 EHR claims
    for p in data["patients"]:
        pid = p["patient_id"]
        evs = store.events(pid)
        print(f"  [+] Patient {pid:<20} : {len(evs)} EHR claims loaded.")

    # 2. Setup Retrievers
    print("\n[+] Initializing Retrievers...")
    bm25 = BM25Retriever(store)
    dense_base = DenseRetriever(store, DEFAULT_DENSE_MODEL)
    
    dense_active = dense_base
    dense_name = "Dense (base)"
    if os.path.isdir(FINETUNED_DIR):
        dense_ft = DenseRetriever(store, FINETUNED_DIR)
        dense_active = dense_ft
        dense_name = "Dense (fine-tuned)"
    
    hybrid = HybridRetriever(bm25, dense_active)

    retrievers = {
        "BM25": lambda q, pid: [e for e, _ in bm25.search(q, pid, top_k=20)],
        dense_name: lambda q, pid: [e for e, _ in dense_active.search(q, pid, top_k=20)],
        "Hybrid RRF": lambda q, pid: [r[0] for r in hybrid.search_events(q, pid, top_k=20)]
    }

    # 3. Execute Benchmark Queries
    queries = data.get("test_queries", [])
    results_by_system = {k: [] for k in retrievers}
    detailed_logs = []

    print(f"\n[+] Executing {len(queries)} Complex Evaluation Queries over 20-EHR Histories...\n")
    print(f"{'QID':<14} | {'CATEGORY':<24} | {'TARGET':<8} | {'BM25':<6} | {'DENSE':<6} | {'HYBRID':<6} | {'DOSE_FLAG':<9}")
    print("-" * 85)

    for q in queries:
        qid = q["query_id"]
        pid = q["patient_id"]
        cat = q["category"]
        text = q["claim_text"]
        target = q["target_claim_id"]

        ranks = {}
        top1_hits = {}

        # Also get hybrid detailed output for dose agreement check
        hyb_details = hybrid.search(text, pid, top_k=20)
        dose_flag = "n/a"
        for item in hyb_details:
            if item["event"]["doc_id"] == target:
                dose_flag = item.get("dose_agreement", "n/a")
                break

        for name, fn in retrievers.items():
            ranked_events = fn(text, pid)
            doc_ids = [e.doc_id for e in ranked_events]
            
            rank = doc_ids.index(target) + 1 if target in doc_ids else 999
            ranks[name] = rank
            top1_hits[name] = (rank == 1)
            
            results_by_system[name].append({
                "query_id": qid,
                "category": cat,
                "rank": rank,
                "top1": rank == 1,
                "top3": rank <= 3,
                "top5": rank <= 5,
                "rr": 1.0 / rank if rank <= 20 else 0.0
            })

        log_entry = {
            "query_id": qid,
            "category": cat,
            "patient_id": pid,
            "claim_text": text,
            "target_claim_id": target,
            "ranks": ranks,
            "dose_agreement": dose_flag
        }
        detailed_logs.append(log_entry)

        bm_status = f"#{ranks['BM25']}"
        de_status = f"#{ranks[dense_name]}"
        hy_status = f"#{ranks['Hybrid RRF']}"
        print(f"{qid:<14} | {cat:<24} | {target:<8} | {bm_status:<6} | {de_status:<6} | {hy_status:<6} | {dose_flag:<9}")

    # 4. Compute Aggregate Metrics
    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE RESULTS (POOL SIZE = 20 EHR CLAIMS PER PATIENT)")
    print("=" * 80)
    print(f"{'SYSTEM':<20} | {'TOP-1 ACC':<10} | {'TOP-3 ACC':<10} | {'TOP-5 ACC':<10} | {'MRR':<10}")
    print("-" * 68)

    summary_metrics = {}
    for name, res_list in results_by_system.items():
        n = len(res_list)
        top1 = sum(r["top1"] for r in res_list) / n
        top3 = sum(r["top3"] for r in res_list) / n
        top5 = sum(r["top5"] for r in res_list) / n
        mrr = sum(r["rr"] for r in res_list) / n

        summary_metrics[name] = {
            "top1_acc": round(top1, 4),
            "top3_acc": round(top3, 4),
            "top5_acc": round(top5, 4),
            "mrr": round(mrr, 4)
        }
        print(f"{name:<20} | {top1 * 100:>8.1f}% | {top3 * 100:>8.1f}% | {top5 * 100:>8.1f}% | {mrr:>10.4f}")

    # 5. Category Breakdown for Hybrid RRF
    print("\n" + "-" * 80)
    print("HYBRID RRF PERFORMANCE BREAKDOWN BY CHALLENGE CATEGORY:")
    print("-" * 80)
    categories = sorted(list(set(q["category"] for q in queries)))
    for cat in categories:
        cat_items = [r for r in results_by_system["Hybrid RRF"] if r["category"] == cat]
        cat_top1 = sum(r["top1"] for r in cat_items)
        print(f"  - {cat:<28} : {cat_top1}/{len(cat_items)} correct Top-1 ({cat_top1/len(cat_items)*100:.0f}%)")

    # 6. Save JSON Report
    report_data = {
        "benchmark_file": BENCHMARK_FILE,
        "total_queries": len(queries),
        "ehr_pool_per_patient": 20,
        "summary_metrics": summary_metrics,
        "detailed_query_results": detailed_logs
    }
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[+] Full diagnostic report saved to: {OUTPUT_REPORT}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
