"""
Benchmark Runner for Synthetic Complex EHR Retrieval
Tests BM25, Dense (base & fine-tuned), Hybrid RRF, and Cross-Encoder Reranker
across patients with 20 historical EHR claims.
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
from api.retrieval.reranker import CrossEncoderReranker

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

    # 2. Setup Retrievers & Cross-Encoder Reranker
    print("\n[+] Initializing Retrievers and Reranker...")
    bm25 = BM25Retriever(store)
    
    dense_active = None
    dense_name = "Dense (base)"
    dense_error = None
    try:
        dense_base = DenseRetriever(store, DEFAULT_DENSE_MODEL)
        dense_active = dense_base
        if os.path.isdir(FINETUNED_DIR):
            dense_ft = DenseRetriever(store, FINETUNED_DIR)
            dense_active = dense_ft
            dense_name = "Dense (fine-tuned)"
        print(f"  [+] {dense_name} initialized.")
    except Exception as e:
        dense_error = str(e)
        print(f"  [-] Dense retriever unavailable ({e}).")

    hybrid = None
    if dense_active is not None:
        hybrid = HybridRetriever(bm25, dense_active)
        print("  [+] HybridRetriever (BM25 + Dense + RRF) initialized.")
    else:
        print("  [-] HybridRetriever unavailable because Dense retriever could not be initialized.")

    reranker = None
    reranker_error = None
    try:
        reranker = CrossEncoderReranker()
        print(f"  [+] CrossEncoderReranker initialized ({reranker.model_name}).")
    except Exception as e:
        reranker_error = str(e)
        print(f"  [-] CrossEncoderReranker unavailable ({e}).")

    retrievers = {
        "BM25": lambda q, pid: [e for e, _ in bm25.search(q, pid, top_k=20)]
    }

    if dense_active is not None:
        retrievers[dense_name] = lambda q, pid: [e for e, _ in dense_active.search(q, pid, top_k=20)]

    if hybrid is not None:
        retrievers["Hybrid RRF"] = lambda q, pid: [r[0] for r in hybrid.search_events(q, pid, top_k=20)]

    # Add Cross-Encoder Reranking path: takes top-20 candidate pool from Hybrid RRF and reranks with CrossEncoder
    if hybrid is not None and reranker is not None:
        def cross_encoder_search(q, pid):
            # 1. Retrieve top-20 candidate pool via Hybrid RRF
            candidate_pool = hybrid.search(q, pid, top_k=20)
            if not candidate_pool:
                return []
            # 2. Rerank candidate pool with CrossEncoder
            reranked_pool = reranker.rerank(q, candidate_pool, top_k=20)
            # 3. Map back to EHREvent objects for unified benchmark evaluation
            events_by_id = {e.doc_id: e for e in store.events(pid)}
            return [events_by_id[item["event"]["doc_id"]] for item in reranked_pool if item["event"]["doc_id"] in events_by_id]

        retrievers["Hybrid RRF + Cross-Encoder"] = cross_encoder_search

    # 3. Execute Benchmark Queries
    queries = data.get("test_queries", [])
    results_by_system = {k: [] for k in retrievers}
    detailed_logs = []

    print(f"\n[+] Executing {len(queries)} Complex Evaluation Queries over 20-EHR Histories...\n")
    hdr_retrievers = " | ".join(f"{n[:8]:<8}" for n in retrievers)
    print(f"{'QID':<14} | {'CATEGORY':<24} | {'TARGET':<8} | {hdr_retrievers} | {'DOSE_FLAG':<9}")
    print("-" * (62 + len(retrievers) * 11))

    for q in queries:
        qid = q["query_id"]
        pid = q["patient_id"]
        cat = q["category"]
        text = q["claim_text"]
        target = q["target_claim_id"]

        ranks = {}
        top1_hits = {}

        # Also get hybrid detailed output for dose agreement check
        dose_flag = "n/a"
        if hybrid is not None:
            hyb_details = hybrid.search(text, pid, top_k=20)
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

        status_strs = " | ".join(f"{f'#{ranks[n]}':<8}" for n in retrievers)
        print(f"{qid:<14} | {cat:<24} | {target:<8} | {status_strs} | {dose_flag:<9}")

    # 4. Compute Aggregate Metrics
    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE RESULTS (POOL SIZE = 20 EHR CLAIMS PER PATIENT)")
    print("=" * 80)
    print(f"{'SYSTEM':<28} | {'TOP-1 ACC':<10} | {'TOP-3 ACC':<10} | {'TOP-5 ACC':<10} | {'MRR':<10}")
    print("-" * 76)

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
        print(f"{name:<28} | {top1 * 100:>8.1f}% | {top3 * 100:>8.1f}% | {top5 * 100:>8.1f}% | {mrr:>10.4f}")

    # 5. Category Breakdown: Hybrid RRF vs Cross-Encoder
    categories = sorted(list(set(q["category"] for q in queries)))
    if "Hybrid RRF + Cross-Encoder" in results_by_system and "Hybrid RRF" in results_by_system:
        print("\n" + "-" * 80)
        print("PERFORMANCE BREAKDOWN BY CHALLENGE CATEGORY (HYBRID RRF vs CROSS-ENCODER):")
        print("-" * 80)
        print(f"{'CATEGORY':<28} | {'HYBRID RRF':<15} | {'RRF + CROSS-ENCODER':<20}")
        print("-" * 70)
        for cat in categories:
            hyb_items = [r for r in results_by_system["Hybrid RRF"] if r["category"] == cat]
            ce_items = [r for r in results_by_system["Hybrid RRF + Cross-Encoder"] if r["category"] == cat]
            hyb_top1 = sum(r["top1"] for r in hyb_items)
            ce_top1 = sum(r["top1"] for r in ce_items)
            hyb_str = f"{hyb_top1}/{len(hyb_items)} ({hyb_top1/len(hyb_items)*100:.0f}%)"
            ce_str = f"{ce_top1}/{len(ce_items)} ({ce_top1/len(ce_items)*100:.0f}%)"
            print(f"{cat:<28} | {hyb_str:<15} | {ce_str:<20}")
    elif "Hybrid RRF" in results_by_system:
        print("\n" + "-" * 80)
        print("HYBRID RRF PERFORMANCE BREAKDOWN BY CHALLENGE CATEGORY:")
        print("-" * 80)
        for cat in categories:
            cat_items = [r for r in results_by_system["Hybrid RRF"] if r["category"] == cat]
            cat_top1 = sum(r["top1"] for r in cat_items)
            print(f"  - {cat:<28} : {cat_top1}/{len(cat_items)} correct Top-1 ({cat_top1/len(cat_items)*100:.0f}%)")

    # 6. Status Report for Unavailable Systems
    if reranker is None:
        print("\n" + "!" * 80)
        print("[!] NOTICE: Cross-Encoder Reranker evaluation could not run.")
        print(f"    Reason: {reranker_error}")
        print("    Stage 3 baseline (BM25 / Dense / Hybrid RRF) was evaluated where dependencies permitted,")
        print("    but Stage 4 Cross-Encoder requires `sentence-transformers` and model weights to be installed.")
        print("!" * 80)

    # 7. Save JSON Report (only when full benchmark runs to preserve baseline report)
    if reranker is not None:
        report_data = {
            "benchmark_file": BENCHMARK_FILE,
            "total_queries": len(queries),
            "ehr_pool_per_patient": 20,
            "summary_metrics": summary_metrics,
            "detailed_query_results": detailed_logs,
            "cross_encoder_status": "evaluated"
        }
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[+] Full diagnostic report saved to: {OUTPUT_REPORT}")
    else:
        print(f"\n[i] Preserving existing baseline report at: {OUTPUT_REPORT} (Cross-Encoder did not run).")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
