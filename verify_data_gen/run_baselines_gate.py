import os
import sys
from pathlib import Path

# Ensure backend root is on sys.path
root_dir = str(Path(__file__).parent.parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import logging
from typing import Dict, List, Any
import torch

from verify.eval.metrics import compute_verification_metrics, VERIFICATION_CLASSES
from verify.model.baselines import (
    MajorityClassBaseline,
    BaselineFeatureExtractor,
    train_mlp_baseline,
)
from verify.graph.transcript_store import TranscriptStore
from api.retrieval.text_utils import dose_agreement, tokenize_clinical
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BaselinesGate")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def build_evidence_pool_for_claims(
    claims: List[Dict[str, Any]],
    rendered_ehr: List[Dict[str, Any]],
    transcript_store: TranscriptStore,
    top_k: int = 10,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves top candidate evidence (EHR + transcript) for each claim.
    """
    # Group EHR records by patient_id
    ehr_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for r in rendered_ehr:
        pid = r.get("patient_id", "")
        ehr_by_pid.setdefault(pid, []).append(r)

    # Pre-index BM25 for each patient's EHR
    ehr_indices: Dict[str, BM25Okapi] = {}
    for pid, records in ehr_by_pid.items():
        corpus = [tokenize_clinical(r.get("claim_text", "")) for r in records]
        ehr_indices[pid] = BM25Okapi(corpus)

    candidates_by_claim: Dict[str, List[Dict[str, Any]]] = {}

    for c in claims:
        cid = c["claim_id"]
        pid = c.get("patient_id", "")
        enc_id = c.get("encounter_id", "")
        claim_text = c.get("claim_text", "")

        cands = []
        # 1. Search patient EHR
        if pid in ehr_indices:
            bm25 = ehr_indices[pid]
            toks = tokenize_clinical(claim_text)
            scores = bm25.get_scores(toks)
            records = ehr_by_pid[pid]
            for r, s in zip(records, scores):
                cands.append({
                    "record_id": r.get("record_id"),
                    "claim_text": r.get("claim_text"),
                    "source": "EHR",
                    "valid_from": r.get("valid_from"),
                    "timestamp": r.get("valid_from"),
                    "score": float(s),
                    "dose_agreement": dose_agreement(claim_text, r.get("claim_text", "")),
                })

        # 2. Search transcript
        trans_cands = transcript_store.search(claim_text, enc_id, top_k=5)
        for tc in trans_cands:
            tc["dose_agreement"] = dose_agreement(claim_text, tc.get("claim_text", ""))
            cands.append(tc)

        # Sort combined candidates by score descending
        cands.sort(key=lambda x: x["score"], reverse=True)
        candidates_by_claim[cid] = cands[:top_k]

    return candidates_by_claim

def run_gate_evaluation():
    print("[+] Loading claims and EHR data...")
    train_claims = load_jsonl("verify_data/claims_train.jsonl")
    val_claims = load_jsonl("verify_data/claims_val.jsonl")
    test_claims = load_jsonl("verify_data/claims_test.jsonl")
    rendered_ehr = load_jsonl("verify_data/rendered_ehr.jsonl")

    # Ingest dialogues into TranscriptStore
    print("[+] Ingesting dialogues into TranscriptStore...")
    transcript_store = TranscriptStore()
    from verify_data_gen.generate_verification_data import load_aci_dialogues_and_notes
    aci_data = load_aci_dialogues_and_notes()
    for enc_id, d in aci_data.items():
        transcript_store.ingest_dialogue(enc_id, d.get("dialogue", ""))

    print("[+] Retrieving candidate evidence pools for all splits...")
    cands_train = build_evidence_pool_for_claims(train_claims, rendered_ehr, transcript_store)
    cands_val = build_evidence_pool_for_claims(val_claims, rendered_ehr, transcript_store)
    cands_test = build_evidence_pool_for_claims(test_claims, rendered_ehr, transcript_store)

    y_train = [c["label"] for c in train_claims]
    y_val = [c["label"] for c in val_claims]
    y_test = [c["label"] for c in test_claims]

    results = {}

    # -------------------------------------------------------------
    # Baseline A0: Majority Class Predictor
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE A0: Majority Class Predictor")
    print("=" * 60)
    a0 = MajorityClassBaseline()
    a0.fit(y_train)
    a0_preds_test = a0.predict(len(y_test))
    m_a0 = compute_verification_metrics(y_test, a0_preds_test)
    results["A0"] = m_a0
    print(f"A0 Test Accuracy : {m_a0['accuracy']:.4f}")
    print(f"A0 Test Macro-F1 : {m_a0['macro_f1']:.4f}")

    # -------------------------------------------------------------
    # Baseline A1: MLP on [claim emb, top-1 evidence emb, ce_score, dose tag]
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE A1: MLP Classifier")
    print("=" * 60)
    fe = BaselineFeatureExtractor()
    print("  Extracting features for Train, Val, Test...")
    X_train_a1, _ = fe.extract_features(train_claims, cands_train, recency_only=False)
    X_val_a1, _ = fe.extract_features(val_claims, cands_val, recency_only=False)
    X_test_a1, _ = fe.extract_features(test_claims, cands_test, recency_only=False)

    model_a1, val_m_a1 = train_mlp_baseline(X_train_a1, y_train, X_val_a1, y_val, epochs=20, lr=3e-4)

    # Test evaluation
    idx2label = {i: l for i, l in enumerate(VERIFICATION_CLASSES)}
    dev = next(model_a1.parameters()).device
    model_a1.eval()
    with torch.no_grad():
        test_out_a1 = model_a1(X_test_a1.to(dev))
        test_preds_a1 = [idx2label[int(idx)] for idx in test_out_a1.argmax(dim=-1).cpu().numpy()]
        m_a1 = compute_verification_metrics(y_test, test_preds_a1)
        results["A1"] = m_a1

    print(f"A1 Test Accuracy : {m_a1['accuracy']:.4f}")
    print(f"A1 Test Macro-F1 : {m_a1['macro_f1']:.4f}")
    print(f"A1 Stale-as-Supported: {m_a1['stale_as_supported_rate']:.4f}")

    # -------------------------------------------------------------
    # Baseline A2: A1 with recency_only filter
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE A2: MLP with Recency-Only Filter")
    print("=" * 60)
    print("  Extracting features with recency_only=True...")
    X_train_a2, _ = fe.extract_features(train_claims, cands_train, recency_only=True)
    X_val_a2, _ = fe.extract_features(val_claims, cands_val, recency_only=True)
    X_test_a2, _ = fe.extract_features(test_claims, cands_test, recency_only=True)

    model_a2, val_m_a2 = train_mlp_baseline(X_train_a2, y_train, X_val_a2, y_val, epochs=20, lr=3e-4)

    model_a2.eval()
    with torch.no_grad():
        test_out_a2 = model_a2(X_test_a2.to(dev))
        test_preds_a2 = [idx2label[int(idx)] for idx in test_out_a2.argmax(dim=-1).cpu().numpy()]
        m_a2 = compute_verification_metrics(y_test, test_preds_a2)
        results["A2"] = m_a2

    print(f"A2 Test Accuracy : {m_a2['accuracy']:.4f}")
    print(f"A2 Test Macro-F1 : {m_a2['macro_f1']:.4f}")
    print(f"A2 Stale-as-Supported: {m_a2['stale_as_supported_rate']:.4f}")

    # -------------------------------------------------------------
    # Gate 9 Verification Check
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("GATE 9 AUDIT & LEAKAGE CHECK:")
    print("=" * 60)
    f1_a0 = m_a0["macro_f1"]
    f1_a1 = m_a1["macro_f1"]
    f1_a2 = m_a2["macro_f1"]

    print(f"A0 (Majority Class) Macro-F1 : {f1_a0:.4f}")
    print(f"A1 (MLP Standard)   Macro-F1 : {f1_a1:.4f}")
    print(f"A2 (MLP Recency)    Macro-F1 : {f1_a2:.4f}")

    if f1_a1 > 0.95 or f1_a2 > 0.95:
        print("[!] WARNING: A1 or A2 exceeds 0.95 Macro-F1! Dataset may be too easy or leaky.")
    else:
        print("[✓] PASS: Neither A1 nor A2 exceeds 0.95 Macro-F1 (Dataset is challenging and non-trivial).")

    # Check that A1 / A2 requires deeper graph/temporal reasoning
    print("\nPer-Class F1 Breakdown for A1:")
    for lbl, stats in m_a1["per_class"].items():
        print(f"  {lbl:22s} -> F1: {stats['f1']:.4f} (P: {stats['precision']:.4f}, R: {stats['recall']:.4f})")

    # Save results to results/baselines_gate.json
    import os
    os.makedirs("results", exist_ok=True)
    with open("results/baselines_gate.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n[+] Gate results saved to results/baselines_gate.json")

if __name__ == "__main__":
    run_gate_evaluation()
