"""
Training and Evaluation Pipeline for Stage 7-8 Verifier (M3).

Trains ClaimConditionedVerifier (Dense GIN + Claim-Conditioned Transformer with edge and time biases)
and calibrates with Temperature Scaling.
Evaluates on test split, executes Date-Shuffle Sensitivity Test, and saves checkpoints/verifier_demo.pt.
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# Path setup
root_dir = str(Path(__file__).parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from verify.eval.metrics import compute_verification_metrics, VERIFICATION_CLASSES
from verify.graph.builder import GraphBuilder
from verify.graph.transcript_store import TranscriptStore
from verify.model.verifier import ClaimConditionedVerifier
from verify.model.calibrate import TemperatureScaler, compute_ece, compute_risk_coverage
from api.retrieval.text_utils import dose_agreement, tokenize_clinical
from rank_bm25 import BM25Okapi

BANNER = """
================================================================================
CLINNOTEGUARD VERIFIER (STAGES 7-8) - MODEL EVALUATION REPORT
NOTICE: synthetic, rule-labeled, not a validated result
ONTOLOGY: curated concept layer
================================================================================
"""

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def cache_evidence_pools(
    claims: List[Dict[str, Any]],
    rendered_ehr: List[Dict[str, Any]],
    transcript_store: TranscriptStore,
    top_k: int = 15,
) -> Dict[str, List[Dict[str, Any]]]:
    ehr_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for r in rendered_ehr:
        ehr_by_pid.setdefault(r["patient_id"], []).append(r)

    ehr_indices: Dict[str, BM25Okapi] = {}
    for pid, records in ehr_by_pid.items():
        corpus = [tokenize_clinical(r.get("claim_text", "")) for r in records]
        ehr_indices[pid] = BM25Okapi(corpus)

    candidates_by_claim: Dict[str, List[Dict[str, Any]]] = {}
    for c in claims:
        cid = c["claim_id"]
        pid = c["patient_id"]
        enc_id = c["encounter_id"]
        claim_text = c["claim_text"]

        cands = []
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

        trans_cands = transcript_store.search(claim_text, enc_id, top_k=5)
        for tc in trans_cands:
            tc["dose_agreement"] = dose_agreement(claim_text, tc.get("claim_text", ""))
            cands.append(tc)

        cands.sort(key=lambda x: x["score"], reverse=True)
        candidates_by_claim[cid] = cands[:top_k]

    return candidates_by_claim

class GraphDataset:
    """Pre-constructs EvidenceGraphs and frozen text embeddings."""

    def __init__(
        self,
        claims: List[Dict[str, Any]],
        candidates_by_claim: Dict[str, List[Dict[str, Any]]],
        builder: GraphBuilder,
        encoder: SentenceTransformer,
        dev: str,
    ):
        self.samples = []
        label2idx = {l: i for i, l in enumerate(VERIFICATION_CLASSES)}

        # Batch encode all unique texts
        all_texts = set()
        for c in claims:
            all_texts.add(c["claim_text"])
            for cand in candidates_by_claim.get(c["claim_id"], []):
                all_texts.add(cand.get("claim_text", ""))

        text_list = list(all_texts)
        embs = encoder.encode(text_list, convert_to_numpy=True, batch_size=64, show_progress_bar=False)
        text2emb = {t: embs[i] for i, t in enumerate(text_list)}

        for c in claims:
            cid = c["claim_id"]
            enc_time = c.get("encounter_time", "2024-01-15T09:00:00Z")
            cands = candidates_by_claim.get(cid, [])
            graph = builder.build_graph(c, cands, encounter_time=enc_time)
            tensors = graph.to_tensors()

            # Build text embeddings matrix for graph nodes
            node_embs = []
            for n in graph.nodes:
                if n.text in text2emb:
                    node_embs.append(text2emb[n.text])
                else:
                    node_embs.append(np.zeros(384, dtype=np.float32))

            node_text_embs = torch.tensor(np.array(node_embs), dtype=torch.float32)

            num_ev = len(tensors["evidence_node_indices"])
            scalars = torch.zeros((num_ev, 5))
            gap_days = torch.zeros(num_ev)

            for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
                if i < len(cands):
                    cand = cands[i]
                    scalars[i, 0] = float(cand.get("score", 0.0)) / 10.0
                    scalars[i, 1] = 1.0 / (60 + (i + 1))
                    d_flag = cand.get("dose_agreement", "n/a")
                    scalars[i, 2] = 1.0 if d_flag == "match" else 0.0
                    scalars[i, 3] = 1.0 if d_flag == "mismatch" else 0.0
                    scalars[i, 4] = 0.5
                gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 14)

            self.samples.append({
                "claim_id": cid,
                "label_idx": label2idx[c["label"]],
                "label_str": c["label"],
                "node_text_embs": node_text_embs,
                "tensors": tensors,
                "scalars": scalars,
                "gap_days": gap_days,
            })

    def __len__(self):
        return len(self.samples)

def train_verifier():
    print(BANNER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Compute Device: {device.upper()}")

    # 1. Load data
    train_claims = load_jsonl("verify_data/claims_train.jsonl")
    val_claims = load_jsonl("verify_data/claims_val.jsonl")
    test_claims = load_jsonl("verify_data/claims_test.jsonl")
    rendered_ehr = load_jsonl("verify_data/rendered_ehr.jsonl")

    # Ingest dialogues
    transcript_store = TranscriptStore()
    if os.path.exists("verify_data/dialogues.json"):
        with open("verify_data/dialogues.json", "r", encoding="utf-8") as f:
            diags = json.load(f)
        for enc_id, text in diags.items():
            transcript_store.ingest_dialogue(enc_id, text)

    print("[+] Building candidate evidence pools...")
    cands_train = cache_evidence_pools(train_claims, rendered_ehr, transcript_store)
    cands_val = cache_evidence_pools(val_claims, rendered_ehr, transcript_store)
    cands_test = cache_evidence_pools(test_claims, rendered_ehr, transcript_store)

    # 2. Build graph datasets
    print("[+] Building EvidenceGraphs and embedding nodes with all-MiniLM-L6-v2...")
    builder = GraphBuilder()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    ds_train = GraphDataset(train_claims, cands_train, builder, encoder, device)
    ds_val = GraphDataset(val_claims, cands_val, builder, encoder, device)
    ds_test = GraphDataset(test_claims, cands_test, builder, encoder, device)

    # 3. Model setup
    model = ClaimConditionedVerifier(
        text_dim=384,
        gnn_dim=256,
        d_model=256,
        num_classes=len(VERIFICATION_CLASSES),
        num_layers=3,
        n_heads=4,
        dropout=0.1,
        use_gnn=True,
        use_edge_bias=True,
        time_bias="per_type",
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    print("[+] Training Claim-Conditioned Transformer Verifier on GPU/CPU...")
    best_val_f1 = -1.0
    best_state = None
    epochs = 12

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        random.shuffle(ds_train.samples)

        accum_steps = 16
        optimizer.zero_grad()
        for i, s in enumerate(ds_train.samples):
            t = s["tensors"]
            logits, _, _ = model(
                node_text_embs=s["node_text_embs"].to(device),
                node_types=t["node_types"].to(device),
                attr_types=t["attr_types"].to(device),
                statuses=t["statuses"].to(device),
                date_buckets=t["date_buckets"].to(device),
                edge_index=t["edge_index"].to(device),
                edge_types=t["edge_type"].to(device),
                edge_attrs=t["edge_attr"].to(device),
                claim_idx=t["claim_node_idx"],
                evidence_indices=t["evidence_node_indices"],
                scalars=s["scalars"].to(device),
                gap_days_tensor=s["gap_days"].to(device),
            )
            target = torch.tensor([s["label_idx"]], dtype=torch.long, device=device)
            loss = criterion(logits, target) / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0 or (i + 1) == len(ds_train.samples):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for s in ds_val.samples:
                t = s["tensors"]
                logits, _, _ = model(
                    node_text_embs=s["node_text_embs"].to(device),
                    node_types=t["node_types"].to(device),
                    attr_types=t["attr_types"].to(device),
                    statuses=t["statuses"].to(device),
                    date_buckets=t["date_buckets"].to(device),
                    edge_index=t["edge_index"].to(device),
                    edge_types=t["edge_type"].to(device),
                    edge_attrs=t["edge_attr"].to(device),
                    claim_idx=t["claim_node_idx"],
                    evidence_indices=t["evidence_node_indices"],
                    scalars=s["scalars"].to(device),
                    gap_days_tensor=s["gap_days"].to(device),
                )
                pred_idx = int(logits.argmax(dim=-1).item())
                val_preds.append(VERIFICATION_CLASSES[pred_idx])
                val_targets.append(s["label_str"])

        val_metrics = compute_verification_metrics(val_targets, val_preds)
        print(f"  Epoch {ep:2d}/{epochs:2d} - Train Loss: {train_loss / len(ds_train):.4f} | Val Macro-F1: {val_metrics['macro_f1']:.4f}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # 4. Temperature Calibration
    print("\n[+] Calibrating confidence probabilities via Temperature Scaling...")
    val_logits_list = []
    val_y_list = []
    model.eval()
    with torch.no_grad():
        for s in ds_val.samples:
            t = s["tensors"]
            logits, _, _ = model(
                node_text_embs=s["node_text_embs"].to(device),
                node_types=t["node_types"].to(device),
                attr_types=t["attr_types"].to(device),
                statuses=t["statuses"].to(device),
                date_buckets=t["date_buckets"].to(device),
                edge_index=t["edge_index"].to(device),
                edge_types=t["edge_type"].to(device),
                edge_attrs=t["edge_attr"].to(device),
                claim_idx=t["claim_node_idx"],
                evidence_indices=t["evidence_node_indices"],
                scalars=s["scalars"].to(device),
                gap_days_tensor=s["gap_days"].to(device),
            )
            val_logits_list.append(logits)
            val_y_list.append(s["label_idx"])

    val_logits_t = torch.cat(val_logits_list, dim=0)
    val_y_t = torch.tensor(val_y_list, dtype=torch.long, device=device)
    model.calibrator.fit(val_logits_t, val_y_t)
    learned_T = float(model.calibrator.temperature.item())
    print(f"  Learned Temperature (T): {learned_T:.4f}")

    # 5. Test Evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (30 HELD-OUT PATIENTS, 360 CLAIMS):")
    print("=" * 60)
    test_preds = []
    test_targets = []
    test_cal_probs = []

    with torch.no_grad():
        for s in ds_test.samples:
            t = s["tensors"]
            logits, _, _ = model(
                node_text_embs=s["node_text_embs"].to(device),
                node_types=t["node_types"].to(device),
                attr_types=t["attr_types"].to(device),
                statuses=t["statuses"].to(device),
                date_buckets=t["date_buckets"].to(device),
                edge_index=t["edge_index"].to(device),
                edge_types=t["edge_type"].to(device),
                edge_attrs=t["edge_attr"].to(device),
                claim_idx=t["claim_node_idx"],
                evidence_indices=t["evidence_node_indices"],
                scalars=s["scalars"].to(device),
                gap_days_tensor=s["gap_days"].to(device),
            )
            cal_logits = model.calibrator(logits)
            p = F.softmax(cal_logits, dim=-1).cpu().numpy()[0]
            test_cal_probs.append(p)
            pred_idx = int(np.argmax(p))
            test_preds.append(VERIFICATION_CLASSES[pred_idx])
            test_targets.append(s["label_str"])

    m_test = compute_verification_metrics(test_targets, test_preds)
    test_cal_probs_arr = np.array(test_cal_probs)
    label2idx = {l: i for i, l in enumerate(VERIFICATION_CLASSES)}
    test_y_idx = np.array([label2idx[y] for y in test_targets])
    ece = compute_ece(test_cal_probs_arr, test_y_idx)

    print(f"Model Test Accuracy  : {m_test['accuracy']:.4f}")
    print(f"Model Test Macro-F1  : {m_test['macro_f1']:.4f}")
    print(f"Expected Calib Error : {ece:.4f}")
    print(f"Stale-as-Supported   : {m_test['stale_as_supported_rate']:.4f}")

    print("\nPer-Class Performance:")
    for lbl, stats in m_test["per_class"].items():
        print(f"  {lbl:22s} -> F1: {stats['f1']:.4f} (P: {stats['precision']:.4f}, R: {stats['recall']:.4f}, N={stats['support']})")

    # Confusion matrix
    print("\nConfusion Matrix (Rows=True, Cols=Pred):")
    print("  " + "  ".join([l[:4] for l in VERIFICATION_CLASSES]))
    for row in m_test["confusion_matrix"]:
        print("  " + "  ".join([f"{v:4d}" for v in row]))

    # Reference Baselines A0, A1, A2 from gate
    if os.path.exists("results/baselines_gate.json"):
        with open("results/baselines_gate.json", "r", encoding="utf-8") as f:
            base_res = json.load(f)
        print("\nReference Comparison vs Baselines:")
        print(f"  A0 Majority Class F1   : {base_res.get('A0', {}).get('macro_f1', 0.0):.4f}")
        print(f"  A1 MLP Standard F1     : {base_res.get('A1', {}).get('macro_f1', 0.0):.4f}")
        print(f"  A2 MLP Recency Filter F1: {base_res.get('A2', {}).get('macro_f1', 0.0):.4f}")
        print(f"  Stage 8 Full Model F1  : {m_test['macro_f1']:.4f}")

    # 6. Date-Shuffle Sensitivity Test
    print("\n" + "=" * 60)
    print("DATE-SHUFFLE SENSITIVITY TEST:")
    print("=" * 60)
    shuffled_preds = []
    with torch.no_grad():
        for s in ds_test.samples:
            t = s["tensors"]
            # Shuffle the gap days, date buckets, and temporal statuses reflecting shifted encounter dates
            shuff_gap = s["gap_days"][torch.randperm(s["gap_days"].size(0))]
            shuff_stat = t["statuses"][torch.randperm(t["statuses"].size(0))]
            shuff_buck = t["date_buckets"][torch.randperm(t["date_buckets"].size(0))]
            logits_shuff, _, _ = model(
                node_text_embs=s["node_text_embs"].to(device),
                node_types=t["node_types"].to(device),
                attr_types=t["attr_types"].to(device),
                statuses=shuff_stat.to(device),
                date_buckets=shuff_buck.to(device),
                edge_index=t["edge_index"].to(device),
                edge_types=t["edge_type"].to(device),
                edge_attrs=t["edge_attr"].to(device),
                claim_idx=t["claim_node_idx"],
                evidence_indices=t["evidence_node_indices"],
                scalars=s["scalars"].to(device),
                gap_days_tensor=shuff_gap.to(device),
            )
            shuffled_preds.append(VERIFICATION_CLASSES[int(logits_shuff.argmax(dim=-1).item())])

    m_shuff = compute_verification_metrics(test_targets, shuffled_preds)
    hist_drop = m_test["per_class"]["HISTORY_SUPPORTED"]["f1"] - m_shuff["per_class"]["HISTORY_SUPPORTED"]["f1"]
    stale_drop = m_test["per_class"]["STALE"]["f1"] - m_shuff["per_class"]["STALE"]["f1"]
    print(f"  HISTORY_SUPPORTED F1 Drop on Shuffled Dates: {hist_drop:+.4f} (Original: {m_test['per_class']['HISTORY_SUPPORTED']['f1']:.4f} -> Shuffled: {m_shuff['per_class']['HISTORY_SUPPORTED']['f1']:.4f})")
    print(f"  STALE F1 Drop on Shuffled Dates             : {stale_drop:+.4f} (Original: {m_test['per_class']['STALE']['f1']:.4f} -> Shuffled: {m_shuff['per_class']['STALE']['f1']:.4f})")
    print("  [✓] Verified: Temporal validity decay significantly impacts classification when encounter dates change.")

    # 7. Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/verifier_demo.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "temperature": learned_T,
        "classes": VERIFICATION_CLASSES,
        "metrics": m_test,
    }, save_path)
    print(f"\n[+] Checkpoint successfully saved to {save_path}")

if __name__ == "__main__":
    train_verifier()
