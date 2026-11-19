"""
V2 Ablation, Calibration, and Candidate-Pool Evaluation Runner (Synthea + Gemini Longitudinal Dataset).

Leaves all v1 scripts/results untouched.
Writes:
  - results/ablations_v2.json
  - results/ablations_v2.md
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

root_dir = str(Path(__file__).parent.parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from scripts.train_verifier import GraphDataset, cache_evidence_pools, load_jsonl
from verify.eval.metrics import VERIFICATION_CLASSES, compute_verification_metrics
from verify.graph.builder import GraphBuilder
from verify.graph.transcript_store import TranscriptStore
from verify.model.calibrate import compute_ece
from verify.model.verifier import ClaimConditionedVerifier


def _eval_model_on_dataset(
    model: ClaimConditionedVerifier,
    ds: GraphDataset,
    device: str,
    temperature: float = 1.0,
    zero_time: bool = False,
    shuffle_time: bool = False,
) -> Dict[str, Any]:
    model.eval()
    preds, targets = [], []
    uncal_probs, cal_probs = [], []

    orig_time_mode = model.time_bias_mode
    if zero_time:
        model.time_bias_mode = "off"

    with torch.no_grad():
        for s in ds.samples:
            t = s["tensors"]
            gap = s["gap_days"]
            stat = t["statuses"]
            buck = t["date_buckets"]
            if zero_time:
                gap = torch.zeros_like(gap)
                buck = torch.zeros_like(buck)
            elif shuffle_time:
                perm = torch.randperm(gap.size(0))
                gap = gap[perm]
                stat = t["statuses"][torch.randperm(stat.size(0))]
                buck = t["date_buckets"][torch.randperm(buck.size(0))]

            logits, _, _ = model(
                node_text_embs=s["node_text_embs"].to(device),
                node_types=t["node_types"].to(device),
                attr_types=t["attr_types"].to(device),
                statuses=stat.to(device),
                date_buckets=buck.to(device),
                edge_index=t["edge_index"].to(device),
                edge_types=t["edge_type"].to(device),
                edge_attrs=t["edge_attr"].to(device),
                claim_idx=t["claim_node_idx"],
                evidence_indices=t["evidence_node_indices"],
                scalars=s["scalars"].to(device),
                gap_days_tensor=gap.to(device),
            )
            p_uncal = F.softmax(logits, dim=-1).cpu().numpy()[0]
            p_cal = F.softmax(logits / temperature, dim=-1).cpu().numpy()[0]
            uncal_probs.append(p_uncal)
            cal_probs.append(p_cal)
            preds.append(VERIFICATION_CLASSES[int(np.argmax(p_cal))])
            targets.append(s["label_str"])

    model.time_bias_mode = orig_time_mode
    metrics = compute_verification_metrics(targets, preds)

    label2idx = {l: i for i, l in enumerate(VERIFICATION_CLASSES)}
    y_idx = np.array([label2idx[y] for y in targets])
    uncal_arr = np.array(uncal_probs)
    cal_arr = np.array(cal_probs)

    return {
        "metrics": metrics,
        "ece_uncalibrated": float(compute_ece(uncal_arr, y_idx)),
        "ece_calibrated": float(compute_ece(cal_arr, y_idx)),
        "uncal_probs": uncal_arr,
        "cal_probs": cal_arr,
        "y_idx": y_idx,
    }


def _reliability_bins(probs: np.ndarray, y_idx: np.ndarray, num_bins: int = 5) -> Dict[str, List[float]]:
    """Uses quantile-based binning so every bin has ~N/num_bins samples (no 1-sample spikes)."""
    confidences = np.max(probs, axis=1)
    accuracies = (np.argmax(probs, axis=1) == y_idx).astype(float)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(confidences, quantiles)
    # Fallback to uniform edges if quantiles collapse near 1.0
    if len(np.unique(np.round(edges, 4))) < 3:
        edges = np.array([0.0, 0.75, 0.88, 0.94, 0.98, 1.01])

    bin_centers, bin_accs, bin_counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == len(edges) - 2 else (confidences >= lo) & (confidences < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            bin_counts.append(cnt)
            bin_centers.append(float(np.mean(confidences[mask])))
            bin_accs.append(float(np.mean(accuracies[mask])))
    return {"conf_centers": bin_centers, "empirical_acc": bin_accs, "counts": bin_counts}


def _load_verifier(ckpt_path: str, use_gnn: bool, use_edge_bias: bool, time_bias: str, device: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ClaimConditionedVerifier(
        text_dim=384,
        gnn_dim=256,
        d_model=256,
        num_classes=len(VERIFICATION_CLASSES),
        num_layers=3,
        n_heads=4,
        dropout=0.1,
        use_gnn=use_gnn,
        use_edge_bias=use_edge_bias,
        time_bias=time_bias,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    temp = float(ckpt.get("temperature", 1.0))
    return model, temp


def main():
    os.makedirs("results", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    test_claims = load_jsonl("verify_data/claims_test.jsonl")
    rendered_ehr = load_jsonl("verify_data/rendered_ehr.jsonl")

    enc_time_map = {
        c["encounter_id"]: c["encounter_time"]
        for c in test_claims
        if c.get("encounter_id") and c.get("encounter_time")
    }
    transcript_store = TranscriptStore()
    with open("verify_data/dialogues.json", "r", encoding="utf-8") as f:
        diags = json.load(f)
    for enc_id, text in diags.items():
        transcript_store.ingest_dialogue(
            enc_id, text, encounter_time=enc_time_map.get(enc_id, "2024-01-15T09:00:00Z")
        )

    builder = GraphBuilder()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    # Load baselines A0, A1, A2
    with open("results/baselines_gate.json", "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Load A3 Transformer-only and A6 Full GNN + Temporal Transformer
    model_a3, temp_a3 = _load_verifier(
        "checkpoints/transformer_only_synthea.pt",
        use_gnn=False, use_edge_bias=False, time_bias="off", device=device,
    )
    model_a6, temp_a6 = _load_verifier(
        "checkpoints/verifier_demo.pt",
        use_gnn=True, use_edge_bias=True, time_bias="per_type", device=device,
    )

    # Full test pool (K=15 default)
    cands_k15 = cache_evidence_pools(test_claims, rendered_ehr, transcript_store, top_k=15)
    ds_k15 = GraphDataset(test_claims, cands_k15, builder, encoder, device)

    res_a3 = _eval_model_on_dataset(model_a3, ds_k15, device, temperature=temp_a3)
    res_a6 = _eval_model_on_dataset(model_a6, ds_k15, device, temperature=temp_a6)

    # A7: EHR-only (strip transcript candidates)
    cands_ehr_only = {
        cid: [c for c in cands if c.get("source") != "TRANSCRIPT"]
        for cid, cands in cands_k15.items()
    }
    ds_ehr_only = GraphDataset(test_claims, cands_ehr_only, builder, encoder, device)
    res_a7 = _eval_model_on_dataset(model_a6, ds_ehr_only, device, temperature=temp_a6)

    # A8: No time features
    res_a8 = _eval_model_on_dataset(model_a6, ds_k15, device, temperature=temp_a6, zero_time=True)

    # Date-shuffle sensitivity
    res_shuff = _eval_model_on_dataset(model_a6, ds_k15, device, temperature=temp_a6, shuffle_time=True)

    # Real K-sweep on test set (K in {3, 5, 10, 15, 20})
    k_sweep = []
    for k in (3, 5, 10, 15, 20):
        cands_k = cache_evidence_pools(test_claims, rendered_ehr, transcript_store, top_k=k)
        ds_k = GraphDataset(test_claims, cands_k, builder, encoder, device)
        rk = _eval_model_on_dataset(model_a6, ds_k, device, temperature=temp_a6)
        k_sweep.append({
            "k": k,
            "accuracy": rk["metrics"]["accuracy"],
            "macro_f1": rk["metrics"]["macro_f1"],
            "stale_as_supported_rate": rk["metrics"]["stale_as_supported_rate"],
        })

    ablations = [
        {
            "config": "A0 Majority Class",
            "desc": "Always predict training majority class",
            "accuracy": base_data["A0"]["accuracy"],
            "macro_f1": base_data["A0"]["macro_f1"],
            "stale_as_supp": base_data["A0"].get("stale_as_supported_rate", 0.0),
        },
        {
            "config": "A1 MLP Base",
            "desc": "MLP on [claim emb, top-1 evidence emb, BM25 score, dose tag]",
            "accuracy": base_data["A1"]["accuracy"],
            "macro_f1": base_data["A1"]["macro_f1"],
            "stale_as_supp": base_data["A1"]["stale_as_supported_rate"],
        },
        {
            "config": "A2 MLP Recency",
            "desc": "A1 with recency_only evidence filter",
            "accuracy": base_data["A2"]["accuracy"],
            "macro_f1": base_data["A2"]["macro_f1"],
            "stale_as_supp": base_data["A2"]["stale_as_supported_rate"],
        },
        {
            "config": "A3 Transf (No GNN)",
            "desc": "Claim-Conditioned Transformer without DenseGIN, edge bias, or time bias",
            "accuracy": res_a3["metrics"]["accuracy"],
            "macro_f1": res_a3["metrics"]["macro_f1"],
            "stale_as_supp": res_a3["metrics"]["stale_as_supported_rate"],
        },
        {
            "config": "A6 Full (GNN+Time)",
            "desc": "3-layer DenseGIN + Transformer with edge bias & per-type continuous time decay",
            "accuracy": res_a6["metrics"]["accuracy"],
            "macro_f1": res_a6["metrics"]["macro_f1"],
            "stale_as_supp": res_a6["metrics"]["stale_as_supported_rate"],
        },
        {
            "config": "A7 EHR-Only",
            "desc": "A6 with transcript evidence excluded from candidate pool",
            "accuracy": res_a7["metrics"]["accuracy"],
            "macro_f1": res_a7["metrics"]["macro_f1"],
            "stale_as_supp": res_a7["metrics"]["stale_as_supported_rate"],
        },
        {
            "config": "A8 No Time Feat",
            "desc": "A6 with log gap days, date buckets, and time bias zeroed out",
            "accuracy": res_a8["metrics"]["accuracy"],
            "macro_f1": res_a8["metrics"]["macro_f1"],
            "stale_as_supp": res_a8["metrics"]["stale_as_supported_rate"],
        },
    ]

    output_json = {
        "dataset": "Synthea + Gemini Longitudinal EHR (v2)",
        "test_patients": len({c["patient_id"] for c in test_claims}),
        "test_claims": len(test_claims),
        "classes": VERIFICATION_CLASSES,
        "full_model_metrics": res_a6["metrics"],
        "transformer_only_metrics": res_a3["metrics"],
        "calibration": {
            "temperature": temp_a6,
            "ece_uncalibrated": res_a6["ece_uncalibrated"],
            "ece_calibrated": res_a6["ece_calibrated"],
            "uncalibrated_bins": _reliability_bins(res_a6["uncal_probs"], res_a6["y_idx"]),
            "calibrated_bins": _reliability_bins(res_a6["cal_probs"], res_a6["y_idx"]),
        },
        "date_shuffle_sensitivity": {
            "original_history_f1": res_a6["metrics"]["per_class"]["HISTORY_SUPPORTED"]["f1"],
            "shuffled_history_f1": res_shuff["metrics"]["per_class"]["HISTORY_SUPPORTED"]["f1"],
            "original_stale_f1": res_a6["metrics"]["per_class"]["STALE"]["f1"],
            "shuffled_stale_f1": res_shuff["metrics"]["per_class"]["STALE"]["f1"],
        },
        "ablations": ablations,
        "k_sweep": k_sweep,
    }

    with open("results/ablations_v2.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    md_lines = [
        "# ClinNoteGuard Verification Ablation & Evaluation Report (v2 — Synthea + Gemini)",
        "",
        f"**Dataset**: 170 Synthea+Gemini patients (110 train / 26 val / {output_json['test_patients']} held-out test, N={output_json['test_claims']} test claims)  ",
        "**Ontology**: Curated concept layer (`concept_hierarchy.json`)  ",
        "",
        "---",
        "",
        "## 1. System Architecture & Real Inference Ablations",
        "",
        "| Config | Description | Test Accuracy | Macro-F1 | Stale-as-Supported |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    for row in ablations:
        md_lines.append(
            f"| **{row['config']}** | {row['desc']} | {row['accuracy']:.4f} | **{row['macro_f1']:.4f}** | {row['stale_as_supp']:.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Candidate Pool Size Sweep (Top-K)",
        "",
        "| Top-K | Test Accuracy | Macro-F1 | Stale-as-Supported |",
        "| :---: | :---: | :---: | :---: |",
    ])
    for row in k_sweep:
        md_lines.append(
            f"| **K = {row['k']}** | {row['accuracy']:.4f} | **{row['macro_f1']:.4f}** | {row['stale_as_supported_rate']:.4f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 3. Calibration & Temporal Sensitivity",
        f"- **Temperature Scaling**: $T = {temp_a6:.4f}$ | Uncalibrated ECE = `{res_a6['ece_uncalibrated']:.4f}` -> Calibrated ECE = `{res_a6['ece_calibrated']:.4f}`",
        f"- **Date-Shuffle Sensitivity**: `HISTORY_SUPPORTED` F1 `{output_json['date_shuffle_sensitivity']['original_history_f1']:.4f}` -> `{output_json['date_shuffle_sensitivity']['shuffled_history_f1']:.4f}`, `STALE` F1 `{output_json['date_shuffle_sensitivity']['original_stale_f1']:.4f}` -> `{output_json['date_shuffle_sensitivity']['shuffled_stale_f1']:.4f}`",
    ])

    with open("results/ablations_v2.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("[+] Saved results/ablations_v2.json and results/ablations_v2.md")


if __name__ == "__main__":
    main()
