"""
Date-Flip & Counterfactual Temporal Sensitivity Evaluation v2 (Synthea + Gemini Verifier).

Evaluates checkpoints/verifier_demo.pt on:
  1. Clinical date-flip scenarios (dose titration, discontinued med, persistent allergy, stale lab)
  2. Synthea counterfactual pairs in data/counterfactual_pairs_v2.jsonl
Writes:
  - results/date_flip_v2.json
"""

import json
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

from api.retrieval.text_utils import dose_agreement
from scripts.demo_date_flip import EXAMPLES
from verify.eval.counterfactual import compute_temporal_sensitivity_index
from verify.eval.metrics import VERIFICATION_CLASSES
from verify.graph.builder import GraphBuilder
from verify.model.verifier import ClaimConditionedVerifier


def _run_forward(
    model: ClaimConditionedVerifier,
    builder: GraphBuilder,
    encoder: SentenceTransformer,
    patient_id: str,
    claim_text: str,
    ehr_timeline: List[Dict[str, Any]],
    enc_date: str,
    device: str,
    temp: float,
) -> Dict[str, Any]:
    cands = []
    for r in ehr_timeline:
        rc = dict(r)
        rc["score"] = 1.0
        rc["dose_agreement"] = dose_agreement(claim_text, rc.get("claim_text", ""))
        cands.append(rc)

    claim_obj = {
        "claim_id": "DF_V2",
        "patient_id": patient_id,
        "encounter_id": "ENC_DF",
        "encounter_time": enc_date if "T" in enc_date else f"{enc_date}T09:00:00Z",
        "claim_text": claim_text,
    }
    graph = builder.build_graph(claim_obj, cands, encounter_time=claim_obj["encounter_time"])
    tensors = graph.to_tensors()

    node_texts = [n.text for n in graph.nodes]
    embs = encoder.encode(node_texts, convert_to_numpy=True, show_progress_bar=False)
    node_text_embs = torch.tensor(embs, dtype=torch.float32).to(device)

    num_ev = len(tensors["evidence_node_indices"])
    scalars = torch.zeros((num_ev, 5), device=device)
    gap_days = torch.zeros(num_ev, device=device)
    for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
        if i < len(cands):
            scalars[i, 0] = 0.8
            scalars[i, 1] = 1.0 / (60 + (i + 1))
            d_flag = cands[i].get("dose_agreement", "n/a")
            scalars[i, 2] = 1.0 if d_flag == "match" else 0.0
            scalars[i, 3] = 1.0 if d_flag == "mismatch" else 0.0
            scalars[i, 4] = 0.5
        gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 14)

    with torch.no_grad():
        logits, _, _ = model(
            node_text_embs=node_text_embs,
            node_types=tensors["node_types"].to(device),
            attr_types=tensors["attr_types"].to(device),
            statuses=tensors["statuses"].to(device),
            date_buckets=tensors["date_buckets"].to(device),
            edge_index=tensors["edge_index"].to(device),
            edge_types=tensors["edge_type"].to(device),
            edge_attrs=tensors["edge_attr"].to(device),
            claim_idx=tensors["claim_node_idx"],
            evidence_indices=tensors["evidence_node_indices"],
            scalars=scalars,
            gap_days_tensor=gap_days,
        )
        probs = F.softmax(logits / temp, dim=-1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    return {"label": VERIFICATION_CLASSES[pred_idx], "confidence": float(probs[pred_idx])}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load("checkpoints/verifier_demo.pt", map_location=device, weights_only=False)
    model = ClaimConditionedVerifier(
        text_dim=384, gnn_dim=256, d_model=256,
        num_classes=len(VERIFICATION_CLASSES),
        num_layers=3, n_heads=4, dropout=0.1,
        use_gnn=True, use_edge_bias=True, time_bias="per_type",
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    temp = float(ckpt.get("temperature", 1.0))

    builder = GraphBuilder()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    scenario_results = []
    for ex in EXAMPLES:
        r1 = _run_forward(model, builder, encoder, ex["patient_id"], ex["claim"], ex["ehr_timeline"], ex["date_early"], device, temp)
        r2 = _run_forward(model, builder, encoder, ex["patient_id"], ex["claim"], ex["ehr_timeline"], ex["date_late"], device, temp)
        scenario_results.append({
            "title": ex["title"],
            "early": r1,
            "late": r2,
            "expected_early": ex["expected_early"],
            "expected_late": ex["expected_late"],
        })
        print(f"Scenario: {ex['title']}\n  Early ({ex['date_early']}): {r1['label']} ({r1['confidence']:.3f})\n  Late  ({ex['date_late']}): {r2['label']} ({r2['confidence']:.3f})")

    # Evaluate Synthea counterfactual pairs if present
    cf_path = Path("data/counterfactual_pairs_v2.jsonl")
    ehr_path = Path("verify_data/rendered_ehr.jsonl")
    tsi_stats = {}
    if cf_path.exists() and ehr_path.exists():
        with open(ehr_path, "r", encoding="utf-8") as f:
            ehr_by_pid: Dict[str, List[Dict[str, Any]]] = {}
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    ehr_by_pid.setdefault(rec["patient_id"], []).append(rec)
        with open(cf_path, "r", encoding="utf-8") as f:
            pairs = [json.loads(line) for line in f if line.strip()][:50]

        evaluated_pairs = []
        for p in pairs:
            timeline = ehr_by_pid.get(p["patient_id"], [])[:15]
            p1 = _run_forward(model, builder, encoder, p["patient_id"], p["claim_text"], timeline, p["t1_encounter_time"], device, temp)
            p2 = _run_forward(model, builder, encoder, p["patient_id"], p["claim_text"], timeline, p["t2_encounter_time"], device, temp)
            p["pred_t1"] = p1["label"]
            p["pred_t2"] = p2["label"]
            evaluated_pairs.append(p)

        if evaluated_pairs:
            tsi_stats = compute_temporal_sensitivity_index(evaluated_pairs)
            print("\nSynthea Counterfactual Date-Flip TSI:", json.dumps(tsi_stats, indent=2))

    out = {"scenarios": scenario_results, "counterfactual_tsi": tsi_stats}
    with open("results/date_flip_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[+] Saved results/date_flip_v2.json")


if __name__ == "__main__":
    main()
