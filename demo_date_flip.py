"""
Milestone 5: Date-Flip Clinical Demo (demo_date_flip.py).

Demonstrates temporal sensitivity of the Stage 7-8 Verifier:
Evaluates identical clinical claims against identical patient EHR histories across two different
encounter dates straddling medication changes, discontinuations, and chronic persistence.

Every verdict is computed directly via the trained model's neural forward pass.
"""

import sys
import json
import torch
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from sentence_transformers import SentenceTransformer

# Path setup
root_dir = str(Path(__file__).parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from verify.eval.metrics import VERIFICATION_CLASSES
from verify.graph.builder import GraphBuilder
from verify.model.verifier import ClaimConditionedVerifier
from api.retrieval.text_utils import dose_agreement

BANNER = """
================================================================================
CLINNOTEGUARD DATE-FLIP DEMONSTRATION (MILESTONE 5)
NOTICE: synthetic-trained, unvalidated
ONTOLOGY: curated concept layer
CHECKPOINT: checkpoints/verifier_demo.pt
================================================================================
"""

EXAMPLES = [
    {
        "title": "Example 1: Medication Dose Titration (Atorvastatin 10mg -> 20mg)",
        "patient_id": "P_DEMO_CARDIAC_01",
        "claim": "Patient takes Atorvastatin 10 mg oral daily.",
        "ehr_timeline": [
            {
                "record_id": "EHR_REC_01",
                "claim_text": "Patient started on Atorvastatin 10 mg oral daily for hyperlipidemia.",
                "valid_from": "2023-01-15T09:00:00Z",
                "valid_to": "2023-06-15T09:00:00Z",
                "event_type": "START",
                "source": "EHR",
            },
            {
                "record_id": "EHR_REC_02",
                "claim_text": "Atorvastatin dose increased to 20 mg oral daily.",
                "valid_from": "2023-06-15T09:00:00Z",
                "valid_to": None,
                "event_type": "CHANGE",
                "source": "EHR",
            },
        ],
        "date_early": "2023-03-15", # Before titration
        "date_late": "2023-09-15",  # After titration
        "expected_early": "HISTORY_SUPPORTED (10 mg was current)",
        "expected_late": "STALE (superseded by 20 mg dose)",
    },
    {
        "title": "Example 2: Discontinued Medication Therapy (Furosemide Stopped)",
        "patient_id": "P_DEMO_CARDIAC_02",
        "claim": "Patient takes Furosemide 20 mg oral daily.",
        "ehr_timeline": [
            {
                "record_id": "EHR_REC_03",
                "claim_text": "Initiated Furosemide 20 mg once daily for mild congestive fluid retention.",
                "valid_from": "2023-01-10T09:00:00Z",
                "valid_to": "2023-05-10T09:00:00Z",
                "event_type": "START",
                "source": "EHR",
            },
            {
                "record_id": "EHR_REC_04",
                "claim_text": "Discontinued Furosemide therapy due to resolution of fluid retention.",
                "valid_from": "2023-05-10T09:00:00Z",
                "valid_to": None,
                "event_type": "STOP",
                "source": "EHR",
            },
        ],
        "date_early": "2023-03-01", # Before discontinuation
        "date_late": "2023-08-01",  # After discontinuation
        "expected_early": "HISTORY_SUPPORTED (active prescription)",
        "expected_late": "STALE (stopped / ended prior to encounter)",
    },
    {
        "title": "Example 3: Persistent Allergy (Penicillin Hypersensitivity Across 3 Years)",
        "patient_id": "P_DEMO_ALLERGY_01",
        "claim": "Patient has an active allergy to penicillin.",
        "ehr_timeline": [
            {
                "record_id": "EHR_REC_05",
                "claim_text": "Patient has an active allergy to penicillin with historical hives and facial swelling.",
                "valid_from": "2021-01-15T09:00:00Z",
                "valid_to": None,
                "event_type": "START",
                "section": "Allergy",
                "source": "EHR",
            },
        ],
        "date_early": "2021-06-15", # 5 months after recording
        "date_late": "2024-01-15",  # 3 years later
        "expected_early": "HISTORY_SUPPORTED (active allergy)",
        "expected_late": "HISTORY_SUPPORTED (persists indefinitely despite >1000 day gap)",
    },
]

def run_date_flip_demo():
    print(BANNER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path("checkpoints/verifier_demo.pt")

    model = ClaimConditionedVerifier().to(device)
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded model checkpoint from {ckpt_path} on {device.upper()}.\n")
    else:
        print("[-] Checkpoint not found. Running with initialized weights.\n")

    model.eval()
    builder = GraphBuilder()
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    for ex in EXAMPLES:
        print("=" * 80)
        print(ex["title"])
        print(f"Claim: \"{ex['claim']}\"")
        print("-" * 80)

        # Run for Early Date vs Late Date
        results = {}
        for d_key, date_val, exp in [("EARLY", ex["date_early"], ex["expected_early"]), ("LATE", ex["date_late"], ex["expected_late"])]:
            enc_time = f"{date_val}T09:00:00Z"
            claim_dict = {"claim_id": "CLM-DEMO", "claim_text": ex["claim"]}
            cands = ex["ehr_timeline"]

            graph = builder.build_graph(claim_dict, cands, encounter_time=enc_time)
            tensors = graph.to_tensors()

            # Encode node texts
            node_texts = [n.text or " " for n in graph.nodes]
            embs = encoder.encode(node_texts, convert_to_numpy=True, show_progress_bar=False)
            node_text_embs = torch.tensor(embs, dtype=torch.float32, device=device)

            num_ev = len(tensors["evidence_node_indices"])
            scalars = torch.zeros((num_ev, 5), device=device)
            gap_days = torch.zeros(num_ev, device=device)

            for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
                if i < len(cands):
                    cand = cands[i]
                    scalars[i, 0] = 0.8
                    scalars[i, 1] = 0.05
                    d_flag = dose_agreement(ex["claim"], cand["claim_text"])
                    scalars[i, 2] = 1.0 if d_flag == "match" else 0.0
                    scalars[i, 3] = 1.0 if d_flag == "mismatch" else 0.0
                    scalars[i, 4] = 0.5
                gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 14)

            with torch.no_grad():
                logits, _, attn_w = model(
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
                cal_logits = model.calibrator(logits)
                probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()[0]

            pred_idx = int(np.argmax(probs))
            pred_lbl = VERIFICATION_CLASSES[pred_idx]
            conf = float(probs[pred_idx])

            # Top evidence used
            top_rec = tensors["record_ids"][tensors["evidence_node_indices"][0]] if num_ev > 0 else "None"
            results[d_key] = {
                "date": date_val,
                "label": pred_lbl,
                "confidence": conf,
                "expected": exp,
                "top_evidence": top_rec,
                "validity": graph.nodes[1].status.value if len(graph.nodes) > 1 else "UNKNOWN",
            }

        # Print Side by Side Comparison
        r1, r2 = results["EARLY"], results["LATE"]
        print(f"{'PARAMETER':<20} | {'ENCOUNTER DATE 1 (BEFORE)':<26} | {'ENCOUNTER DATE 2 (AFTER)':<26}")
        print("-" * 80)
        print(f"{'Encounter Date':<20} | {r1['date']:<26} | {r2['date']:<26}")
        print(f"{'Predicted Label':<20} | {r1['label']:<26} | {r2['label']:<26}")
        print(f"{'Confidence':<20} | {r1['confidence']:.2%}{'':<20} | {r2['confidence']:.2%}")
        print(f"{'Temporal Validity':<20} | {r1['validity']:<26} | {r2['validity']:<26}")
        print(f"{'Top Evidence Used':<20} | {r1['top_evidence']:<26} | {r2['top_evidence']:<26}")
        print(f"{'Clinical Ground Truth':<20} | {r1['expected']:<26} | {r2['expected']:<26}")
        print("=" * 80 + "\n")

if __name__ == "__main__":
    run_date_flip_demo()
