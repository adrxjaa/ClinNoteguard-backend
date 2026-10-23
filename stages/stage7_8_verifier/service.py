"""
Stage 7 & Stage 8: Dense GIN & Claim-Conditioned Transformer Verifier.
Executes deep neural verification and temperature-scaled confidence calibration.
"""

from typing import List, Dict, Any, Optional
import os
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from verify.eval.metrics import VERIFICATION_CLASSES
from verify.model.verifier import ClaimConditionedVerifier, Verdict
from verify.graph.builder import EvidenceGraph

class Stage78VerifierService:
    """
    Stage 7 & 8 Service: Claim-Conditioned Transformer Verifier.
    Performs GIN message passing, transformer cross-attention with edge and time biases,
    and returns calibrated faithfulness verdicts.
    """

    def __init__(self, checkpoint_path: str = "checkpoints/verifier_demo.pt", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.model = ClaimConditionedVerifier().to(self.device)
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)
        self._load_checkpoint()

    def _load_checkpoint(self):
        ckpt_p = Path(self.checkpoint_path)
        if ckpt_p.exists():
            ckpt = torch.load(ckpt_p, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "temperature" in ckpt:
                self.model.calibrator.temperature.data.fill_(ckpt["temperature"])
            self.model.eval()

    def verify_graph(
        self,
        graph: EvidenceGraph,
        claim_text: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Runs neural verification forward pass over a built EvidenceGraph.
        """
        tensors = graph.to_tensors()
        dev = self.device

        # Encode node texts
        node_texts = [n.text or " " for n in graph.nodes]
        embs = self.encoder.encode(node_texts, convert_to_numpy=True, show_progress_bar=False)
        node_text_embs = torch.tensor(embs, dtype=torch.float32, device=dev)

        num_ev = len(tensors["evidence_node_indices"])
        scalars = torch.zeros((num_ev, 5), device=dev)
        gap_days = torch.zeros(num_ev, device=dev)

        for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
            if i < len(candidates):
                c = candidates[i]
                scalars[i, 0] = float(c.get("score", 0.0)) / 10.0
                scalars[i, 1] = 1.0 / (60 + (i + 1))
                d_flag = c.get("dose_agreement", "n/a")
                scalars[i, 2] = 1.0 if d_flag == "match" else 0.0
                scalars[i, 3] = 1.0 if d_flag == "mismatch" else 0.0
                scalars[i, 4] = 0.5
            gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 14)

        with torch.no_grad():
            logits, _, attn_w = self.model(
                node_text_embs=node_text_embs,
                node_types=tensors["node_types"].to(dev),
                attr_types=tensors["attr_types"].to(dev),
                statuses=tensors["statuses"].to(dev),
                date_buckets=tensors["date_buckets"].to(dev),
                edge_index=tensors["edge_index"].to(dev),
                edge_types=tensors["edge_type"].to(dev),
                edge_attrs=tensors["edge_attr"].to(dev),
                claim_idx=tensors["claim_node_idx"],
                evidence_indices=tensors["evidence_node_indices"],
                scalars=scalars,
                gap_days_tensor=gap_days,
            )
            cal_logits = self.model.calibrator(logits)
            probs = torch.softmax(cal_logits, dim=-1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        pred_label = VERIFICATION_CLASSES[pred_idx]
        conf = float(probs[pred_idx])

        # Top 3 attributed evidence
        top_ev = []
        if num_ev > 0 and attn_w.numel() > 0:
            top_ev_indices = torch.argsort(attn_w, descending=True)[:3].cpu().numpy()
            for tidx in top_ev_indices:
                node_idx = tensors["evidence_node_indices"][tidx]
                top_ev.append({
                    "record_id": tensors["record_ids"][node_idx],
                    "source": "EHR" if tensors["source_types"][node_idx] == 1 else "TRANSCRIPT",
                    "attention_weight": round(float(attn_w[tidx].item()), 4),
                })

        return {
            "label": pred_label,
            "confidence": conf,
            "probabilities": {VERIFICATION_CLASSES[k]: float(probs[k]) for k in range(len(VERIFICATION_CLASSES))},
            "top_evidence": top_ev,
            "checkpoint": self.checkpoint_path,
        }

def get_stage7_8_verifier() -> Stage78VerifierService:
    return Stage78VerifierService()
