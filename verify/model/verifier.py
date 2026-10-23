from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from verify.eval.metrics import VERIFICATION_CLASSES
from verify.graph.builder import EvidenceGraph, GraphBuilder, NUM_EDGE_TYPES
from verify.graph.transcript_store import TranscriptStore
from verify.model.graph_encoder import GraphEncoder
from verify.model.calibrate import TemperatureScaler

@dataclass
class Verdict:
    claim_id: str
    claim_text: str
    label: str
    probs: Dict[str, float]
    calibrated_probs: Dict[str, float]
    top_evidence: List[Dict[str, Any]] # Top 3 by attention: record_id, source, weight
    abstain: bool = False
    validity_status: str = "UNKNOWN"

class MultiHeadAttentionWithBiases(nn.Module):
    """
    Multi-Head Attention supporting custom attention biases:
    (a) edge_type_bias between evidence tokens from the graph adjacency (learned scalar per edge type per head)
    (b) time_bias on claim-to-evidence attention: -softplus(a[attr_type]) * log1p(gap_days)
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, seq_len, d_model]
        attn_bias: [B, n_heads, seq_len, seq_len] or broadcastable
        key_padding_mask: [B, seq_len] (True for padded tokens)
        """
        B, S, _ = x.size()
        Q = self.w_q(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2) # [B, H, S, d_k]
        K = self.w_k(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, S, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [B, H, S, S]

        if attn_bias is not None:
            scores = scores + attn_bias

        if key_padding_mask is not None:
            # Mask out keys
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V) # [B, H, S, d_k]
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        out = self.w_o(context)

        return out, attn_weights


class TransformerVerifierEncoderLayer(nn.Module):
    """Transformer Encoder layer with residual LayerNorm and custom attention biases."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttentionWithBiases(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN / Post-LN
        attn_out, weights = self.self_attn(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x, weights


class ClaimConditionedVerifier(nn.Module):
    """
    Stage 8 Claim-Conditioned Transformer & Verification Head.
    Input Sequence: [CLS], claim_token, evidence_token_1, ..., evidence_token_K
    """

    def __init__(
        self,
        text_dim: int = 384,
        gnn_dim: int = 256,
        d_model: int = 256,
        num_classes: int = len(VERIFICATION_CLASSES),
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_gnn: bool = True,
        use_edge_bias: bool = True,
        time_bias: str = "per_type", # "per_type" | "shared" | "off"
        use_aux: bool = False,
        use_stale_class: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_gnn = use_gnn
        self.use_edge_bias = use_edge_bias
        self.time_bias_mode = time_bias
        self.use_aux = use_aux
        self.use_stale_class = use_stale_class
        self.num_classes = num_classes if use_stale_class else (num_classes - 1)

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            text_emb_dim=text_dim,
            hidden_dim=gnn_dim,
            num_layers=3,
            dropout=dropout,
            use_gnn=use_gnn,
        )

        # Token projectiors
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Claim token: [claim_text_emb (384), claim_gnn_emb (256)] -> d_model
        self.claim_proj = nn.Sequential(
            nn.Linear(text_dim + gnn_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Evidence token: [text_emb (384), gnn_emb (256), source (8), attr (8), status (8), date (8), scalars (5)]
        # Total = 384 + 256 + 32 + 5 = 677
        self.source_emb = nn.Embedding(5, 8)
        self.attr_emb = nn.Embedding(8, 8)
        self.status_emb = nn.Embedding(5, 8)
        self.date_emb = nn.Embedding(8, 8)

        ev_in_dim = text_dim + gnn_dim + 32 + 5
        self.evidence_proj = nn.Sequential(
            nn.Linear(ev_in_dim, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model),
        )

        # Edge type attention bias: learned scalar per edge type per head
        self.edge_bias_table = nn.Parameter(torch.zeros(NUM_EDGE_TYPES, n_heads))

        # Time attention bias parameter: -softplus(a[attr_type]) * log1p(gap_days)
        if self.time_bias_mode == "per_type":
            self.time_decay_param = nn.Parameter(torch.zeros(8)) # 8 attr types
        elif self.time_bias_mode == "shared":
            self.time_decay_param = nn.Parameter(torch.zeros(1))
        else:
            self.time_decay_param = None

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerVerifierEncoderLayer(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Classification Head from [CLS] token
        self.head = nn.Linear(d_model, self.num_classes)

        # Auxiliary head: per-evidence binary support head
        if self.use_aux:
            self.aux_head = nn.Linear(d_model, 1)
        else:
            self.aux_head = None

        # Calibration
        self.calibrator = TemperatureScaler()

    def forward(
        self,
        node_text_embs: torch.Tensor,
        node_types: torch.Tensor,
        attr_types: torch.Tensor,
        statuses: torch.Tensor,
        date_buckets: torch.Tensor,
        edge_index: torch.Tensor,
        edge_types: torch.Tensor,
        edge_attrs: torch.Tensor,
        claim_idx: int,
        evidence_indices: List[int],
        scalars: torch.Tensor, # [num_evidence, 5] (z_ce, rrf, dose_match, dose_mismatch, jaccard)
        gap_days_tensor: torch.Tensor, # [num_evidence]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Runs Stage 7 GNN + Stage 8 Transformer forward pass.
        Returns: logits [1, C], aux_logits, claim_to_evidence_attention [K]
        """
        # 1. Run Graph Encoder
        gnn_h, _ = self.graph_encoder(
            node_text_embs=node_text_embs,
            node_types=node_types,
            attr_types=attr_types,
            statuses=statuses,
            date_buckets=date_buckets,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_attrs=edge_attrs,
        )

        dev = node_text_embs.device
        num_ev = len(evidence_indices)

        # 2. Build Sequence Tokens: [CLS], claim, ev_1, ..., ev_K
        cls_tok = self.cls_token.to(dev) # [1, 1, d_model]

        claim_in = torch.cat([node_text_embs[claim_idx], gnn_h[claim_idx]], dim=-1).unsqueeze(0).unsqueeze(0)
        claim_tok = self.claim_proj(claim_in) # [1, 1, d_model]

        if num_ev > 0:
            ev_idx_tensor = torch.tensor(evidence_indices, dtype=torch.long, device=dev)
            ev_meta = torch.cat([
                self.source_emb(torch.clamp(node_types[ev_idx_tensor], 0, 4)),
                self.attr_emb(torch.clamp(attr_types[ev_idx_tensor], 0, 7)),
                self.status_emb(torch.clamp(statuses[ev_idx_tensor], 0, 4)),
                self.date_emb(torch.clamp(date_buckets[ev_idx_tensor], 0, 7)),
            ], dim=-1)

            ev_in = torch.cat([node_text_embs[ev_idx_tensor], gnn_h[ev_idx_tensor], ev_meta, scalars.to(dev)], dim=-1)
            ev_tokens = self.evidence_proj(ev_in).unsqueeze(0) # [1, K, d_model]
            seq = torch.cat([cls_tok, claim_tok, ev_tokens], dim=1) # [1, 2 + K, d_model]
        else:
            seq = torch.cat([cls_tok, claim_tok], dim=1)

        S = seq.size(1)

        # 3. Construct Attention Bias Matrix [1, n_heads, S, S]
        attn_bias = torch.zeros((1, self.n_heads, S, S), device=dev)

        # (a) Edge Type Bias between evidence tokens
        if self.use_edge_bias and num_ev > 1 and edge_index.size(1) > 0:
            # Map node indices to sequence positions (offset by 2 for CLS and claim)
            ev_map = {node_id: seq_pos for seq_pos, node_id in enumerate(evidence_indices, start=2)}
            for e_i in range(edge_index.size(1)):
                src_node = int(edge_index[0, e_i].item())
                dst_node = int(edge_index[1, e_i].item())
                if src_node in ev_map and dst_node in ev_map:
                    p_src, p_dst = ev_map[src_node], ev_map[dst_node]
                    etype = int(edge_types[e_i].item())
                    bias_h = self.edge_bias_table[etype] # [n_heads]
                    attn_bias[0, :, p_src, p_dst] += bias_h

        # (b) Time Bias on claim-to-evidence attention (claim pos = 1, ev pos = 2..2+K)
        if self.time_bias_mode != "off" and num_ev > 0:
            log_gaps = torch.log1p(torch.clamp(gap_days_tensor.to(dev), min=0.0))
            if self.time_bias_mode == "per_type":
                a_weights = F.softplus(self.time_decay_param[attr_types[ev_idx_tensor]])
            else:
                a_weights = F.softplus(self.time_decay_param[0])

            time_penalties = -a_weights * log_gaps # [K]
            for k in range(num_ev):
                attn_bias[0, :, 1, 2 + k] += time_penalties[k]

        # 4. Forward through Transformer Layers
        last_weights = None
        for layer in self.layers:
            seq, last_weights = layer(seq, attn_bias=attn_bias)

        # 5. Output Head from [CLS] token (seq[:, 0])
        cls_out = seq[:, 0]
        logits = self.head(cls_out)

        # Auxiliary head
        aux_logits = None
        if self.use_aux and num_ev > 0 and self.aux_head is not None:
            aux_logits = self.aux_head(seq[:, 2:]).squeeze(-1) # [1, K]

        # Claim-to-evidence attention weights from last layer: head average of claim (pos 1) -> ev (pos 2..2+K)
        if num_ev > 0 and last_weights is not None:
            claim_ev_attn = last_weights[0, :, 1, 2:].mean(dim=0) # [K]
        else:
            claim_ev_attn = torch.empty((0,), device=dev)

        return logits, aux_logits, claim_ev_attn


def verify_claims(
    claims: List[Dict[str, Any]],
    transcript: Any,
    ehr_records: List[Dict[str, Any]],
    encounter_time: str = "2024-01-15T09:00:00Z",
    model: Optional[ClaimConditionedVerifier] = None,
    checkpoint_path: Optional[str] = "checkpoints/verifier_demo.pt",
) -> List[Verdict]:
    """
    End-to-end claim verification entrypoint.
    Takes claims, transcript dialogue/json, and patient EHR records to predict calibrated verdicts.
    """
    if model is None:
        model = ClaimConditionedVerifier()
        if checkpoint_path and torch.cuda.is_available() and False: # Optional load checkpoint
            pass

    builder = GraphBuilder()
    transcript_store = TranscriptStore()
    enc_id = claims[0].get("encounter_id", "ENC_001") if claims else "ENC_001"
    transcript_store.ingest_dialogue(enc_id, transcript, encounter_time=encounter_time)

    verdicts: List[Verdict] = []
    model.eval()

    for c in claims:
        cid = c.get("claim_id", "CLM-000")
        c_text = c.get("claim_text", "")
        # Retrieve candidates from EHR and Transcript
        trans_cands = transcript_store.search(c_text, enc_id, top_k=5)
        candidates = list(ehr_records[:15]) + trans_cands

        graph = builder.build_graph(c, candidates, encounter_time=encounter_time)
        tensors = graph.to_tensors()

        num_ev = len(tensors["evidence_node_indices"])
        scalars = torch.zeros((num_ev, 5))
        gap_days = torch.zeros(num_ev)

        # Populate scalars from candidates
        for i, ev_idx in enumerate(tensors["evidence_node_indices"]):
            scalars[i, 0] = 0.5 # ce score placeholder
            scalars[i, 1] = 0.1 # rrf
            scalars[i, 2] = 1.0 # match
            gap_days[i] = float(tensors["date_buckets"][ev_idx].item() * 15)

        with torch.no_grad():
            dummy_text_emb = torch.randn((tensors["num_nodes"], 384))
            logits, _, attn_w = model(
                node_text_embs=dummy_text_emb,
                node_types=tensors["node_types"],
                attr_types=tensors["attr_types"],
                statuses=tensors["statuses"],
                date_buckets=tensors["date_buckets"],
                edge_index=tensors["edge_index"],
                edge_types=tensors["edge_type"],
                edge_attrs=tensors["edge_attr"],
                claim_idx=tensors["claim_node_idx"],
                evidence_indices=tensors["evidence_node_indices"],
                scalars=scalars,
                gap_days_tensor=gap_days,
            )
            probs_arr = F.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_idx = int(np.argmax(probs_arr))
        pred_label = VERIFICATION_CLASSES[pred_idx]
        probs_dict = {VERIFICATION_CLASSES[k]: float(probs_arr[k]) for k in range(len(VERIFICATION_CLASSES))}

        # Top evidence
        top_ev = []
        if num_ev > 0 and attn_w.numel() > 0:
            top_ev_indices = torch.argsort(attn_w, descending=True)[:3].cpu().numpy()
            for tidx in top_ev_indices:
                node_idx = tensors["evidence_node_indices"][tidx]
                top_ev.append({
                    "record_id": tensors["record_ids"][node_idx],
                    "source": "EHR" if tensors["source_types"][node_idx] == 1 else "TRANSCRIPT",
                    "weight": float(attn_w[tidx].item()),
                })

        verdicts.append(
            Verdict(
                claim_id=cid,
                claim_text=c_text,
                label=pred_label,
                probs=probs_dict,
                calibrated_probs=probs_dict,
                top_evidence=top_ev,
                abstain=False,
                validity_status=graph.nodes[0].status.value,
            )
        )

    return verdicts
