import logging
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from verify.graph.builder import NUM_EDGE_TYPES

logger = logging.getLogger("GraphEncoder")

class DenseGINELayer(nn.Module):
    """
    Dense PyTorch implementation of GINEConv (GIN with edge features) with learned edge gating.
    Scales each message by: g = sigmoid(MLP([edge_features, edge_type_embedding]))
    """

    def __init__(self, hidden_dim: int = 256, edge_dim: int = 18, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = nn.Parameter(torch.zeros(1))

        # Learned edge gate
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Edge feature projection
        self.edge_proj = nn.Linear(hidden_dim + edge_dim, hidden_dim)

        # GIN node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [num_nodes, hidden_dim]
        edge_index: [2, num_edges] (src, dst)
        edge_features: [num_edges, edge_dim]
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)

        if num_edges == 0:
            out = self.norm(h + self.dropout(self.node_mlp((1 + self.eps) * h)))
            return out, torch.empty((0,), device=h.device)

        src, dst = edge_index[0], edge_index[1]
        h_src = h[src] # [num_edges, hidden_dim]

        # Compute edge gate
        gates = self.edge_gate_mlp(edge_features) # [num_edges, 1]

        # Combine neighbor embedding with edge features
        msg_in = torch.cat([h_src, edge_features], dim=-1)
        msg = F.relu(self.edge_proj(msg_in)) # [num_edges, hidden_dim]
        gated_msg = gates * msg

        # Aggregate messages at destination nodes
        aggr = torch.zeros((num_nodes, self.hidden_dim), dtype=h.dtype, device=h.device)
        aggr.index_add_(0, dst, gated_msg)

        # GIN update
        out = (1 + self.eps) * h + aggr
        out = self.node_mlp(out)
        out = self.norm(h + self.dropout(out)) # Residual + LayerNorm

        return out, gates.squeeze(-1)


class GraphEncoder(nn.Module):
    """
    Stage 7 Graph Encoder: 3-layer GIN with edge features, residual LayerNorm, and learned edge gating.
    Computes node embeddings across the entire EvidenceGraph.
    """

    def __init__(
        self,
        text_emb_dim: int = 384,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_edge_types: int = NUM_EDGE_TYPES,
        dropout: float = 0.1,
        use_gnn: bool = True,
    ):
        super().__init__()
        self.use_gnn = use_gnn
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Learned 8-d categorical embeddings
        self.node_type_emb = nn.Embedding(5, 8)
        self.attr_type_emb = nn.Embedding(8, 8)
        self.status_emb = nn.Embedding(5, 8)
        self.date_bucket_emb = nn.Embedding(8, 8)
        self.edge_type_emb = nn.Embedding(num_edge_types, 8)

        # Initial node projection: 384 (frozen text) + 32 (metadata) -> hidden_dim
        total_in_dim = text_emb_dim + 32
        self.input_proj = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge feature dim: 10 continuous edge features + 8 learned edge_type embedding = 18
        self.edge_dim = 10 + 8

        self.layers = nn.ModuleList([
            DenseGINELayer(hidden_dim=hidden_dim, edge_dim=self.edge_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Track gate statistics per edge type for inspection
        self.register_buffer("mean_gates_per_type", torch.zeros(num_edge_types))
        self.register_buffer("gate_counts_per_type", torch.zeros(num_edge_types))

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
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Encodes the graph nodes into contextual representations.
        """
        # Node metadata embeddings
        meta = torch.cat([
            self.node_type_emb(node_types),
            self.attr_type_emb(attr_types),
            self.status_emb(statuses),
            self.date_bucket_emb(date_buckets),
        ], dim=-1)

        x = torch.cat([node_text_embs, meta], dim=-1)
        h = self.input_proj(x)

        if not self.use_gnn or edge_index.size(1) == 0:
            return h, {"mean_gate": 1.0}

        # Build full edge features = [continuous_edge_attr (10), edge_type_embedding (8)]
        e_type_emb = self.edge_type_emb(edge_types)
        full_edge_feats = torch.cat([edge_attrs, e_type_emb], dim=-1)

        last_gates = None
        for layer in self.layers:
            h, gates = layer(h, edge_index, full_edge_feats)
            last_gates = gates

        # Log mean gates per edge type during forward passes
        if self.training and last_gates is not None and last_gates.numel() > 0:
            with torch.no_grad():
                for et in range(self.edge_type_emb.num_embeddings):
                    mask = (edge_types == et)
                    if mask.any():
                        g_mean = last_gates[mask].mean()
                        cnt = mask.sum()
                        old_sum = self.mean_gates_per_type[et] * self.gate_counts_per_type[et]
                        self.gate_counts_per_type[et] += cnt
                        self.mean_gates_per_type[et] = (old_sum + g_mean * cnt) / self.gate_counts_per_type[et]

        return h, {"mean_gate": float(last_gates.mean().item()) if last_gates is not None and last_gates.numel() > 0 else 1.0}
