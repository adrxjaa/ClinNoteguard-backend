from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sentence_transformers import SentenceTransformer

from verify.eval.metrics import VERIFICATION_CLASSES, compute_verification_metrics

class MajorityClassBaseline:
    """Baseline A0: Always predicts the training set majority class."""

    def __init__(self):
        self.majority_label = VERIFICATION_CLASSES[0]

    def fit(self, y_train: List[str]):
        counts: Dict[str, int] = {}
        for y in y_train:
            counts[y] = counts.get(y, 0) + 1
        self.majority_label = max(counts.items(), key=lambda x: x[1])[0]

    def predict(self, num_samples: int) -> List[str]:
        return [self.majority_label] * num_samples


class MLPBaselineModel(nn.Module):
    """
    Baseline A1 / A2: 2-layer MLP on concatenated features:
    [claim_embedding (384), top1_evidence_embedding (384), ce_score (1), dose_tag (3)]
    Total input dimension = 772
    """

    def __init__(self, in_features: int = 772, hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineFeatureExtractor:
    """Extracts features for MLP baselines."""

    def __init__(self, encoder: Optional[SentenceTransformer] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder or SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=self.device)

    def extract_features(
        self,
        claims: List[Dict[str, Any]],
        candidates_by_claim: Dict[str, List[Dict[str, Any]]],
        recency_only: bool = False,
    ) -> Tuple[torch.Tensor, List[str]]:
        features = []
        labels = []

        claim_texts = [c["claim_text"] for c in claims]
        claim_embs = self.encoder.encode(claim_texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

        for i, c in enumerate(claims):
            cid = c.get("claim_id", "")
            cands = candidates_by_claim.get(cid, [])

            if recency_only and cands:
                # Filter strictly to the newest candidate by valid_from / timestamp
                sorted_cands = sorted(
                    cands,
                    key=lambda x: x.get("valid_from") or x.get("timestamp") or "",
                    reverse=True,
                )
                top1_cand = sorted_cands[0]
            elif cands:
                top1_cand = cands[0]
            else:
                top1_cand = {}

            cand_text = top1_cand.get("claim_text", "")
            if cand_text:
                cand_emb = self.encoder.encode([cand_text], convert_to_numpy=True, show_progress_bar=False)[0]
            else:
                cand_emb = np.zeros(384, dtype=np.float32)

            ce_score = float(top1_cand.get("cross_encoder_score", top1_cand.get("score", 0.0)))
            dose_tag = top1_cand.get("dose_agreement", "n/a").lower()
            dose_one_hot = [
                1.0 if dose_tag == "match" else 0.0,
                1.0 if dose_tag == "mismatch" else 0.0,
                1.0 if dose_tag not in ["match", "mismatch"] else 0.0,
            ]

            feat_vec = np.concatenate([claim_embs[i], cand_emb, [ce_score], dose_one_hot])
            features.append(feat_vec)
            labels.append(c.get("label", VERIFICATION_CLASSES[0]))

        X = torch.tensor(np.array(features), dtype=torch.float32)
        return X, labels


def train_mlp_baseline(
    X_train: torch.Tensor,
    y_train_str: List[str],
    X_val: torch.Tensor,
    y_val_str: List[str],
    epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Tuple[MLPBaselineModel, Dict[str, Any]]:
    """
    Trains MLP baseline with AdamW and class-weighted cross-entropy.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    label2idx = {lbl: idx for idx, lbl in enumerate(VERIFICATION_CLASSES)}
    idx2label = {idx: lbl for idx, lbl in enumerate(VERIFICATION_CLASSES)}

    y_train = torch.tensor([label2idx[y] for y in y_train_str], dtype=torch.long)
    y_val = torch.tensor([label2idx[y] for y in y_val_str], dtype=torch.long)

    # Class weights for balanced loss
    class_counts = torch.bincount(y_train, minlength=len(VERIFICATION_CLASSES)).float()
    class_weights = (len(y_train) / (len(VERIFICATION_CLASSES) * torch.clamp(class_counts, min=1.0))).to(dev)

    model = MLPBaselineModel(in_features=X_train.shape[1], num_classes=len(VERIFICATION_CLASSES)).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(dev))
            val_preds = [idx2label[int(idx)] for idx in val_out.argmax(dim=-1).cpu().numpy()]
            val_metrics = compute_verification_metrics(y_val_str, val_preds)
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
        model.to(dev)

    # Final validation predictions
    model.eval()
    with torch.no_grad():
        final_val_out = model(X_val.to(dev))
        final_val_preds = [idx2label[int(idx)] for idx in final_val_out.argmax(dim=-1).cpu().numpy()]
        final_metrics = compute_verification_metrics(y_val_str, final_val_preds)

    return model, final_metrics
