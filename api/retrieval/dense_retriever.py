"""
Dense (semantic) bi-encoder retriever, patient-scoped.

Claim and EHR event are encoded independently by a sentence-transformer and compared
by cosine similarity. The model name/path is configurable so we can swap in the
fine-tuned checkpoint (see retrieval_training/train_dense.py) or a clinical encoder.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ehr_store import EHREvent, EHRStore

DEFAULT_DENSE_MODEL = os.environ.get("DENSE_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FINETUNED_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "retrieval_models", "dense_finetuned")
)


def default_model_path() -> str:
    """Prefer the fine-tuned checkpoint when it exists."""
    return FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else DEFAULT_DENSE_MODEL


class DenseRetriever:
    def __init__(self, store: EHRStore, model_name_or_path: Optional[str] = None, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        self.store = store
        self.model_name = model_name_or_path or default_model_path()
        self.model = SentenceTransformer(self.model_name, device=device)
        self._cache: Dict[str, Tuple[np.ndarray, List[EHREvent]]] = {}

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    def _index(self, patient_id: str):
        if patient_id not in self._cache:
            events = self.store.events(patient_id)
            emb = self._embed([e.text for e in events]) if events else np.zeros((0, 1))
            self._cache[patient_id] = (emb, events)
        return self._cache[patient_id]

    def search(self, claim: str, patient_id: str, top_k: int = 20) -> List[Tuple[EHREvent, float]]:
        emb, events = self._index(patient_id)
        if not events:
            return []
        q = self._embed([claim])[0]
        scores = emb @ q
        order = np.argsort(-scores)[:top_k]
        return [(events[i], float(scores[i])) for i in order]
