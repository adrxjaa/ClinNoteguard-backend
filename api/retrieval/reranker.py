"""
Cross-Encoder Reranker for ClinNoteGuard (Stage 4).

Reranks candidate evidence retrieved from Stage 3 Hybrid (BM25 + Dense + RRF)
using a deep cross-attention transformer network over concatenated (claim, document) pairs:
    [CLS] claim [SEP] evidence [SEP]

Unlike bi-encoders which encode claim and document into separate vectors and compare
via cosine similarity, cross-encoders perform full all-to-all cross-attention across
all tokens of both the claim and document simultaneously. This allows deep contextual
interaction, better handling of qualifying clauses, look-alike drug names, and complex
syntactic dependencies.

NOTE ON CLINICAL VALIDATION:
----------------------------
The default model specified here ("cross-encoder/ms-marco-MiniLM-L-6-v2") is a general-domain
baseline pre-trained on web search passage ranking (MS MARCO). It is provided as an initial
engineering baseline and is NOT clinically validated. For clinical production deployment,
it should be replaced by a specialized clinical cross-encoder (e.g. Bio_ClinicalBERT cross-encoder)
or fine-tuned on clinical claim-evidence pairs.
"""

import copy
import logging
import math
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CrossEncoderReranker")

DEFAULT_CROSS_ENCODER_MODEL = os.environ.get(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


class CrossEncoderReranker:
    """
    Cross-Encoder Reranker for clinical atomic claim evidence candidates.

    Takes top candidate documents from HybridRetriever.search() and re-scores them
    using cross-attention relevance scoring.
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = 512,
    ):
        """
        Initializes and loads the cross-encoder model once per instance.

        Args:
            model_name_or_path: HuggingFace model hub ID or local checkpoint path.
                                Defaults to DEFAULT_CROSS_ENCODER_MODEL.
            device: Compute device ('cpu', 'cuda', 'mps', etc.). If None, auto-detected.
            max_length: Maximum sequence length for tokenization. Defaults to 512.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Please install it via `pip install sentence-transformers`."
            ) from exc

        self.model_name = model_name_or_path or DEFAULT_CROSS_ENCODER_MODEL
        self.device = device
        self.max_length = max_length

        logger.info(f"Loading CrossEncoder model: {self.model_name} on device: {self.device}...")
        # Do not catch or suppress exceptions here: let model loading errors surface immediately
        self.model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        logger.info(f"CrossEncoder model '{self.model_name}' loaded successfully.")

    def rerank(
        self,
        claim: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Reranks candidate evidence dictionaries by cross-encoder relevance.

        Args:
            claim: The atomic clinical claim assertion to be verified.
            candidates: List of candidate dicts as returned by HybridRetriever.search().
            top_k: Number of highest-ranked candidates to return. Must be a positive integer.

        Returns:
            List of top_k candidate dictionaries, sorted by descending cross_encoder_score,
            preserving all original fields and adding 'cross_encoder_score' and 'cross_encoder_rank'.

        Raises:
            TypeError: If top_k is not an integer (including booleans) or candidates is not a list of dicts.
            ValueError: If top_k <= 0, claim is empty/non-string, candidate text is missing,
                        or cross-encoder returns invalid/non-finite scores.
            Exception: Any scoring or model inference errors are propagated directly.
        """
        # 1. Validate top_k (strictly reject booleans, non-integers, and values <= 0)
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}: {top_k!r}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        # 2. Validate claim
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string.")

        # 3. Validate candidates container
        if not isinstance(candidates, list):
            raise TypeError(f"candidates must be a list of dictionaries, got {type(candidates).__name__}")

        if not candidates:
            return []

        # 4. Validate candidate dictionary structure and extract usable document text
        pairs: List[tuple] = []
        for idx, cand in enumerate(candidates):
            if not isinstance(cand, dict):
                raise TypeError(
                    f"Candidate at index {idx} must be a dictionary, got {type(cand).__name__}"
                )

            # Check candidate['text'] first
            doc_text = cand.get("text")
            if doc_text is None or (isinstance(doc_text, str) and not doc_text.strip()):
                # Fallback to event description
                event_dict = cand.get("event")
                if isinstance(event_dict, dict):
                    doc_text = event_dict.get("description")

            # Validate that doc_text is a non-empty string
            if doc_text is None or not isinstance(doc_text, str) or not doc_text.strip():
                cand_id = (
                    cand.get("event", {}).get("doc_id")
                    if isinstance(cand.get("event"), dict)
                    else None
                )
                id_info = f" (doc_id: {cand_id})" if cand_id else ""
                raise ValueError(
                    f"Candidate at index {idx}{id_info} does not have usable document text in 'text' "
                    "or 'event.description'."
                )

            pairs.append((claim.strip(), doc_text.strip()))

        # 5. Predict cross-encoder relevance scores for all pairs (propagate errors)
        raw_scores = self.model.predict(pairs)

        # 6. Verify exactly one score per candidate
        if len(raw_scores) != len(candidates):
            raise ValueError(
                f"Mismatch between number of candidates ({len(candidates)}) "
                f"and number of scores returned by model ({len(raw_scores)})."
            )

        # 7. Validate numerical scores and clone candidate dictionaries
        scored_candidates: List[tuple] = []
        for idx, (cand, raw_val) in enumerate(zip(candidates, raw_scores)):
            try:
                score = float(raw_val)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Cross-encoder score at index {idx} could not be converted to float: {raw_val!r}"
                ) from exc

            # Reject non-finite scores (NaN, +inf, -inf)
            if not math.isfinite(score):
                raise ValueError(
                    f"Cross-encoder returned a non-finite score ({score}) for candidate at index {idx}."
                )

            # Deep copy to guarantee no mutation of input dictionaries
            item = copy.deepcopy(cand)
            item["cross_encoder_score"] = round(score, 4)
            # Store tuple of (raw_score, original_index, item) for deterministic sorting on full precision
            scored_candidates.append((score, idx, item))

        # 8. Deterministic sorting on full-precision raw model scores:
        # Primary key: descending by raw cross-encoder score (-raw_score)
        # Tie-breaker key: ascending by original candidate index (preserves RRF input order)
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))

        # 9. Apply top_k cutoff and assign 1-based cross_encoder_rank
        top_results: List[Dict[str, Any]] = []
        for rank, (_, _, item) in enumerate(scored_candidates[:top_k], start=1):
            item["cross_encoder_rank"] = rank
            top_results.append(item)

        return top_results
