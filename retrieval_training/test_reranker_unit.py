"""
Unit tests for CrossEncoderReranker (api/retrieval/reranker.py).

Tests input validation, scoring, sorting, tie-breaking, text extraction,
and error handling using a fake/mock model without downloading model weights
or initializing external dependencies.
"""

import copy
import math
import os
import sys
import unittest
from typing import Any, Dict, List

# Ensure ClinNoteguard-backend root is in sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    """Mock CrossEncoder that avoids downloading weights and network access."""

    def __init__(self, fixed_scores=None, score_fn=None, side_effect=None):
        self.fixed_scores = fixed_scores
        self.score_fn = score_fn
        self.side_effect = side_effect
        self.call_count = 0
        self.last_pairs = None

    def predict(self, pairs):
        self.call_count += 1
        self.last_pairs = pairs
        if self.side_effect is not None:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            if callable(self.side_effect):
                return self.side_effect(pairs)
            raise RuntimeError(f"Unexpected side_effect: {self.side_effect}")

        if self.fixed_scores is not None:
            return list(self.fixed_scores)

        if self.score_fn is not None:
            return [self.score_fn(pair) for pair in pairs]

        return [0.5 for _ in pairs]


def create_test_reranker(fake_model: FakeCrossEncoder) -> CrossEncoderReranker:
    """Instantiate CrossEncoderReranker without calling __init__ to bypass model loading."""
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker.model = fake_model
    reranker.model_name = "mock-cross-encoder"
    reranker.device = "cpu"
    reranker.max_length = 512
    return reranker


def make_candidate(
    doc_id: str = "EHR_C01",
    text: str = "Medication: Aspirin 81 mg daily. [6 months ago]",
    description: str = "Aspirin 81 mg daily",
    rrf_score: float = 0.0325,
    dose_agreement: str = "match",
    bm25_rank: int = 1,
    bm25_score: float = 14.2,
    dense_rank: int = 2,
    dense_score: float = 0.81,
) -> Dict[str, Any]:
    """Creates a sample candidate dictionary matching HybridRetriever.search() format."""
    return {
        "event": {
            "doc_id": doc_id,
            "patient_id": "P_SYNTH_01",
            "relative_time": "T-6 months",
            "event_type": "Medication",
            "description": description,
            "diagnosis": "",
            "medication": "Aspirin",
            "investigation": "",
        },
        "text": text,
        "rrf_score": rrf_score,
        "dose_agreement": dose_agreement,
        "bm25_rank": bm25_rank,
        "bm25_score": bm25_score,
        "dense_rank": dense_rank,
        "dense_score": dense_score,
    }


class TestCrossEncoderReranker(unittest.TestCase):
    """Test suite covering the 14 core validation and reranking specifications."""

    def test_01_descending_ranking_distinct_scores(self):
        """1. Correct descending ranking when the fake model returns distinct scores."""
        c1 = make_candidate("DOC_1", text="Candidate 1")
        c2 = make_candidate("DOC_2", text="Candidate 2")
        c3 = make_candidate("DOC_3", text="Candidate 3")

        # Model returns scores: C1 -> 0.1, C2 -> 0.9, C3 -> 0.5
        fake_model = FakeCrossEncoder(fixed_scores=[0.1, 0.9, 0.5])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Test claim", [c1, c2, c3], top_k=3)

        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0]["event"]["doc_id"], "DOC_2")
        self.assertEqual(reranked[0]["cross_encoder_score"], 0.9)
        self.assertEqual(reranked[0]["cross_encoder_rank"], 1)

        self.assertEqual(reranked[1]["event"]["doc_id"], "DOC_3")
        self.assertEqual(reranked[1]["cross_encoder_score"], 0.5)
        self.assertEqual(reranked[1]["cross_encoder_rank"], 2)

        self.assertEqual(reranked[2]["event"]["doc_id"], "DOC_1")
        self.assertEqual(reranked[2]["cross_encoder_score"], 0.1)
        self.assertEqual(reranked[2]["cross_encoder_rank"], 3)

    def test_02_top_k_truncation(self):
        """2. Correct top_k truncation when top_k < len(candidates)."""
        c1 = make_candidate("DOC_1")
        c2 = make_candidate("DOC_2")
        c3 = make_candidate("DOC_3")

        fake_model = FakeCrossEncoder(fixed_scores=[0.2, 0.8, 0.6])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Test claim", [c1, c2, c3], top_k=2)

        self.assertEqual(len(reranked), 2)
        self.assertEqual([r["event"]["doc_id"] for r in reranked], ["DOC_2", "DOC_3"])
        self.assertEqual([r["cross_encoder_rank"] for r in reranked], [1, 2])

    def test_03_preservation_of_all_original_candidate_fields(self):
        """3. Preservation of all original candidate fields, including event metadata and retriever stats."""
        c = make_candidate(
            doc_id="DOC_SPECIAL",
            text="Medication: Lisinopril 10 mg. [1 year ago]",
            description="Lisinopril 10 mg daily",
            rrf_score=0.04123,
            dose_agreement="mismatch",
            bm25_rank=3,
            bm25_score=11.234,
            dense_rank=1,
            dense_score=0.9123,
        )

        fake_model = FakeCrossEncoder(fixed_scores=[0.8888])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Patient takes lisinopril 20 mg", [c], top_k=1)
        item = reranked[0]

        # Verify all original keys exist with uncorrupted values
        self.assertEqual(item["event"]["doc_id"], "DOC_SPECIAL")
        self.assertEqual(item["event"]["description"], "Lisinopril 10 mg daily")
        self.assertEqual(item["event"]["medication"], "Aspirin")  # from make_candidate default
        self.assertEqual(item["text"], "Medication: Lisinopril 10 mg. [1 year ago]")
        self.assertEqual(item["rrf_score"], 0.04123)
        self.assertEqual(item["dose_agreement"], "mismatch")
        self.assertEqual(item["bm25_rank"], 3)
        self.assertEqual(item["bm25_score"], 11.234)
        self.assertEqual(item["dense_rank"], 1)
        self.assertEqual(item["dense_score"], 0.9123)

        # Verify new fields were properly appended
        self.assertEqual(item["cross_encoder_score"], 0.8888)
        self.assertEqual(item["cross_encoder_rank"], 1)

    def test_04_immutability_of_input_candidates(self):
        """4. Input candidates and their nested dictionaries are not mutated."""
        orig_cand = make_candidate("DOC_IMMUTABLE")
        orig_cand_copy = copy.deepcopy(orig_cand)

        fake_model = FakeCrossEncoder(fixed_scores=[0.95])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Test claim", [orig_cand], top_k=1)

        # Mutate the output dictionary to verify independence
        reranked[0]["cross_encoder_score"] = -999.0
        reranked[0]["event"]["doc_id"] = "MUTATED"

        # The input candidate must remain completely identical to orig_cand_copy
        self.assertEqual(orig_cand, orig_cand_copy)
        self.assertNotIn("cross_encoder_score", orig_cand)
        self.assertNotIn("cross_encoder_rank", orig_cand)
        self.assertEqual(orig_cand["event"]["doc_id"], "DOC_IMMUTABLE")

    def test_05_equal_scores_deterministic_ordering(self):
        """5. Equal scores preserve the original candidate order deterministically."""
        c1 = make_candidate("FIRST_DOC")
        c2 = make_candidate("SECOND_DOC")
        c3 = make_candidate("THIRD_DOC")

        fake_model = FakeCrossEncoder(fixed_scores=[0.75, 0.75, 0.75])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Test claim", [c1, c2, c3], top_k=3)

        self.assertEqual([r["event"]["doc_id"] for r in reranked], ["FIRST_DOC", "SECOND_DOC", "THIRD_DOC"])
        self.assertEqual([r["cross_encoder_rank"] for r in reranked], [1, 2, 3])

    def test_06_empty_candidates_returns_empty_list(self):
        """6. Empty candidate list returns an empty list without calling model."""
        fake_model = FakeCrossEncoder()
        reranker = create_test_reranker(fake_model)

        res = reranker.rerank("Test claim", [], top_k=3)
        self.assertEqual(res, [])
        self.assertEqual(fake_model.call_count, 0)

    def test_07_invalid_top_k_rejection(self):
        """7. Invalid top_k values are rejected, including True, False, zero, negative, floats, and strings."""
        fake_model = FakeCrossEncoder()
        reranker = create_test_reranker(fake_model)
        candidates = [make_candidate()]

        # Booleans must raise TypeError (even though bool inherits from int)
        with self.assertRaises(TypeError):
            reranker.rerank("claim", candidates, top_k=True)
        with self.assertRaises(TypeError):
            reranker.rerank("claim", candidates, top_k=False)

        # Non-integers must raise TypeError
        with self.assertRaises(TypeError):
            reranker.rerank("claim", candidates, top_k=2.5)
        with self.assertRaises(TypeError):
            reranker.rerank("claim", candidates, top_k="3")
        with self.assertRaises(TypeError):
            reranker.rerank("claim", candidates, top_k=None)

        # Values <= 0 must raise ValueError
        with self.assertRaises(ValueError):
            reranker.rerank("claim", candidates, top_k=0)
        with self.assertRaises(ValueError):
            reranker.rerank("claim", candidates, top_k=-1)

    def test_08_invalid_claims_rejection(self):
        """8. Empty or invalid claims are rejected with ValueError."""
        fake_model = FakeCrossEncoder()
        reranker = create_test_reranker(fake_model)
        candidates = [make_candidate()]

        invalid_claims = ["", "   ", "\n\t", None, 12345, []]
        for bad_claim in invalid_claims:
            with self.subTest(claim=bad_claim):
                with self.assertRaises(ValueError):
                    reranker.rerank(bad_claim, candidates, top_k=1)

    def test_09_invalid_candidate_container_and_records_rejection(self):
        """9. Invalid candidate container types and malformed candidate records are rejected."""
        fake_model = FakeCrossEncoder()
        reranker = create_test_reranker(fake_model)

        # Container is not a list
        with self.assertRaises(TypeError):
            reranker.rerank("claim", "not-a-list", top_k=1)
        with self.assertRaises(TypeError):
            reranker.rerank("claim", {"cand": 1}, top_k=1)

        # Element inside candidates is not a dict
        with self.assertRaises(TypeError):
            reranker.rerank("claim", [make_candidate(), "not-a-dict"], top_k=2)
        with self.assertRaises(TypeError):
            reranker.rerank("claim", [123], top_k=1)

    def test_10_missing_text_falls_back_to_event_description(self):
        """10. Missing text falls back to event.description."""
        c = make_candidate(text="", description="Description fallback text for aspirin")
        del c["text"]  # Explicitly remove 'text' key

        fake_model = FakeCrossEncoder(fixed_scores=[0.85])
        reranker = create_test_reranker(fake_model)

        reranked = reranker.rerank("Patient takes aspirin", [c], top_k=1)
        self.assertEqual(len(reranked), 1)

        # Check what pair was passed to the model
        self.assertEqual(fake_model.call_count, 1)
        claim_arg, doc_arg = fake_model.last_pairs[0]
        self.assertEqual(claim_arg, "Patient takes aspirin")
        self.assertEqual(doc_arg, "Description fallback text for aspirin")

    def test_11_missing_usable_document_text_raises_error(self):
        """11. Missing usable document text in both text and event.description raises a clear error."""
        fake_model = FakeCrossEncoder()
        reranker = create_test_reranker(fake_model)

        # Case A: text is None, event.description is None
        c1 = {"event": {"description": None, "doc_id": "BAD_1"}, "text": None}
        with self.assertRaises(ValueError) as ctx:
            reranker.rerank("claim", [c1], top_k=1)
        self.assertIn("BAD_1", str(ctx.exception))

        # Case B: text is whitespace, event.description is whitespace
        c2 = {"event": {"description": "   "}, "text": "   "}
        with self.assertRaises(ValueError):
            reranker.rerank("claim", [c2], top_k=1)

        # Case C: event dict is missing entirely and text is None
        c3 = {"text": None}
        with self.assertRaises(ValueError):
            reranker.rerank("claim", [c3], top_k=1)

    def test_12_mismatched_scores_count_raises_error(self):
        """12. A model returning the wrong number of scores raises ValueError."""
        c1 = make_candidate("DOC_1")
        c2 = make_candidate("DOC_2")

        # Model returns 1 score for 2 candidates
        fake_model = FakeCrossEncoder(fixed_scores=[0.5])
        reranker = create_test_reranker(fake_model)

        with self.assertRaises(ValueError) as ctx:
            reranker.rerank("claim", [c1, c2], top_k=2)
        self.assertIn("Mismatch between number of candidates", str(ctx.exception))

    def test_13_non_finite_scores_rejection(self):
        """13. NaN and positive or negative infinity are rejected."""
        c1 = make_candidate("DOC_1")
        c2 = make_candidate("DOC_2")

        non_finite_cases = [
            [float("nan"), 0.5],
            [float("inf"), 0.5],
            [float("-inf"), 0.5],
            [0.5, float("nan")],
        ]

        for bad_scores in non_finite_cases:
            with self.subTest(scores=bad_scores):
                fake_model = FakeCrossEncoder(fixed_scores=bad_scores)
                reranker = create_test_reranker(fake_model)
                with self.assertRaises(ValueError) as ctx:
                    reranker.rerank("claim", [c1, c2], top_k=2)
                self.assertIn("non-finite", str(ctx.exception).lower())

    def test_14_inference_exceptions_propagate(self):
        """14. Model inference exceptions propagate rather than being silently suppressed."""
        c = make_candidate("DOC_1")

        fake_model = FakeCrossEncoder(side_effect=RuntimeError("CUDA out of memory in cross-encoder"))
        reranker = create_test_reranker(fake_model)

        with self.assertRaises(RuntimeError) as ctx:
            reranker.rerank("claim", [c], top_k=1)
        self.assertIn("CUDA out of memory", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
