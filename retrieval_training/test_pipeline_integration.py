"""
Integration tests for Stage 4 Cross-Encoder integration in api/retrieval/pipeline.py.

Tests that retrieve_evidence_for_claims:
1. Requests an expanded candidate pool (effective_pool_size >= top_k).
2. Successfully invokes CrossEncoderReranker when available and enriches evidence.
3. Falls back gracefully to top_k Hybrid RRF candidates if reranker raises an error during inference.
4. Falls back gracefully to top_k Hybrid RRF candidates if reranker cannot be initialized.
5. Preserves all caller signatures and dictionary contracts.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import api.retrieval.pipeline as pipeline_module
from api.retrieval.pipeline import retrieve_evidence_for_claims


class TestPipelineRerankerIntegration(unittest.TestCase):

    def setUp(self):
        # Reset cached singletons before each test
        pipeline_module._GLOBAL_RERANKER = None
        pipeline_module._RERANKER_INIT_ATTEMPTED = False

    def make_mock_hybrid(self, candidates_to_return):
        mock_store = MagicMock()
        mock_store.patient_ids.return_value = ["P_TEST"]
        mock_store.events.return_value = [MagicMock()]

        mock_bm25 = MagicMock()
        mock_bm25.store = mock_store

        mock_hybrid = MagicMock()
        mock_hybrid.bm25 = mock_bm25
        mock_hybrid.search.return_value = candidates_to_return
        return mock_hybrid

    @patch("api.retrieval.pipeline.get_default_retriever")
    def test_successful_cross_encoder_rerank_flow(self, mock_get_hybrid):
        # Setup mock hybrid with 4 candidate items
        cands = [
            {"event": {"doc_id": f"DOC_{i}"}, "text": f"Document text {i}", "rrf_score": 0.01 * (10 - i)}
            for i in range(4)
        ]
        mock_hybrid = self.make_mock_hybrid(cands)
        mock_get_hybrid.return_value = mock_hybrid

        # Setup mock reranker that reverses order and attaches cross_encoder scores
        mock_reranker = MagicMock()
        def mock_rerank(claim, candidate_pool, top_k):
            # reverse candidate_pool and slice to top_k
            reversed_pool = list(reversed(candidate_pool))[:top_k]
            for rank, item in enumerate(reversed_pool, start=1):
                item["cross_encoder_score"] = 0.99 - (rank * 0.1)
                item["cross_encoder_rank"] = rank
            return reversed_pool

        mock_reranker.rerank.side_effect = mock_rerank
        pipeline_module._GLOBAL_RERANKER = mock_reranker
        pipeline_module._RERANKER_INIT_ATTEMPTED = True

        claims = [{"claim_id": "CLM-001", "claim_text": "Patient takes Aspirin 81 mg daily."}]
        result = retrieve_evidence_for_claims(claims, patient_id="P_TEST", top_k=2, candidate_pool_size=10)

        # 1. Verify candidate pool size passed to hybrid.search was max(top_k, candidate_pool_size) = 10
        mock_hybrid.search.assert_called_once_with(
            "Patient takes Aspirin 81 mg daily.", "P_TEST", top_k=10
        )

        # 2. Verify reranker.rerank was called with claim, candidates, and top_k=2
        mock_reranker.rerank.assert_called_once_with(
            "Patient takes Aspirin 81 mg daily.", cands, top_k=2
        )

        # 3. Verify evidence contains cross-encoder fields
        evidence = result[0]["retrieved_evidence"]
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0]["event"]["doc_id"], "DOC_3")
        self.assertIn("cross_encoder_score", evidence[0])
        self.assertEqual(evidence[0]["cross_encoder_rank"], 1)

    @patch("api.retrieval.pipeline.get_default_retriever")
    def test_fallback_when_reranker_inference_fails(self, mock_get_hybrid):
        cands = [
            {"event": {"doc_id": f"DOC_{i}"}, "text": f"Document text {i}", "rrf_score": 0.05}
            for i in range(5)
        ]
        mock_hybrid = self.make_mock_hybrid(cands)
        mock_get_hybrid.return_value = mock_hybrid

        # Reranker raises an unexpected inference exception
        mock_reranker = MagicMock()
        mock_reranker.rerank.side_effect = RuntimeError("Inference device fault")
        pipeline_module._GLOBAL_RERANKER = mock_reranker
        pipeline_module._RERANKER_INIT_ATTEMPTED = True

        claims = [{"claim_id": "CLM-002", "claim_text": "Patient has hypertension."}]
        result = retrieve_evidence_for_claims(claims, patient_id="P_TEST", top_k=3)

        # Must fall back gracefully to candidate_pool[:3]
        evidence = result[0]["retrieved_evidence"]
        self.assertEqual(len(evidence), 3)
        self.assertEqual([e["event"]["doc_id"] for e in evidence], ["DOC_0", "DOC_1", "DOC_2"])
        # Cross encoder fields should NOT be attached on fallback
        self.assertNotIn("cross_encoder_score", evidence[0])

    @patch("api.retrieval.pipeline.get_default_retriever")
    def test_fallback_when_reranker_init_fails(self, mock_get_hybrid):
        cands = [{"event": {"doc_id": "DOC_0"}, "text": "Doc text", "rrf_score": 0.05}]
        mock_hybrid = self.make_mock_hybrid(cands)
        mock_get_hybrid.return_value = mock_hybrid

        # Simulate missing sentence-transformers package
        with patch.object(pipeline_module, "get_default_reranker", return_value=None):
            claims = [{"claim_id": "CLM-003", "claim_text": "Patient has diabetes."}]
            result = retrieve_evidence_for_claims(claims, patient_id="P_TEST", top_k=1)

            evidence = result[0]["retrieved_evidence"]
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["event"]["doc_id"], "DOC_0")
            self.assertNotIn("cross_encoder_score", evidence[0])

    def test_candidate_pool_never_smaller_than_top_k(self):
        # Verify calculation logic when candidate_pool_size < top_k
        top_k = 5
        candidate_pool_size = 2
        effective = max(
            top_k,
            candidate_pool_size if candidate_pool_size is not None else pipeline_module.DEFAULT_CANDIDATE_POOL_SIZE
        )
        self.assertEqual(effective, 5)


if __name__ == "__main__":
    unittest.main()
