"""
Stage 3B & Stage 4: Hybrid Clinical Retrieval & Cross-Encoder Reranking.
Combines BM25 lexical matching, fine-tuned dense MiniLM embeddings,
Reciprocal Rank Fusion (k=60), and ms-marco-MiniLM cross-encoder reranking.
"""

from typing import List, Dict, Any, Optional
from api.retrieval.pipeline import get_default_retriever, get_default_reranker
from api.retrieval.text_utils import dose_agreement

class Stage4ClinicalRetriever:
    """
    Stage 3B & 4 Service: Hybrid Retrieval + Cross-Encoder Reranker.
    Retrieves and ranks relevant EHR history and transcript evidence for a claim.
    """

    def __init__(self):
        self.retriever = get_default_retriever()
        self.reranker = get_default_reranker()

    def search(
        self,
        query: str,
        patient_records: List[Dict[str, Any]],
        top_k: int = 15,
        rerank_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval followed by cross-encoder reranking.
        """
        # Hybrid retrieve
        hybrid_hits = self.retriever.retrieve(query, patient_records, top_k=top_k)
        # Cross-encoder rerank
        ranked_hits = self.reranker.rerank(query, hybrid_hits, top_k=rerank_k)

        # Annotate dose agreement flag
        for hit in ranked_hits:
            hit["dose_agreement"] = dose_agreement(query, hit.get("claim_text", ""))

        return ranked_hits

def get_stage4_retriever() -> Stage4ClinicalRetriever:
    return Stage4ClinicalRetriever()
