"""
Hybrid candidate retrieval (Stage 3): BM25 + Dense -> Reciprocal Rank Fusion (+ dose check).

RRF score(d) = sum_r 1 / (k + rank_r(d)),  k = 60 (Cormack et al., 2009)

Dose / number handling (both retrievers are weak here - dense encoders embed
'aspirin 25 mg' ~ 'aspirin 200 mg'; BM25 only sees tokens):
  * every candidate is tagged  dose_agreement = match | mismatch | n/a
  * 'match' gets a small additive bonus so the exact-dose event wins over same-drug
    different-dose events
  * 'mismatch' is NEVER dropped or penalised - "increase lisinopril to 40 mg" legitimately
    retrieves the old 10 mg event as evidence. The flag is passed on to the verifier,
    where it is a strong signal for the 'Clinical Conflict' label.

Output is the Top-N candidate list handed to the Cross-Encoder reranker (next stage).
"""

from typing import Dict, List

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .ehr_store import EHREvent
from .text_utils import dose_agreement

DOSE_MATCH_BONUS = 0.01  # >> typical rank gaps in RRF (~0.0003) but << a full rank-1 score (0.016) x 2


def reciprocal_rank_fusion(rankings: List[List[EHREvent]], k: int = 60) -> List[tuple]:
    """Ties are broken in favour of the LAST ranking passed in (dense), not insertion order."""
    scores: Dict[str, float] = {}
    docs: Dict[str, EHREvent] = {}
    last_rank: Dict[str, int] = {}
    for ranking in rankings:
        for rank, ev in enumerate(ranking, start=1):
            scores[ev.doc_id] = scores.get(ev.doc_id, 0.0) + 1.0 / (k + rank)
            docs[ev.doc_id] = ev
            last_rank[ev.doc_id] = rank
    fused = sorted(scores.items(), key=lambda x: (-x[1], last_rank[x[0]]))
    return [(docs[d], s) for d, s in fused]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60, per_retriever_k: int = 20,
                 dose_bonus: float = DOSE_MATCH_BONUS):
        self.bm25, self.dense = bm25, dense
        self.rrf_k, self.per_k, self.dose_bonus = rrf_k, per_retriever_k, dose_bonus

    def search_events(self, claim: str, patient_id: str, top_k: int = 20):
        """-> [(EHREvent, fused_score, dose_agreement, bm25_rank/score, dense_rank/score)]"""
        bm = self.bm25.search(claim, patient_id, self.per_k)
        de = self.dense.search(claim, patient_id, self.per_k)
        bm_info = {e.doc_id: (i + 1, s) for i, (e, s) in enumerate(bm)}
        de_info = {e.doc_id: (i + 1, s) for i, (e, s) in enumerate(de)}
        fused = reciprocal_rank_fusion([[e for e, _ in bm], [e for e, _ in de]], k=self.rrf_k)
        rows = []
        for ev, score in fused:
            dose = dose_agreement(claim, ev.text)
            if dose == "match":
                score += self.dose_bonus
            rows.append((ev, score, dose, bm_info.get(ev.doc_id), de_info.get(ev.doc_id)))
        rows.sort(key=lambda r: (-r[1], r[4][0] if r[4] else 10 ** 6))
        return rows[:top_k]

    def search(self, claim: str, patient_id: str, top_k: int = 20) -> List[dict]:
        out = []
        for ev, score, dose, bm, de in self.search_events(claim, patient_id, top_k):
            out.append({
                "event": ev.to_dict(),
                "text": ev.text,
                "rrf_score": round(score, 5),
                "dose_agreement": dose,
                "bm25_rank": bm[0] if bm else None,
                "bm25_score": round(bm[1], 4) if bm else None,
                "dense_rank": de[0] if de else None,
                "dense_score": round(de[1], 4) if de else None,
            })
        return out
