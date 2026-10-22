"""
BM25 (lexical) retriever, patient-scoped candidates with CORPUS-WIDE statistics.

Why corpus-wide IDF?
  A single patient has only a handful of EHR events. With IDF computed on 3-5 documents,
  nearly every term appears in "most" documents, so IDF collapses to ~0 (or negative in
  rank_bm25) and the doses / drug names that actually discriminate (e.g. 'aspirin 200 mg'
  vs 'aspirin 25 mg') get no weight. We therefore compute document frequencies and average
  length over ALL events in the store, but only RANK the events of the requested patient,
  so there is still no cross-patient leakage.

Okapi BM25 with the non-negative IDF  log(1 + (N - n + 0.5) / (n + 0.5)).
"""

import math
from collections import Counter
from typing import Dict, List, Tuple

from .ehr_store import EHREvent, EHRStore
from .text_utils import tokenize


class BM25Retriever:
    def __init__(self, store: EHRStore, k1: float = 1.5, b: float = 0.75, global_idf: bool = True):
        self.store = store
        self.k1, self.b, self.global_idf = k1, b, global_idf
        self._tok: Dict[str, List[Tuple[EHREvent, List[str]]]] = {}
        self._stats = None  # (N, avgdl, df) over the whole corpus

    def _docs(self, patient_id: str):
        if patient_id not in self._tok:
            self._tok[patient_id] = [(e, tokenize(e.text)) for e in self.store.events(patient_id)]
        return self._tok[patient_id]

    def _corpus_stats(self, patient_id: str):
        if self.global_idf:
            if self._stats is None:
                toks = [t for pid in self.store.patient_ids() for _, t in self._docs(pid)]
                self._stats = self._compute(toks)
            return self._stats
        return self._compute([t for _, t in self._docs(patient_id)])

    @staticmethod
    def _compute(token_lists):
        n = len(token_lists)
        df = Counter()
        for t in token_lists:
            df.update(set(t))
        avgdl = sum(len(t) for t in token_lists) / max(1, n)
        return n, avgdl or 1.0, df

    def search(self, claim: str, patient_id: str, top_k: int = 20) -> List[Tuple[EHREvent, float]]:
        docs = self._docs(patient_id)
        q = set(tokenize(claim))
        if not docs or not q:
            return []
        n, avgdl, df = self._corpus_stats(patient_id)
        scored = []
        for ev, toks in docs:
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avgdl))
            scored.append((ev, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
