"""
Main Retrieval Pipeline Interface (Stage 3 in ClinNoteGuard).

Enriches extracted atomic claims with candidate evidence from the patient's EHR history
using the Hybrid (BM25 + Dense + RRF) retriever with dose-conflict awareness.
"""

import os
import logging
from typing import List, Dict, Any, Optional

from .ehr_store import EHRStore, EHREvent
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever, default_model_path
from .hybrid import HybridRetriever

logger = logging.getLogger("RetrievalPipeline")

# Global singleton store, hybrid retriever, and cross-encoder reranker
_GLOBAL_STORE: Optional[EHRStore] = None
_GLOBAL_HYBRID: Optional[HybridRetriever] = None
_GLOBAL_RERANKER: Optional[Any] = None
_RERANKER_INIT_ATTEMPTED: bool = False

DEFAULT_CANDIDATE_POOL_SIZE = int(os.environ.get("RERANK_POOL_SIZE", "20"))


def get_default_reranker():
    """
    Lazily initializes and returns the cached singleton CrossEncoderReranker.
    If the model or dependency cannot be loaded, logs an explicit error and returns None.
    """
    global _GLOBAL_RERANKER, _RERANKER_INIT_ATTEMPTED
    if not _RERANKER_INIT_ATTEMPTED:
        _RERANKER_INIT_ATTEMPTED = True
        try:
            from .reranker import CrossEncoderReranker
            logger.info("Initializing Global CrossEncoderReranker...")
            _GLOBAL_RERANKER = CrossEncoderReranker()
            logger.info("CrossEncoderReranker initialized successfully.")
        except Exception as err:
            logger.error(
                f"Failed to initialize CrossEncoderReranker: {err}. "
                "Pipeline will fall back to Stage 3 Hybrid RRF candidate ranking."
            )
            _GLOBAL_RERANKER = None
    return _GLOBAL_RERANKER


def get_default_retriever() -> HybridRetriever:
    """Initializes or returns the cached singleton HybridRetriever."""
    global _GLOBAL_STORE, _GLOBAL_HYBRID
    if _GLOBAL_HYBRID is None:
        logger.info("Initializing Global EHRStore and HybridRetriever...")
        _GLOBAL_STORE = EHRStore()

        # 1. Load synthetic complex benchmark if available
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        bench_file = os.path.join(base_dir, "retrieval_data", "synthetic_complex_benchmark.json")
        if os.path.exists(bench_file):
            try:
                bench_store = EHRStore.from_claims_json(bench_file)
                for pid in bench_store.patient_ids():
                    for ev in bench_store.events(pid):
                        _GLOBAL_STORE.add(ev)
                logger.info(f"Loaded {len(_GLOBAL_STORE.all_events())} benchmark EHR claims into retrieval store.")
            except Exception as e:
                logger.warning(f"Could not load benchmark JSON into store: {e}")

        # 2. Also try loading standard EHR dataset if available
        try:
            csv_store = EHRStore.from_csv(["all"])
            for pid in csv_store.patient_ids():
                for ev in csv_store.events(pid):
                    _GLOBAL_STORE.add(ev)
            logger.info(f"Merged CSV EHR dataset. Total indexed events: {len(_GLOBAL_STORE.all_events())}")
        except Exception:
            pass

        # 3. Instantiate BM25 and Dense bi-encoder
        bm25 = BM25Retriever(_GLOBAL_STORE)
        dense = DenseRetriever(_GLOBAL_STORE, default_model_path())
        _GLOBAL_HYBRID = HybridRetriever(bm25, dense)
        logger.info("HybridRetriever initialized successfully.")

    return _GLOBAL_HYBRID


def retrieve_evidence_for_claims(
    claims: List[Dict[str, Any]],
    patient_id: Optional[str] = None,
    session_code: Optional[str] = None,
    top_k: int = 3,
    candidate_pool_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Enriches atomic claims with candidate supporting/conflicting EHR evidence.
    
    Retrieval Flow (Stage 3 Hybrid + Stage 4 Cross-Encoder):
    1. Fetches an expanded candidate pool (default 20) via HybridRetriever (BM25 + Dense + RRF).
    2. Reranks the candidate pool using CrossEncoderReranker and truncates to top_k.
    3. If the CrossEncoder is unavailable or fails, falls back gracefully to top_k RRF candidates.
    
    If the patient has no EHR history yet (e.g. first encounter), sets
    'retrieved_evidence' to [] and flags 'ehr_status' as 'NO_PRIOR_EHR'.
    """
    if not claims:
        return []

    effective_pool_size = max(
        top_k,
        candidate_pool_size if candidate_pool_size is not None else DEFAULT_CANDIDATE_POOL_SIZE,
    )

    try:
        hybrid = get_default_retriever()
        store = hybrid.bm25.store
    except Exception as init_err:
        logger.error(f"Failed to get hybrid retriever: {init_err}")
        for c in claims:
            c.setdefault("retrieved_evidence", [])
        return claims

    # Resolve target patient identifier in the EHR store
    target_pid = None
    if patient_id and patient_id in store.patient_ids():
        target_pid = patient_id
    elif patient_id and f"P_{patient_id}" in store.patient_ids():
        target_pid = f"P_{patient_id}"
    elif session_code:
        # Check by session code / encounter code
        possible_pids = [f"P_{session_code}", f"P_D2N{session_code}"]
        for p in possible_pids:
            if p in store.patient_ids():
                target_pid = p
                break

    # If still no direct match, check if patient has previous ended sessions in DB to dynamically build EHR claims
    if not target_pid and patient_id:
        try:
            from api.models import ConsultationSession
            past_sessions = ConsultationSession.objects.filter(
                patient__id=patient_id if str(patient_id).isdigit() else None,
                status=ConsultationSession.Status.ENDED
            ).exclude(code=session_code).order_by('-created_at')[:5]

            dynamic_pid = f"PATIENT_{patient_id}"
            has_dynamic_history = False
            for past_s in past_sessions:
                if past_s.atomic_claims_json:
                    for past_c in past_s.atomic_claims_json:
                        ev = EHREvent(
                            doc_id=f"PAST_{past_s.code}_{past_c.get('claim_id', 'CLM')}",
                            patient_id=dynamic_pid,
                            encounter_id=past_s.code,
                            relative_time=f"Encounter {past_s.created_at.strftime('%Y-%m-%d')}",
                            event_type=past_c.get("section", "PriorNote"),
                            description=past_c.get("claim_text", ""),
                            source_basis="prior_encounter"
                        )
                        store.add(ev)
                        has_dynamic_history = True
            if has_dynamic_history:
                target_pid = dynamic_pid
        except Exception as dyn_err:
            logger.debug(f"Could not build dynamic history from past sessions: {dyn_err}")

    # Process each claim
    for claim in claims:
        claim_text = claim.get("claim_text", "")
        if not target_pid or len(store.events(target_pid)) == 0:
            claim["retrieved_evidence"] = []
            claim["ehr_status"] = "NO_PRIOR_EHR"
            continue

        claim["ehr_status"] = "EHR_SEARCHED"
        try:
            # 1. Retrieve expanded candidate pool from Hybrid RRF
            candidate_pool = hybrid.search(claim_text, target_pid, top_k=effective_pool_size)
            if not candidate_pool:
                claim["retrieved_evidence"] = []
                continue

            # 2. Attempt Stage 4 Cross-Encoder Reranking
            reranker = get_default_reranker()
            if reranker is not None:
                try:
                    reranked = reranker.rerank(claim_text, candidate_pool, top_k=top_k)
                    claim["retrieved_evidence"] = reranked
                    logger.info(
                        f"Claim {claim.get('claim_id')}: Successfully reranked top {len(reranked)} "
                        f"evidence candidates using CrossEncoder (from pool of {len(candidate_pool)})."
                    )
                except Exception as rerank_err:
                    logger.error(
                        f"Claim {claim.get('claim_id')}: CrossEncoder reranking failed ({rerank_err}). "
                        f"Falling back to top-{top_k} Hybrid RRF candidates."
                    )
                    claim["retrieved_evidence"] = candidate_pool[:top_k]
            else:
                logger.info(
                    f"Claim {claim.get('claim_id')}: CrossEncoderReranker not active. "
                    f"Using top-{top_k} Hybrid RRF fallback candidates."
                )
                claim["retrieved_evidence"] = candidate_pool[:top_k]

        except Exception as search_err:
            logger.error(f"Search failed for claim {claim.get('claim_id')}: {search_err}")
            claim["retrieved_evidence"] = []

    return claims
