"""
ClinicalVerificationPipeline: Unified Multi-Stage Pipeline Orchestrator.
Allows Django views, background tasks, or scripts to execute the complete verification sequence:
Audio -> Diarized Transcript -> SOAP Note -> Atomic Claims -> Hybrid Retrieval -> Validity Graph -> Verifier.
"""

from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path

from stages.stage1_audio import get_stage1_diarizer, Stage1AudioDiarizer
from stages.stage2_soap import get_stage2_soap_generator, Stage2SOAPGenerator
from stages.stage3_claims import get_stage3_claim_extractor, Stage3ClaimExtractor
from stages.stage4_retrieval import get_stage4_retriever, Stage4ClinicalRetriever
from stages.stage5_concepts import get_stage5_concept_service, Stage5ConceptService
from stages.stage6_validity import get_stage6_validity_service, Stage6ValidityGraphService
from stages.stage7_8_verifier import get_stage7_8_verifier, Stage78VerifierService

class ClinicalVerificationPipeline:
    """
    End-to-end multi-stage pipeline coordinator.
    Exposes clean programmatic entry points for Django applications.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/verifier_demo.pt",
        device: Optional[str] = None,
    ):
        self.stage1 = get_stage1_diarizer()
        self.stage2 = get_stage2_soap_generator()
        self.stage3 = get_stage3_claim_extractor()
        self.stage4 = get_stage4_retriever()
        self.stage5 = get_stage5_concept_service()
        self.stage6 = get_stage6_validity_service()
        self.stage7_8 = Stage78VerifierService(checkpoint_path=checkpoint_path, device=device)

    def run_session(
        self,
        transcript_data: List[Dict[str, Any]],
        patient_records: List[Dict[str, Any]],
        patient_id: str,
        encounter_date: str = "2024-01-15",
        existing_soap_note: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes Stages 2 through 8 on a transcript and patient EHR records.
        """
        encounter_time_iso = f"{encounter_date}T09:00:00Z"

        # 1. Stage 2: SOAP Generation
        if existing_soap_note:
            soap_note = existing_soap_note
        else:
            soap_note = self.stage2.generate(transcript_data)

        # 2. Stage 3A: Atomic Claims
        claims = self.stage3.extract(soap_note, transcript_data)

        # Ingest transcript into Stage 6 store
        enc_id = patient_id.replace("P_", "")
        self.stage6.transcript_store.ingest_dialogue(enc_id, transcript_data, encounter_time=encounter_time_iso)

        verified_claims = []
        for c in claims:
            cid = c.get("claim_id", "CLM-000")
            c_text = c.get("claim_text", "")
            sec = c.get("section", "SOAP")

            # 3. Stage 4: Hybrid Retrieval over EHR + Dialogue
            trans_cands = self.stage6.transcript_store.search(c_text, enc_id, top_k=5)
            all_cands = list(patient_records[:15]) + trans_cands

            # 4. Stage 6: Validity Graph Builder
            graph = self.stage6.build_evidence_graph(
                claim=c,
                candidates=all_cands,
                encounter_time=encounter_time_iso,
            )

            # 5. Stage 7-8: Verifier Forward Pass
            verdict = self.stage7_8.verify_graph(graph, c_text, all_cands)

            verified_claims.append({
                "claim_id": cid,
                "section": sec,
                "claim_text": c_text,
                "label": verdict["label"],
                "calibrated_confidence": round(verdict["confidence"], 4),
                "top_evidence": verdict["top_evidence"],
                "validity_status": graph.nodes[1].status.value if len(graph.nodes) > 1 else "UNKNOWN",
                "checkpoint": verdict["checkpoint"],
            })

        return {
            "patient_id": patient_id,
            "encounter_date": encounter_date,
            "soap_note": soap_note,
            "claims": verified_claims,
        }
