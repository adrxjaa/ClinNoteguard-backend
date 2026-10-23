import os
import django

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clinnote_backend.settings")
    try:
        django.setup()
    except Exception:
        pass

from .stage1_audio import get_stage1_diarizer, Stage1AudioDiarizer
from .stage2_soap import get_stage2_soap_generator, Stage2SOAPGenerator
from .stage3_claims import get_stage3_claim_extractor, Stage3ClaimExtractor
from .stage4_retrieval import get_stage4_retriever, Stage4ClinicalRetriever
from .stage5_concepts import get_stage5_concept_service, Stage5ConceptService
from .stage6_validity import get_stage6_validity_service, Stage6ValidityGraphService
from .stage7_8_verifier import get_stage7_8_verifier, Stage78VerifierService
from .pipeline import ClinicalVerificationPipeline

__all__ = [
    "get_stage1_diarizer",
    "Stage1AudioDiarizer",
    "get_stage2_soap_generator",
    "Stage2SOAPGenerator",
    "get_stage3_claim_extractor",
    "Stage3ClaimExtractor",
    "get_stage4_retriever",
    "Stage4ClinicalRetriever",
    "get_stage5_concept_service",
    "Stage5ConceptService",
    "get_stage6_validity_service",
    "Stage6ValidityGraphService",
    "get_stage7_8_verifier",
    "Stage78VerifierService",
    "ClinicalVerificationPipeline",
]
