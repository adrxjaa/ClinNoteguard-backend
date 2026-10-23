"""
Stage 3A: Atomic Claim Extraction & Proposition Decomposition.
Decomposes clinical SOAP notes into fine-grained atomic factual claims with unique IDs.
"""

from typing import List, Dict, Any, Optional
from api.services.claim_extractor import extractor as global_claim_extractor

class Stage3ClaimExtractor:
    """
    Stage 3A Service: Atomic Claim Extractor.
    Extracts atomic claims (CLM-001, CLM-002, ...) with section mapping and clinical entities.
    """

    def __init__(self):
        self.extractor = global_claim_extractor

    def extract(
        self,
        soap_note: Dict[str, Any],
        transcript_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts atomic claims from a SOAP note.
        """
        return self.extractor.extract_claims(soap_note, transcript_data)

def get_stage3_claim_extractor() -> Stage3ClaimExtractor:
    return Stage3ClaimExtractor()
