"""
Helpers to turn an ACI-Bench clinical note (free text with ALL-CAPS headers)
into the SOAP dict format consumed by AtomicClaimExtractor.
"""

import re
from typing import Dict

HEADER_RE = re.compile(r"^([A-Z][A-Z &/,\-]{2,})\s*$", re.M)

SECTION_MAP = {
    "CHIEF COMPLAINT": "Subjective",
    "HISTORY OF PRESENT ILLNESS": "Subjective",
    "REVIEW OF SYSTEMS": "Subjective",
    "SOCIAL HISTORY": "Subjective",
    "MEDICAL HISTORY": "Subjective",
    "PAST HISTORY": "Subjective",
    "FAMILY HISTORY": "Subjective",
    "SURGICAL HISTORY": "Subjective",
    "MEDICATIONS": "Subjective",
    "CURRENT MEDICATIONS": "Subjective",
    "ALLERGIES": "Subjective",
    "PHYSICAL EXAM": "Objective",
    "PHYSICAL EXAMINATION": "Objective",
    "EXAM": "Objective",
    "VITALS": "Objective",
    "VITALS REVIEWED": "Objective",
    "RESULTS": "Objective",
    "ASSESSMENT": "Assessment",
    "IMPRESSION": "Assessment",
    "ASSESSMENT AND PLAN": "Assessment",
    "PLAN": "Plan",
    "INSTRUCTIONS": "Plan",
    "PROCEDURE": "Plan",
}


def aci_note_to_soap(note: str) -> Dict[str, str]:
    soap = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
    matches = list(HEADER_RE.finditer(note))
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        body = note[m.end(): matches[i + 1].start() if i + 1 < len(matches) else len(note)].strip()
        section = SECTION_MAP.get(header)
        if section and body:
            soap[section] += ("\n" if soap[section] else "") + body
    return soap
