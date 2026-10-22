"""
Atomic Claim Extraction Service
Stage 2 -> Stage 3 in ClinNoteGuard Pipeline.

Decomposes structured clinical SOAP notes into fine-grained atomic claims (propositions).
Each claim represents an independently verifiable medical fact, observation, or action plan.
Supports:
1. Clinical LLM generation (Gemini / BioMistral adapter)
2. Rule-based / Clinical NLP proposition decomposition fallback
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from django.conf import settings

logger = logging.getLogger("ClaimExtractor")


class AtomicClaimExtractor:
    """
    Decomposes SOAP note sections (Subjective, Objective, Assessment, Plan)
    into atomic, testable claims with extracted clinical entities.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        self.api_key = getattr(settings, "GEMINI_API_KEY", "")

    def extract_claims(self, soap_note: Dict[str, Any], transcript: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Main entrypoint: takes a SOAP note dict and returns a list of atomic claim dicts.
        """
        if not soap_note or not isinstance(soap_note, dict):
            logger.warning("Empty or invalid SOAP note provided to claim extractor.")
            return []

        # Attempt extraction via LLM if API key is present
        if self.api_key:
            try:
                claims = self._extract_via_llm(soap_note, transcript)
                if claims and len(claims) > 0:
                    logger.info(f"Successfully extracted {len(claims)} atomic claims via LLM.")
                    return claims
            except Exception as e:
                logger.warning(f"LLM extraction failed ({e}). Using NLP proposition splitter fallback.")

        # Fallback to clinical NLP sentence & clause proposition decomposition
        logger.info("Using clinical NLP sentence & clause proposition decomposition fallback.")
        return self._extract_via_nlp_fallback(soap_note)

    def _extract_via_llm(self, soap_note: Dict[str, Any], transcript: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Extract atomic claims using Google GenAI SDK (Gemini) or BioMistral."""
        from google import genai

        formatted_soap = json.dumps(soap_note, indent=2)
        prompt = (
            "You are an expert AI clinical verification engineer.\n"
            "Your task is to take a structured medical SOAP note and extract all individual ATOMIC CLAIMS.\n"
            "An ATOMIC CLAIM is a minimal, self-contained factual assertion of a single clinical observation, "
            "symptom, medical history item, physical measurement, diagnostic assessment, or treatment step.\n\n"
            "CRITICAL EXTRACTION RULES:\n"
            "1. Decompose compound sentences into single atomic claims. "
            "(e.g., 'Patient has headache and nausea for 3 days' -> Claim 1: 'Patient has had a headache for 3 days', "
            "Claim 2: 'Patient has experienced nausea for 3 days').\n"
            "2. Make each claim self-contained: include the clinical subject/context rather than dangling phrases.\n"
            "3. Identify key clinical entities (conditions, symptoms, drugs, anatomical sites, durations, dosages).\n"
            "4. Categorize by SOAP section: 'Subjective', 'Objective', 'Assessment', or 'Plan'.\n"
            "5. Ignore generic filler statements like 'No physical measurements recorded'.\n\n"
            "Respond ONLY with a valid JSON array of objects with the exact keys:\n"
            "[\n"
            "  {\n"
            "    \"claim_id\": \"CLM-001\",\n"
            "    \"section\": \"Subjective\",\n"
            "    \"claim_text\": \"Patient reports a severe headache for the past three days.\",\n"
            "    \"entities\": [\"severe headache\", \"past three days\"],\n"
            "    \"verification_status\": \"UNVERIFIED\"\n"
            "  }\n"
            "]\n\n"
            f"SOAP Note to process:\n{formatted_soap}"
        )

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text_response = response.text.strip()

        # Clean markdown fences
        if text_response.startswith("```"):
            lines = text_response.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            text_response = "\n".join(lines).strip()

        parsed = json.loads(text_response)
        if isinstance(parsed, list):
            # Ensure standard schema
            for idx, item in enumerate(parsed):
                if not item.get("claim_id"):
                    item["claim_id"] = f"CLM-{idx + 1:03d}"
                if not item.get("verification_status"):
                    item["verification_status"] = "UNVERIFIED"
            return parsed

        return []

    def _extract_via_nlp_fallback(self, soap_note: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Rule-based clinical sentence and clause decomposition.
        Splits paragraphs, bullet points, and compound clauses into atomic propositions.
        """
        claims = []
        claim_counter = 1

        for section_name in ["Subjective", "Objective", "Assessment", "Plan"]:
            content = soap_note.get(section_name, "")
            if not content:
                continue

            # Skip empty or default placeholder strings
            if "no physical measurements" in content.lower() or "not recorded" in content.lower():
                continue

            # 1. Normalize lines and bullets
            lines = [line.strip("-*• \t\r") for line in content.split("\n") if line.strip("-*• \t\r")]

            for line in lines:
                # 2. Split into sentences by punctuation (. ? !)
                sentences = re.split(r'(?<=[.!?])\s+', line)

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    # 3. Decompose compound clauses (splitting on ';', 'as well as', 'along with', ', and')
                    clauses = self._split_compound_clause(sentence, section_name)

                    for clause in clauses:
                        clause_clean = clause.strip(". ")
                        if len(clause_clean) < 5:
                            continue

                        # Extract basic entities using regex heuristic (symptoms, drugs, numbers, time)
                        entities = self._extract_simple_entities(clause_clean)

                        claims.append({
                            "claim_id": f"CLM-{claim_counter:03d}",
                            "section": section_name,
                            "claim_text": clause_clean + ".",
                            "entities": entities,
                            "verification_status": "UNVERIFIED"
                        })
                        claim_counter += 1

        return claims

    def _split_compound_clause(self, sentence: str, section: str) -> List[str]:
        """
        Splits compound clinical sentences into atomic propositions while preserving subject context.
        """
        # Split on semicolons first
        subparts = [p.strip() for p in sentence.split(";") if p.strip()]
        atomic_clauses = []

        # Coordinate conjunction split patterns
        split_pattern = r'\s+(?:,\s+and|\bas well as\b|\balong with\b|\band also\b)\s+'

        for part in subparts:
            # Check if this part contains coordinate clauses
            chunks = re.split(split_pattern, part, flags=re.IGNORECASE)
            if len(chunks) > 1:
                # Deduce subject from first chunk (e.g. 'The patient reports...', 'Prescribed...')
                first_chunk = chunks[0].strip()
                atomic_clauses.append(first_chunk)

                # Heuristic to carry forward subject if missing in second chunk
                subject_prefix = ""
                if first_chunk.lower().startswith("the patient") or first_chunk.lower().startswith("patient"):
                    subject_prefix = "Patient has "
                elif first_chunk.lower().startswith("prescribed") or first_chunk.lower().startswith("recommended"):
                    subject_prefix = "Prescribed "

                for following_chunk in chunks[1:]:
                    following_clean = following_chunk.strip()
                    # If clause doesn't have a subject/verb, prepend context
                    if subject_prefix and not any(following_clean.lower().startswith(w) for w in ["patient", "the patient", "prescribed", "recommended", "advised"]):
                        following_clean = f"{subject_prefix}{following_clean}"
                    atomic_clauses.append(following_clean)
            else:
                atomic_clauses.append(part)

        return atomic_clauses

    def _extract_simple_entities(self, text: str) -> List[str]:
        """Simple heuristic extractor for clinical entities (durations, drugs, symptoms)."""
        entities = []

        # Match numbers with units (e.g. 3 days, 500mg, 1 hour)
        durations = re.findall(r'\b\d+\s*(?:days?|hours?|weeks?|months?|mg|ml|bpm|mmhg)\b', text, re.IGNORECASE)
        entities.extend(durations)

        # Common clinical keywords
        clinical_vocab = [
            "headache", "fever", "nausea", "chills", "sinus infection", "sinusitis",
            "nasal discharge", "cough", "chest pain", "shortness of breath", "blurry vision",
            "antibiotics", "antihistamine", "decongestant", "nasal spray", "metformin",
            "blood pressure", "fluid in head", "postural nausea", "forehead", "chin"
        ]

        text_lower = text.lower()
        for term in clinical_vocab:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                if term not in [e.lower() for e in entities]:
                    entities.append(term)

        return entities


# Singleton instance for easy import across Django views and commands
extractor = AtomicClaimExtractor()
