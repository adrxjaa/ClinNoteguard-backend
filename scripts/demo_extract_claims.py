"""
Demo and Test Script: Extract Atomic Claims from SQLite Consultation Session
Stage 2 -> Stage 3 (SOAP Note -> Atomic Propositions)
"""

import os
import sys
import json
import django

# Setup Django Environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'clinnote_backend.settings')
django.setup()

from api.models import ConsultationSession
from api.services.claim_extractor import extractor as claim_extractor


def run_demo(session_code: str = "036722"):
    print("=" * 70)
    print("CLINICAL PIPELINE: ATOMIC CLAIM EXTRACTION (STAGE 2 -> STAGE 3)")
    print("=" * 70)

    # 1. Fetch Session from SQLite
    try:
        session = ConsultationSession.objects.get(code=session_code)
    except ConsultationSession.DoesNotExist:
        # Fallback to any ended session that has a SOAP note
        session = ConsultationSession.objects.filter(status='ENDED').exclude(soap_note_json={}).first()
        if not session:
            print("No suitable consultation session found in SQLite.")
            return

    print(f"\n[+] Loaded Consultation Session: {session.code}")
    print(f"    - Patient: {session.patient.get_full_name() or session.patient.email}")
    print(f"    - Status: {session.status}")
    print(f"    - Transcript Utterances: {len(session.transcript_json)}")

    # 2. Display SOAP Note from SQLite
    soap_note = session.soap_note_json
    print("\n--- Source SOAP Note from SQLite ---")
    for sec in ["Subjective", "Objective", "Assessment", "Plan"]:
        content = soap_note.get(sec, "N/A")
        print(f"\n[{sec.upper()}]")
        print(f"  {content}")

    # 3. Extract Atomic Claims
    print("\n" + "=" * 70)
    print("[+] Extracting Atomic Claims (Sentence & Clause Decomposition)...")
    print("=" * 70)

    claims = claim_extractor.extract_claims(soap_note, session.transcript_json)
    print(f"\n[+] Successfully Extracted {len(claims)} Atomic Propositions:")

    # 3.5 Retrieve Candidate Evidence (Stage 3 Hybrid Retrieval)
    print("\n" + "=" * 70)
    print("[+] Running Hybrid Candidate Retrieval against EHR History...")
    print("=" * 70)
    from api.retrieval.pipeline import retrieve_evidence_for_claims
    claims = retrieve_evidence_for_claims(
        claims,
        patient_id=str(session.patient.id) if session.patient else None,
        session_code=session.code,
        top_k=2
    )

    # 4. Display Extracted Claims Formatted
    print("\n" + "-" * 70)
    print(f"{'ID':<10} | {'SECTION':<12} | {'EHR STATUS':<14} | CLAIM ASSERTION")
    print("-" * 70)
    for c in claims:
        ehr_stat = c.get("ehr_status", "N/A")
        print(f"{c['claim_id']:<10} | {c['section']:<12} | {ehr_stat:<14} | {c['claim_text'][:60]}")
        evidence = c.get("retrieved_evidence", [])
        for ev in evidence:
            dose_tag = f"[{ev.get('dose_agreement')}]" if ev.get('dose_agreement') != 'n/a' else ""
            print(f"   ↳ [RRF: {ev.get('rrf_score', 0):.4f} {dose_tag}] {ev.get('text', '')[:70]}")

    # 5. Persist back to SQLite
    session.atomic_claims_json = claims
    session.save()
    print(f"\n[+] Saved {len(claims)} atomic claims to SQLite session ({session.code}).")

    # 6. Save JSON Output Artifact
    output_filename = "demo_atomic_claims.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({
            "session_code": session.code,
            "patient": session.patient.email,
            "claims_count": len(claims),
            "soap_note": soap_note,
            "atomic_claims": claims
        }, f, indent=2)

    print(f"[+] Exported structured JSON artifact to: {output_filename}")
    print("=" * 70)


if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "036722"
    run_demo(code)
