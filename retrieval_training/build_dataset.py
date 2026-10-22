"""
Build the retrieval training / evaluation data.

For every ACI-Bench encounter:
  note  --(note_parser)-->  SOAP dict  --(AtomicClaimExtractor, rule-based)-->  atomic claims
  EHR   -->  events of that encounter's patient (P_<encounter_id>)

SILVER relevance labels (no human labels exist yet):
  claim <-> event is positive if
    (a) every content word of a structured term (diagnosis / medication / investigation)
        of the event appears in the claim, OR
    (b) >=60% of the event description's content words (and >=2) appear in the claim.
  Claims with no positive are "no-EHR-evidence" claims (they must be supported by the
  transcript instead) and are kept separately - they are NOT used for training/eval.

Output: retrieval_data/{split}_claims.jsonl
  {split, encounter_id, patient_id, claim_id, section, claim_text, positives:[doc_id], negatives:[doc_id]}
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clinnote_backend.settings")
import django  # noqa: E402

django.setup()

from api.retrieval.ehr_store import DEFAULT_DATASET_DIR, EHRStore  # noqa: E402
from api.retrieval.text_utils import content_terms  # noqa: E402
from api.services.claim_extractor import AtomicClaimExtractor  # noqa: E402
from api.services.note_parser import aci_note_to_soap  # noqa: E402

NOTE_FILES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}
OUT_DIR = os.path.join(ROOT, "retrieval_data")


def silver_positive(claim_text: str, ev) -> bool:
    claim_terms = content_terms(claim_text)
    for term in ev.structured_terms:
        t = content_terms(term)
        if t and t <= claim_terms:
            return True
    desc = content_terms(ev.description)
    if desc:
        shared = desc & claim_terms
        if len(shared) >= 2 and len(shared) / len(desc) >= 0.6:
            return True
    return False


def load_notes(split: str):
    path = os.path.join(DEFAULT_DATASET_DIR, "aci-bench-corpus", "challenge_data", NOTE_FILES[split])
    with open(path, newline="", encoding="utf-8") as f:
        return {r["encounter_id"]: r["note"] for r in csv.DictReader(f)}


def build_split(split: str, extractor: AtomicClaimExtractor):
    store = EHRStore.from_csv([split])
    notes = load_notes(split)
    rows, n_claims, n_no_ev = [], 0, 0
    for enc_id, note in notes.items():
        pid = EHRStore.patient_id_for_encounter(enc_id)
        events = store.events(pid)
        claims = extractor.extract_claims(aci_note_to_soap(note))
        for c in claims:
            n_claims += 1
            pos = [e.doc_id for e in events if silver_positive(c["claim_text"], e)]
            if not pos:
                n_no_ev += 1
                continue
            neg = [e.doc_id for e in events if e.doc_id not in pos]
            rows.append({
                "split": split, "encounter_id": enc_id, "patient_id": pid,
                "claim_id": c["claim_id"], "section": c["section"], "claim_text": c["claim_text"],
                "positives": pos, "negatives": neg,
            })
    return rows, len(notes), n_claims, n_no_ev, len(store.all_events())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    extractor = AtomicClaimExtractor()
    extractor.api_key = ""  # force the deterministic rule-based extractor for dataset building
    print(f"{'split':<7}{'notes':>6}{'events':>8}{'claims':>8}{'w/ EHR evidence':>17}{'no evidence':>13}")
    for split in NOTE_FILES:
        rows, n_notes, n_claims, n_no_ev, n_events = build_split(split, extractor)
        with open(os.path.join(OUT_DIR, f"{split}_claims.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split:<7}{n_notes:>6}{n_events:>8}{n_claims:>8}{len(rows):>17}{n_no_ev:>13}")


if __name__ == "__main__":
    main()
