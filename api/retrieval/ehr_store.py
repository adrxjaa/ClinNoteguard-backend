"""
EHR store: loads the (synthetic) EHR event tables into patient-scoped documents.

One *document* = one EHR event (diagnosis, medication, investigation, procedure, allergy ...).
Retrieval is ALWAYS scoped to a single patient so a claim can never be "supported" by
another patient's history.

Source of events (priority order, for the real system):
  1. `EHRRecord` rows built from previous encounters + sign-up profile   (future)
  2. CSV files from ClinNoteguard-dataset                                (demo / training)
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEFAULT_DATASET_DIR = os.environ.get(
    "EHR_DATA_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ClinNoteguard-dataset")),
)

SPLIT_FILES = {
    "train": "ehr_train.csv",
    "valid": "ehr_valid.csv",
    "test1": "ehr_test1.csv",
    "test2": "ehr_test2.csv",
    "test3": "ehr_test3.csv",
    "all": "ehr.csv",
}


def pretty_time(rel: str) -> str:
    """'T-3 months' -> '3 months ago' (reads better for dense encoders)."""
    rel = (rel or "").strip()
    if rel.upper().startswith("T-"):
        return rel[2:] + " ago"
    return rel


@dataclass
class EHREvent:
    doc_id: str
    patient_id: str
    encounter_id: str
    relative_time: str
    event_type: str
    description: str
    diagnosis: str = ""
    medication: str = ""
    investigation: str = ""
    source_basis: str = ""

    @property
    def structured_terms(self) -> List[str]:
        return [t.strip() for t in (self.diagnosis, self.medication, self.investigation) if t and t.strip()]

    @property
    def text(self) -> str:
        """Text that is indexed by both BM25 and the dense encoder."""
        parts = [f"{self.event_type}: {self.description}"]
        extras = [t for t in self.structured_terms if t.lower() not in self.description.lower()]
        if extras:
            parts.append("(" + ", ".join(extras) + ")")
        parts.append(f"[{pretty_time(self.relative_time)}]")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "patient_id": self.patient_id,
            "relative_time": self.relative_time,
            "event_type": self.event_type,
            "description": self.description,
            "diagnosis": self.diagnosis,
            "medication": self.medication,
            "investigation": self.investigation,
        }


@dataclass
class EHRStore:
    events_by_patient: Dict[str, List[EHREvent]] = field(default_factory=dict)

    # ---------- loading ----------
    @classmethod
    def from_csv(cls, splits=("all",), dataset_dir: Optional[str] = None) -> "EHRStore":
        dataset_dir = dataset_dir or DEFAULT_DATASET_DIR
        store = cls()
        seen = set()
        for split in splits:
            path = os.path.join(dataset_dir, SPLIT_FILES.get(split, split))
            if not os.path.exists(path):
                raise FileNotFoundError(f"EHR file not found: {path}")
            with open(path, newline="", encoding="utf-8") as f:
                for i, row in enumerate(csv.DictReader(f)):
                    key = (row["patient_id"], row["relative_time"], row["description"])
                    if key in seen:
                        continue
                    seen.add(key)
                    ev = EHREvent(
                        doc_id=f"{row['patient_id']}#{len(store.events_by_patient.get(row['patient_id'], []))}",
                        patient_id=row["patient_id"],
                        encounter_id=row.get("encounter_id", ""),
                        relative_time=row.get("relative_time", ""),
                        event_type=row.get("event_type", ""),
                        description=row.get("description", ""),
                        diagnosis=row.get("diagnosis", "") or "",
                        medication=row.get("medication", "") or "",
                        investigation=row.get("investigation", "") or "",
                        source_basis=row.get("source_basis", ""),
                    )
                    store.add(ev)
        return store

    @classmethod
    def from_claims_json(cls, json_path: str) -> "EHRStore":
        """Loads patient EHR claims from a structured JSON file."""
        import json
        store = cls()
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Support dict of {"patients": [...]} or dict of {patient_id: [...]} or list of claims
        patients_list = data.get("patients", []) if isinstance(data, dict) else []
        if isinstance(data, dict) and not patients_list and "claims" not in data:
            # {pid: [claims]}
            for pid, claims in data.items():
                for c in claims:
                    store._add_claim_dict(pid, c)
            return store
            
        for p in patients_list:
            pid = p.get("patient_id", "")
            for c in p.get("ehr_claims", []):
                store._add_claim_dict(pid, c)
        return store

    def _add_claim_dict(self, patient_id: str, c: dict):
        ev = EHREvent(
            doc_id=c.get("claim_id", f"{patient_id}#{len(self.events_by_patient.get(patient_id, []))}"),
            patient_id=patient_id,
            encounter_id=c.get("encounter_id", "ENC_HIST"),
            relative_time=c.get("relative_time", "T-1 year"),
            event_type=c.get("section", c.get("event_type", "observation")),
            description=c.get("claim_text", c.get("description", "")),
            diagnosis=c.get("diagnosis", ""),
            medication=c.get("medication", ""),
            investigation=c.get("investigation", ""),
            source_basis=c.get("source_basis", "ehr_claim")
        )
        self.add(ev)

    def add(self, ev: EHREvent):
        self.events_by_patient.setdefault(ev.patient_id, []).append(ev)

    # ---------- access ----------
    def patient_ids(self) -> List[str]:
        return sorted(self.events_by_patient)

    def events(self, patient_id: str) -> List[EHREvent]:
        return self.events_by_patient.get(patient_id, [])

    def all_events(self) -> List[EHREvent]:
        return [e for evs in self.events_by_patient.values() for e in evs]

    @staticmethod
    def patient_id_for_encounter(encounter_id: str) -> str:
        return f"P_{encounter_id}"
