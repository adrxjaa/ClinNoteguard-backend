#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import date as dt_date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "clinnote_backend.settings")

import django

django.setup()

from api.models import PatientProfile, User


def main() -> int:
    synthea_path = BASE_DIR.parent / "ClinNoteguard-dataset" / "synthea_longitudinal_ehr.jsonl"
    if not synthea_path.exists():
        print(f"Error: Synthea dataset not found at {synthea_path}")
        return 1

    by_pat: dict[str, list[dict]] = defaultdict(list)
    with open(synthea_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            by_pat[r["patient_id"]].append(r)
            if i >= 4000:
                break

    def score_patient(encs: list[dict]) -> int:
        total = 0
        for e in encs:
            sl = e.get("structured_slice") or {}
            total += (
                len(sl.get("conditions", []))
                + len(sl.get("medications", [])) * 2
                + len(sl.get("observations", []))
                + len(sl.get("allergies", [])) * 3
                + len(sl.get("procedures", []))
            )
        return total

    best_pid, best_encs = max(by_pat.items(), key=lambda kv: score_patient(kv[1]))
    best_encs.sort(key=lambda e: e.get("timestamp", ""))

    ehr_records: list[dict] = []
    medications: list[dict] = []
    allergies: list[str] = []
    visit_history: list[dict] = []
    seen_meds: set[str] = set()
    seen_allergies: set[str] = set()

    for idx, enc in enumerate(best_encs):
        ts = (enc.get("timestamp") or "2024-01-01")[:10]
        enc_desc = enc.get("description") or enc.get("encounter_class", "Outpatient Visit").title()
        sl = enc.get("structured_slice") or {}

        conds = sl.get("conditions", [])
        meds = sl.get("medications", [])
        obs = sl.get("observations", [])
        procs = sl.get("procedures", [])
        algs = sl.get("allergies", [])

        for c_idx, c in enumerate(conds):
            desc = c.get("description") or str(c)
            ehr_records.append({
                "record_id": f"SYN_COND_{idx}_{c_idx}",
                "patient_id": best_pid,
                "source": "EHR",
                "event_type": "diagnosis",
                "date": ts,
                "text": f"Diagnosis: {desc} (recorded {ts})",
            })

        enc_rx = []
        for m_idx, m in enumerate(meds):
            desc = m.get("description") or str(m)
            ehr_records.append({
                "record_id": f"SYN_MED_{idx}_{m_idx}",
                "patient_id": best_pid,
                "source": "EHR",
                "event_type": "medication",
                "date": ts,
                "text": f"Medication: {desc} (prescribed {ts})",
            })
            if desc not in seen_meds:
                seen_meds.add(desc)
                medications.append({
                    "name": desc,
                    "dose": "As prescribed",
                    "frequency": f"Started {ts}",
                    "clinic": enc_desc,
                })
            enc_rx.append({"name": desc, "dosage": "As directed", "frequency": "Daily"})

        for o_idx, o in enumerate(obs[:8]):
            desc = o.get("description") or str(o)
            val = o.get("value", "")
            units = o.get("units", "")
            val_str = f": {val} {units}".strip() if val != "" else ""
            ehr_records.append({
                "record_id": f"SYN_OBS_{idx}_{o_idx}",
                "patient_id": best_pid,
                "source": "EHR",
                "event_type": "lab",
                "date": ts,
                "text": f"Observation/Lab: {desc}{val_str} ({ts})",
            })

        for p_idx, p in enumerate(procs):
            desc = p.get("description") or str(p)
            ehr_records.append({
                "record_id": f"SYN_PROC_{idx}_{p_idx}",
                "patient_id": best_pid,
                "source": "EHR",
                "event_type": "procedure",
                "date": ts,
                "text": f"Procedure: {desc} ({ts})",
            })

        for a in algs:
            desc = a.get("description") or str(a)
            if desc not in seen_allergies:
                seen_allergies.add(desc)
                allergies.append(desc)
                ehr_records.append({
                    "record_id": f"SYN_ALG_{len(allergies)}",
                    "patient_id": best_pid,
                    "source": "EHR",
                    "event_type": "allergy",
                    "date": ts,
                    "text": f"Allergy: {desc} ({ts})",
                })

        cond_summary = ", ".join((c.get("description") or str(c)) for c in conds[:3]) or enc_desc
        visit_history.append({
            "session_code": f"SYN-{idx + 1:03d}",
            "clinic_name": enc_desc,
            "doctor_name": "Synthea Longitudinal Record",
            "date": ts,
            "assessment": cond_summary,
            "plan": str(enc.get("reason_description") or "Longitudinal EHR follow-up")[:220],
            "prescriptions": enc_rx[:5],
        })

    out_path = BASE_DIR / "data" / "demo_patient_synthea_ehr.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "synthea_patient_id": best_pid,
        "allergies": allergies or ["Sulfa Drug Allergy"],
        "current_medications": medications[:10],
        "visit_history": list(reversed(visit_history))[:10],
        "ehr_records": ehr_records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    user = User.objects.get(email="demo.patient@clinnote.org")
    profile, _ = PatientProfile.objects.get_or_create(user=user)
    profile.allergies = payload["allergies"]
    profile.date_of_birth = profile.date_of_birth or dt_date(1990, 3, 14)
    profile.blood_group = profile.blood_group or "B+"
    profile.height = profile.height or 174.0
    profile.weight = profile.weight or 72.5
    profile.save()

    print(
        f"Seeded demo.patient@clinnote.org from Synthea patient {best_pid}: "
        f"{len(ehr_records)} EHR records, {len(medications)} meds, "
        f"{len(visit_history)} encounters, allergies={payload['allergies']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
