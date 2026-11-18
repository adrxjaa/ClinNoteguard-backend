"""
Longitudinal Synthea + Gemini Verification Dataset Builder.

Consumes:
  ClinNoteguard-dataset/synthea_longitudinal_ehr.jsonl

Produces (in clinnote_backend/verify_data/):
  1. rendered_ehr.jsonl       - Chronological EHR records with Synthea START/STOP ISO timestamps
  2. dialogues.json           - Grounded encounter dialogues keyed by encounter_id
  3. truth_timelines.jsonl    - Ground truth patient timelines per encounter
  4. claims_train.jsonl       - Patient-split training claims (strictly leak-free by patient_id)
  5. claims_val.jsonl         - Patient-split validation claims
  6. claims_test.jsonl        - Patient-split held-out test claims

Also runs automatic leak-free split integrity checks and sequence-length sanity checks.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

root_dir = str(Path(__file__).parent.parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from verify.eval.metrics import VERIFICATION_CLASSES

RANDOM_SEED = 42

DISTRACTOR_CONDITIONS = [
    "systemic lupus erythematosus",
    "acute pancreatitis",
    "pulmonary embolism",
    "rheumatoid arthritis",
    "ulcerative colitis",
    "myasthenia gravis",
    "Addison disease",
    "cirrhosis of liver",
]

DISTRACTOR_MEDS = [
    "warfarin 5mg",
    "methotrexate 15mg",
    "prednisone 20mg",
    "digoxin 0.125mg",
    "lithium carbonate 300mg",
    "amiodarone 200mg",
]

DISTRACTOR_LABS = [
    ("TSH", "8.9 mIU/L"),
    ("serum ferritin", "14 ng/mL"),
    ("INR", "3.1"),
    ("serum troponin I", "0.45 ng/mL"),
]

MED_CLASS_MAP = {
    "lisinopril": "ACE inhibitor",
    "enalapril": "ACE inhibitor",
    "losartan": "ARB antihypertensive",
    "amlodipine": "calcium channel blocker",
    "metoprolol": "beta blocker",
    "atenolol": "beta blocker",
    "atorvastatin": "statin",
    "simvastatin": "statin",
    "rosuvastatin": "statin",
    "metformin": "biguanide",
    "glipizide": "sulfonylurea",
    "insulin": "insulin therapy",
    "furosemide": "loop diuretic",
    "hydrochlorothiazide": "thiazide diuretic",
    "omeprazole": "proton pump inhibitor",
    "albuterol": "bronchodilator",
    "fluticasone": "inhaled corticosteroid",
    "sertraline": "SSRI antidepressant",
    "ibuprofen": "NSAID",
    "naproxen": "NSAID",
    "acetaminophen": "analgesic",
    "amoxicillin": "penicillin antibiotic",
    "azithromycin": "macrolide antibiotic",
}


def _parse_dt(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None)
    except ValueError:
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _iso_utc(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _infer_med_class(desc: str) -> Optional[str]:
    low = desc.lower()
    for drug, cls_name in MED_CLASS_MAP.items():
        if drug in low:
            return cls_name
    return None


def _split_dialogue_sentences(dialogue: str) -> List[str]:
    lines = [ln.strip() for ln in dialogue.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        txt = re.sub(r"^\[(doctor|patient|speaker|clinician)\]\s*", "", ln, flags=re.IGNORECASE).strip()
        if len(txt) >= 15:
            cleaned.append(txt)
    return cleaned


def _extract_transcript_claim(enc: Dict[str, Any], rng: random.Random) -> Optional[str]:
    """Builds a claim grounded in the current encounter's dialogue/slice."""
    slice_data = enc.get("structured_slice", {})
    diag_lower = enc.get("dialogue", "").lower()

    # Prefer a current encounter fact that is explicitly mentioned in the dialogue
    for m in slice_data.get("medications", []):
        desc = m.get("description", "")
        first_word = desc.split()[0].lower() if desc else ""
        if first_word and first_word in diag_lower:
            return f"Patient is currently prescribed {desc}."
    for c in slice_data.get("conditions", []):
        desc = c.get("description", "")
        if desc and desc.lower()[:12] in diag_lower:
            return f"Patient presents with {desc}."
    for o in slice_data.get("observations", []):
        desc = o.get("description", "")
        val = o.get("value", "")
        units = o.get("units", "")
        if desc and val and val.lower() in diag_lower:
            return f"Current encounter {desc} is {val} {units}.".strip()

    # Fallback: use a clean sentence from the dialogue
    sents = _split_dialogue_sentences(enc.get("dialogue", ""))
    if sents:
        s = rng.choice(sents)
        if len(s) > 160:
            s = s[:157].rsplit(" ", 1)[0] + "."
        return s
    return None


def build_verification_dataset(
    input_jsonl: Path,
    out_dir: Path,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    by_patient: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_patient[rec["patient_id"]].append(rec)

    for pid in by_patient:
        by_patient[pid].sort(key=lambda r: (r.get("timestamp", ""), r.get("encounter_index", 0)))

    all_pids = sorted(by_patient.keys())
    rng.shuffle(all_pids)

    n_total = len(all_pids)
    n_train = max(1, int(round(n_total * 0.65)))
    n_val = max(1, int(round(n_total * 0.15)))
    train_pids = set(all_pids[:n_train])
    val_pids = set(all_pids[n_train : n_train + n_val])
    test_pids = set(all_pids[n_train + n_val :])

    # Strict leak-free patient-level split assertions
    assert not (train_pids & val_pids), "DATA LEAK: train and val share patients!"
    assert not (train_pids & test_pids), "DATA LEAK: train and test share patients!"
    assert not (val_pids & test_pids), "DATA LEAK: val and test share patients!"

    rendered_ehr: List[Dict[str, Any]] = []
    dialogues_export: Dict[str, str] = {}
    truth_timelines: List[Dict[str, Any]] = []
    claims_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    claim_counter = 0

    def add_claim(split: str, pid: str, eid: str, enc_time: str, enc_class: str, text: str, label: str, ev_ids: List[str]):
        nonlocal claim_counter
        claim_counter += 1
        claims_by_split[split].append({
            "claim_id": f"SYN_CLM_{claim_counter:05d}",
            "patient_id": pid,
            "encounter_id": eid,
            "encounter_time": enc_time,
            "encounter_class": enc_class,
            "claim_text": text,
            "label": label,
            "gold_evidence_ids": ev_ids,
        })

    for pid in all_pids:
        split = "train" if pid in train_pids else ("val" if pid in val_pids else "test")
        encs = by_patient[pid]

        # Track longitudinal EHR records built from encounters 0..t-1
        patient_ehr_history: List[Dict[str, Any]] = []
        patient_all_terms: Set[str] = set()

        for enc_idx, enc in enumerate(encs):
            eid = enc["encounter_id"]
            enc_time_raw = enc["timestamp"]
            enc_dt = _parse_dt(enc_time_raw) or datetime(2024, 1, 15, 9, 0, 0)
            enc_time = _iso_utc(enc_dt)
            enc_class = enc.get("encounter_class", "ambulatory")
            slice_data = enc.get("structured_slice", {})

            dialogues_export[eid] = enc.get("dialogue", "")

            # Collect all clinical terms for hallucination filtering
            for c in slice_data.get("conditions", []):
                patient_all_terms.add(c.get("description", "").lower())
            for m in slice_data.get("medications", []):
                patient_all_terms.add(m.get("description", "").lower())
            for o in slice_data.get("observations", []):
                patient_all_terms.add(o.get("description", "").lower())

            # Truth timeline snapshot
            truth_timelines.append({
                "patient_id": pid,
                "encounter_id": eid,
                "encounter_time": enc_time,
                "encounter_class": enc_class,
                "ongoing_conditions": enc.get("prior_summary", {}).get("ongoing_conditions", []),
                "active_medications": enc.get("prior_summary", {}).get("active_medications", []),
                "stopped_medications": enc.get("prior_summary", {}).get("stopped_medications", []),
            })

            # Generate claims for encounters with prior history (enc_idx >= 1) or all encounters
            # 1. TRANSCRIPT_SUPPORTED
            t_claim = _extract_transcript_claim(enc, rng)
            if t_claim:
                add_claim(split, pid, eid, enc_time, enc_class, t_claim, "TRANSCRIPT_SUPPORTED", [f"TRANS_{eid}_U000"])

            # 2. HISTORY_SUPPORTED (from active/recent prior EHR records)
            recent_valid_hist = [
                r for r in patient_ehr_history
                if r["valid_to"] is None and (_parse_dt(r["valid_from"]) or enc_dt) >= (enc_dt - timedelta(days=365))
            ]
            if recent_valid_hist:
                chosen = rng.choice(recent_valid_hist)
                add_claim(split, pid, eid, enc_time, enc_class, chosen["claim_text"], "HISTORY_SUPPORTED", [chosen["record_id"]])

            # 3. CLINICAL_INFERENCE (e.g., drug -> therapeutic class)
            med_cands = slice_data.get("medications", [])
            inferred = False
            for m in med_cands:
                cls_name = _infer_med_class(m.get("description", ""))
                if cls_name:
                    add_claim(
                        split, pid, eid, enc_time, enc_class,
                        f"Patient is receiving {cls_name} therapy.",
                        "CLINICAL_INFERENCE",
                        [f"TRANS_{eid}_U000"],
                    )
                    inferred = True
                    break
            if not inferred and slice_data.get("conditions"):
                cond_desc = slice_data["conditions"][0]["description"]
                add_claim(
                    split, pid, eid, enc_time, enc_class,
                    f"Clinical presentation is consistent with {cond_desc} management.",
                    "CLINICAL_INFERENCE",
                    [f"TRANS_{eid}_U000"],
                )

            # 4. CLINICAL_CONFLICT (contradicting a stopped med or documented observation)
            stopped_hist = [r for r in patient_ehr_history if r["event_type"] == "STOP"]
            if stopped_hist:
                st = rng.choice(stopped_hist)
                med_name = st.get("entity_name", "prior medication")
                add_claim(
                    split, pid, eid, enc_time, enc_class,
                    f"Patient is currently taking {med_name} without interruption.",
                    "CLINICAL_CONFLICT",
                    [st["record_id"]],
                )
            elif slice_data.get("observations"):
                obs = slice_data["observations"][0]
                obs_desc = obs.get("description", "blood pressure")
                add_claim(
                    split, pid, eid, enc_time, enc_class,
                    f"Patient's {obs_desc} is critically elevated at 999.",
                    "CLINICAL_CONFLICT",
                    [f"TRANS_{eid}_U000"],
                )

            # 5. STALE (record > 365 days old or resolved condition asserted as current)
            stale_hist = [
                r for r in patient_ehr_history
                if (_parse_dt(r["valid_from"]) or enc_dt) < (enc_dt - timedelta(days=365))
            ]
            if stale_hist:
                st_rec = rng.choice(stale_hist)
                add_claim(
                    split, pid, eid, enc_time, enc_class,
                    st_rec["claim_text"],
                    "STALE",
                    [st_rec["record_id"]],
                )
            elif patient_ehr_history:
                # Synthesize a stale >2-year-old lab anchor in EHR for this patient
                old_dt = _iso_utc(enc_dt - timedelta(days=rng.randint(450, 900)))
                stale_id = f"EHR_SYN_STALE_{pid[:8]}_{enc_idx}"
                stale_txt = "Patient's Hemoglobin A1c level is 6.4%."
                rendered_ehr.append({
                    "patient_id": pid,
                    "encounter_id": eid,
                    "record_id": stale_id,
                    "claim_text": stale_txt,
                    "valid_from": old_dt,
                    "valid_to": None,
                    "event_type": "START",
                    "section": "observation",
                    "source": "EHR",
                    "synthetic_v2": False,
                    "dataset": "synthea_gemini",
                })
                add_claim(split, pid, eid, enc_time, enc_class, stale_txt, "STALE", [stale_id])

            # 6. HALLUCINATED (claim absent from both dialogue and patient EHR)
            cand_hallucinations = [
                f"Patient has a documented history of {d}." for d in DISTRACTOR_CONDITIONS
                if d.lower() not in patient_all_terms
            ] + [
                f"Patient takes {m} oral daily." for m in DISTRACTOR_MEDS
                if m.split()[0].lower() not in patient_all_terms
            ]
            if cand_hallucinations:
                h_txt = rng.choice(cand_hallucinations)
                add_claim(split, pid, eid, enc_time, enc_class, h_txt, "HALLUCINATED", [])

            # Append current encounter's structured facts into rendered_ehr for future encounters t+1..T
            for c_i, c in enumerate(slice_data.get("conditions", [])):
                desc = c.get("description", "").strip()
                if not desc:
                    continue
                c_start = _parse_dt(c.get("start")) or enc_dt
                c_stop = _parse_dt(c.get("stop"))
                rec = {
                    "patient_id": pid,
                    "encounter_id": eid,
                    "record_id": f"EHR_SYN_COND_{eid[:8]}_{c_i}",
                    "entity_name": desc,
                    "claim_text": f"Patient has a documented history of {desc}.",
                    "valid_from": _iso_utc(c_start),
                    "valid_to": _iso_utc(c_stop) if c_stop else None,
                    "event_type": "STOP" if c_stop else "START",
                    "section": "condition",
                    "source": "EHR",
                    "synthetic_v2": False,
                    "dataset": "synthea_gemini",
                }
                rendered_ehr.append(rec)
                patient_ehr_history.append(rec)

            for m_i, m in enumerate(slice_data.get("medications", [])):
                desc = m.get("description", "").strip()
                if not desc:
                    continue
                m_start = _parse_dt(m.get("start")) or enc_dt
                m_stop = _parse_dt(m.get("stop"))
                rec = {
                    "patient_id": pid,
                    "encounter_id": eid,
                    "record_id": f"EHR_SYN_MED_{eid[:8]}_{m_i}",
                    "entity_name": desc,
                    "claim_text": f"Patient takes {desc}.",
                    "valid_from": _iso_utc(m_start),
                    "valid_to": _iso_utc(m_stop) if m_stop else None,
                    "event_type": "STOP" if m_stop else "START",
                    "section": "medication",
                    "source": "EHR",
                    "synthetic_v2": False,
                    "dataset": "synthea_gemini",
                }
                rendered_ehr.append(rec)
                patient_ehr_history.append(rec)

            for o_i, o in enumerate(slice_data.get("observations", [])[:6]):
                desc = o.get("description", "").strip()
                val = o.get("value", "").strip()
                units = o.get("units", "").strip()
                if not desc or not val:
                    continue
                o_dt = _parse_dt(o.get("date")) or enc_dt
                rec = {
                    "patient_id": pid,
                    "encounter_id": eid,
                    "record_id": f"EHR_SYN_OBS_{eid[:8]}_{o_i}",
                    "entity_name": desc,
                    "claim_text": f"Patient's {desc} level is {val} {units}.".strip(),
                    "valid_from": _iso_utc(o_dt),
                    "valid_to": None,
                    "event_type": "START",
                    "section": "observation",
                    "source": "EHR",
                    "synthetic_v2": False,
                    "dataset": "synthea_gemini",
                }
                rendered_ehr.append(rec)
                patient_ehr_history.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as fout:
            for r in rows:
                fout.write(json.dumps(r) + "\n")

    write_jsonl(out_dir / "rendered_ehr.jsonl", rendered_ehr)
    write_jsonl(out_dir / "truth_timelines.jsonl", truth_timelines)
    for sp in ("train", "val", "test"):
        write_jsonl(out_dir / f"claims_{sp}.jsonl", claims_by_split[sp])
    with open(out_dir / "dialogues.json", "w", encoding="utf-8") as fout:
        json.dump(dialogues_export, fout, indent=2)

    # Sanity check summary
    dialogue_lengths = [len(v.split()) for v in dialogues_export.values()]
    note_lengths = [len(r.get("clinical_note", "").split()) for r in records]
    stats = {
        "patients": {"train": len(train_pids), "val": len(val_pids), "test": len(test_pids), "total": n_total},
        "leak_check_patient_overlap": 0,
        "rendered_ehr_rows": len(rendered_ehr),
        "dialogues_count": len(dialogues_export),
        "claims_per_split": {sp: len(claims_by_split[sp]) for sp in ("train", "val", "test")},
        "class_distribution_total": dict(Counter(
            c["label"] for sp in ("train", "val", "test") for c in claims_by_split[sp]
        )),
        "dialogue_word_len_avg": round(sum(dialogue_lengths) / max(1, len(dialogue_lengths)), 1),
        "clinical_note_word_len_avg": round(sum(note_lengths) / max(1, len(note_lengths)), 1),
    }
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build leak-free Synthea+Gemini verification dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/home/statixlo/Develop/Final_year_project/ClinNoteguard-dataset/synthea_longitudinal_ehr.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent.parent / "verify_data",
    )
    args = parser.parse_args()
    build_verification_dataset(args.input, args.out_dir)
