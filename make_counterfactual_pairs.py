"""
Counterfactual Pair Generator (make_counterfactual_pairs.py).

Constructs counterfactual clinical evaluation pairs:
- Same claim text, same patient evidence pool, differing strictly by encounter_time.
- t1 (before change/stop) vs t2 (after change/stop).
- Gold labels assigned via temporal ground truth:
    current at t -> HISTORY_SUPPORTED
    superseded/stopped at t -> STALE
    never true -> HALLUCINATED
- Saves to data/counterfactual_pairs.jsonl with needs_doctor_review=true.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime, timedelta

root_dir = str(Path(__file__).parent.resolve())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from verify.eval.counterfactual import compute_temporal_sensitivity_index

def generate_counterfactual_pairs():
    os.makedirs("data", exist_ok=True)
    out_path = Path("data/counterfactual_pairs.jsonl")

    # Load rendered EHR records
    ehr_path = Path("verify_data/rendered_ehr.jsonl")
    if not ehr_path.exists():
        print("[-] verify_data/rendered_ehr.jsonl not found. Run generate_verification_data.py first.")
        return

    with open(ehr_path, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    # Group records by patient_id
    records_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        pid = r.get("patient_id")
        if pid:
            records_by_pid.setdefault(pid, []).append(r)

    pairs = []
    pair_id = 0

    anchor = datetime(2024, 1, 15, 9, 0, 0)

    for pid, records in records_by_pid.items():
        # Find medication timelines with change or stop events
        change_events = [r for r in records if r.get("event_type") in ["CHANGE", "STOP"]]
        start_events = [r for r in records if r.get("event_type") == "START" and "Medication" in r.get("section", "")]

        if not change_events or not start_events:
            continue

        for chg in change_events:
            chg_dt_str = chg.get("valid_from")
            if not chg_dt_str:
                continue

            try:
                chg_dt = datetime.fromisoformat(chg_dt_str.replace("Z", "+00:00"))
            except Exception:
                continue

            # t1: 30 days before the change/stop event
            t1_dt = chg_dt - timedelta(days=30)
            t1_iso = t1_dt.isoformat()

            # t2: 30 days after the change/stop event
            t2_dt = chg_dt + timedelta(days=30)
            t2_iso = t2_dt.isoformat()

            # Find corresponding initial start event
            matching_starts = [
                s for s in start_events
                if any(w in chg["claim_text"].lower() for w in s["claim_text"].lower().split() if len(w) > 5)
            ]
            if not matching_starts:
                continue

            start_ev = matching_starts[0]
            claim_text = start_ev["claim_text"]

            # Ground truth determination:
            # At t1 (before change/stop): the initial start claim was CURRENT -> HISTORY_SUPPORTED
            # At t2 (after change/stop): the initial dose is now SUPERSEDED or STOPPED -> STALE
            gold_t1 = "HISTORY_SUPPORTED"
            gold_t2 = "STALE"

            pair_id += 1
            pairs.append({
                "pair_id": f"CF_PAIR_{pair_id:04d}",
                "patient_id": pid,
                "claim_text": claim_text,
                "evidence_pool_ids": [start_ev["record_id"], chg["record_id"]],
                "t1_encounter_time": t1_iso,
                "t2_encounter_time": t2_iso,
                "gold_t1": gold_t1,
                "gold_t2": gold_t2,
                "pred_t1": None, # Filled during model evaluation
                "pred_t2": None,
                "event_type_straddled": chg.get("event_type"),
                "needs_doctor_review": True,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"[+] Successfully generated {len(pairs)} counterfactual date pairs to {out_path}")
    print("    Marked with needs_doctor_review=true for clinician spot-checking.")

if __name__ == "__main__":
    generate_counterfactual_pairs()
