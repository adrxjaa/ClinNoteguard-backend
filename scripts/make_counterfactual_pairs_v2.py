"""
Counterfactual Pair Generator v2 (Synthea + Gemini Dataset).

Reads verify_data/rendered_ehr.jsonl and constructs counterfactual date-flip pairs
straddling medication STOP events (before stop -> HISTORY_SUPPORTED, after stop -> STALE).
Writes:
  - data/counterfactual_pairs_v2.jsonl
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List


def generate_counterfactual_pairs_v2():
    os.makedirs("data", exist_ok=True)
    out_path = Path("data/counterfactual_pairs_v2.jsonl")

    ehr_path = Path("verify_data/rendered_ehr.jsonl")
    if not ehr_path.exists():
        print("[-] verify_data/rendered_ehr.jsonl not found.")
        return

    with open(ehr_path, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    records_by_pid: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_records:
        pid = r.get("patient_id")
        if pid:
            records_by_pid.setdefault(pid, []).append(r)

    pairs = []
    pair_id = 0

    for pid, records in records_by_pid.items():
        med_stops = [
            r for r in records
            if r.get("section") == "medication" and r.get("event_type") == "STOP" and r.get("valid_to")
        ]
        for med in med_stops:
            try:
                start_dt = datetime.fromisoformat(med["valid_from"].replace("Z", "+00:00"))
                stop_dt = datetime.fromisoformat(med["valid_to"].replace("Z", "+00:00"))
            except Exception:
                continue

            if stop_dt <= start_dt:
                continue

            # t1: midpoint while medication was active; t2: 60 days after discontinuation
            t1_dt = start_dt + (stop_dt - start_dt) / 2
            t2_dt = stop_dt + timedelta(days=60)

            pair_id += 1
            pairs.append({
                "pair_id": f"CF_V2_{pair_id:04d}",
                "patient_id": pid,
                "claim_text": med["claim_text"],
                "evidence_pool_ids": [med["record_id"]],
                "t1_encounter_time": t1_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "t2_encounter_time": t2_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gold_t1": "HISTORY_SUPPORTED",
                "gold_t2": "STALE",
                "event_type_straddled": "STOP",
                "needs_doctor_review": True,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"[+] Generated {len(pairs)} counterfactual date pairs -> {out_path}")


if __name__ == "__main__":
    generate_counterfactual_pairs_v2()
