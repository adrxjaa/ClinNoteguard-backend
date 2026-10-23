from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import yaml
from verify.concepts.attr_type import AttrType
from verify.concepts.facts import TypedFact

class StateStatus(str, Enum):
    CURRENT = "CURRENT"
    SUPERSEDED = "SUPERSEDED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

@dataclass
class StateEntry:
    key: Tuple[str, AttrType] # (key_cui, attr_type)
    status: StateStatus
    current_fact: Optional[TypedFact] = None
    historical_facts: List[TypedFact] = field(default_factory=list)
    superseded_facts: List[TypedFact] = field(default_factory=list)
    effective_value: Optional[str] = None

_DEFAULT_RULES_PATH = Path(__file__).parent / "validity_rules.yaml"

def _load_validity_rules(rules_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def parse_iso_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # Normalize ISO strings
        clean = dt_str.replace("Z", "+00:00")
        return datetime.fromisoformat(clean)
    except Exception:
        # Fallback date parsing (e.g., YYYY-MM-DD)
        try:
            return datetime.strptime(dt_str[:10], "%Y-%m-%d")
        except Exception:
            return None

def resolve_state(
    facts: List[TypedFact],
    encounter_time: str,
    rules_path: Optional[str] = None,
    mode: str = "typed",
) -> Dict[Tuple[str, AttrType], StateEntry]:
    """
    Resolves the temporal validity state of each (key_cui, attr_type) at encounter_time.

    Modes:
      - typed: standard clinician-editable rules per AttrType
      - uniform_supersede: all attribute types treated like medication (latest wins)
      - recency_only: keep only the newest records by timestamp
      - none: all assertions treated as current
    """
    enc_dt = parse_iso_datetime(encounter_time) or datetime(2024, 1, 15, 9, 0, 0)
    cfg = _load_validity_rules(rules_path).get("rules", {})

    # Group facts by key
    grouped: Dict[Tuple[str, AttrType], List[TypedFact]] = {}
    for f in facts:
        grouped.setdefault(f.key, []).append(f)

    results: Dict[Tuple[str, AttrType], StateEntry] = {}

    if mode == "none":
        for k, k_facts in grouped.items():
            latest = k_facts[-1] if k_facts else None
            results[k] = StateEntry(
                key=k,
                status=StateStatus.CURRENT,
                current_fact=latest,
                historical_facts=k_facts,
                effective_value=latest.value if latest else None,
            )
        return results

    if mode == "recency_only":
        # Global recency: only the newest fact across all facts for this key is kept
        for k, k_facts in grouped.items():
            sorted_f = sorted(
                k_facts,
                key=lambda x: parse_iso_datetime(x.timestamp) or datetime.min,
            )
            latest = sorted_f[-1]
            older = sorted_f[:-1]
            results[k] = StateEntry(
                key=k,
                status=StateStatus.CURRENT,
                current_fact=latest,
                historical_facts=sorted_f,
                superseded_facts=older,
                effective_value=latest.value,
            )
        return results

    for k, k_facts in grouped.items():
        key_cui, attr_type = k
        # Filter facts up to encounter_time and sort chronologically
        relevant_facts = []
        for f in k_facts:
            f_dt = parse_iso_datetime(f.timestamp)
            if f_dt is None or f_dt <= enc_dt:
                relevant_facts.append(f)

        if not relevant_facts:
            results[k] = StateEntry(key=k, status=StateStatus.UNKNOWN)
            continue

        sorted_facts = sorted(
            relevant_facts,
            key=lambda x: parse_iso_datetime(x.timestamp) or datetime.min,
        )

        rule = cfg.get(attr_type.value, {})
        persistence = rule.get("persistence", "latest_wins")
        if mode == "uniform_supersede":
            persistence = "latest_wins"

        latest_fact = sorted_facts[-1]
        older_facts = sorted_facts[:-1]
        latest_dt = parse_iso_datetime(latest_fact.timestamp) or enc_dt
        gap_days = (enc_dt - latest_dt).days

        if persistence == "latest_wins":
            # Medication semantics
            if latest_fact.event_type in ["STOP", "DISCONTINUED"] or not latest_fact.polarity:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.SUPERSEDED,
                    current_fact=None,
                    historical_facts=sorted_facts,
                    superseded_facts=sorted_facts,
                    effective_value="discontinued",
                )
            else:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.CURRENT,
                    current_fact=latest_fact,
                    historical_facts=sorted_facts,
                    superseded_facts=older_facts,
                    effective_value=latest_fact.value,
                )

        elif persistence == "indefinite":
            # Chronic Dx or Allergy semantics: persists unless explicitly resolved/negated
            has_resolved = any(
                f.event_type in ["RESOLVE", "RESOLVED"] or not f.polarity
                for f in sorted_facts
            )
            if has_resolved:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.SUPERSEDED,
                    current_fact=None,
                    historical_facts=sorted_facts,
                    superseded_facts=sorted_facts,
                    effective_value="resolved",
                )
            else:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.CURRENT,
                    current_fact=latest_fact,
                    historical_facts=sorted_facts,
                    effective_value=latest_fact.value,
                )

        elif persistence == "horizon":
            # Acute Dx or Lab semantics: expires after horizon
            horizon_days = rule.get("default_horizon_days", 30)
            if gap_days > horizon_days:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.EXPIRED,
                    current_fact=None,
                    historical_facts=sorted_facts,
                    effective_value=latest_fact.value,
                )
            else:
                results[k] = StateEntry(
                    key=k,
                    status=StateStatus.CURRENT,
                    current_fact=latest_fact,
                    historical_facts=sorted_facts,
                    effective_value=latest_fact.value,
                )
        else:
            results[k] = StateEntry(
                key=k,
                status=StateStatus.UNKNOWN,
                current_fact=latest_fact,
                historical_facts=sorted_facts,
                effective_value=latest_fact.value,
            )

    return results
