from typing import List, Dict, Any, Tuple
import numpy as np

def compute_temporal_sensitivity_index(
    pairs: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Computes Temporal Sensitivity Index (TSI):
      TSI = P(predicted label changes | gold label changes)
    Also computes stale-as-supported rate across the counterfactual pairs.
    """
    gold_changed = 0
    pred_changed_given_gold_changed = 0

    stale_total = 0
    stale_as_supported = 0

    for pair in pairs:
        gold_t1 = pair.get("gold_t1")
        gold_t2 = pair.get("gold_t2")
        pred_t1 = pair.get("pred_t1")
        pred_t2 = pair.get("pred_t2")

        if gold_t1 != gold_t2:
            gold_changed += 1
            if pred_t1 != pred_t2:
                pred_changed_given_gold_changed += 1

        for g, p in [(gold_t1, pred_t1), (gold_t2, pred_t2)]:
            if g == "STALE":
                stale_total += 1
                if p in ["HISTORY_SUPPORTED", "TRANSCRIPT_SUPPORTED"]:
                    stale_as_supported += 1

    tsi = (pred_changed_given_gold_changed / gold_changed) if gold_changed > 0 else 0.0
    stale_as_supp_rate = (stale_as_supported / stale_total) if stale_total > 0 else 0.0

    return {
        "tsi": float(tsi),
        "gold_changed_count": int(gold_changed),
        "pred_changed_count": int(pred_changed_given_gold_changed),
        "stale_as_supported_rate": float(stale_as_supp_rate),
    }
