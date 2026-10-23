from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_recall_fscore_support

VERIFICATION_CLASSES = [
    "TRANSCRIPT_SUPPORTED",
    "HISTORY_SUPPORTED",
    "CLINICAL_INFERENCE",
    "CLINICAL_CONFLICT",
    "STALE",
    "HALLUCINATED",
]

def compute_verification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics:
    - Accuracy, Macro-F1
    - Per-class precision, recall, F1
    - Confusion matrix
    - Hallucination precision & recall
    - Stale-as-supported rate (STALE predicted as HISTORY_SUPPORTED or TRANSCRIPT_SUPPORTED)
    """
    target_labels = labels or VERIFICATION_CLASSES
    # Filter to labels present in target_labels
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=target_labels, average="macro", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=target_labels, zero_division=0
    )

    per_class = {}
    for i, lbl in enumerate(target_labels):
        per_class[lbl] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=target_labels).tolist()

    # Hallucination metrics
    halluc_p = per_class.get("HALLUCINATED", {}).get("precision", 0.0)
    halluc_r = per_class.get("HALLUCINATED", {}).get("recall", 0.0)

    # Stale-as-supported rate: out of all gold STALE claims, how many were predicted as SUPPORTED
    stale_total = sum(1 for yt in y_true if yt == "STALE")
    stale_as_supp = sum(
        1 for yt, yp in zip(y_true, y_pred)
        if yt == "STALE" and yp in ["HISTORY_SUPPORTED", "TRANSCRIPT_SUPPORTED"]
    )
    stale_as_supported_rate = (stale_as_supp / stale_total) if stale_total > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
        "labels": target_labels,
        "hallucination_precision": halluc_p,
        "hallucination_recall": halluc_r,
        "stale_as_supported_rate": stale_as_supported_rate,
    }
