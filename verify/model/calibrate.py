from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

class TemperatureScaler(nn.Module):
    """
    Post-hoc temperature scaling calibration on validation logits.
    Calibrates probabilities: P_calibrated = softmax(logits / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        # Ensure temperature stays positive and reasonable
        with torch.no_grad():
            self.temperature.clamp_(min=0.1, max=10.0)

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    probs: [N, num_classes]
    labels: [N] (ground truth indices)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)

def compute_risk_coverage(probs: np.ndarray, labels: np.ndarray, target_error: float = 0.05) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Computes risk-coverage curve and selects abstention threshold tau for target error rate.
    Returns: (tau, coverage_at_target_error, curve_points)
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    sorted_indices = np.argsort(confidences) # ascending
    curve = []
    selected_tau = 0.5
    found_tau = False

    n = len(labels)
    for threshold in np.linspace(0.1, 0.99, 50):
        retained = confidences >= threshold
        if np.sum(retained) == 0:
            continue
        coverage = float(np.mean(retained))
        error_rate = float(1.0 - np.mean(accuracies[retained]))
        curve.append({"threshold": float(threshold), "coverage": coverage, "error_rate": error_rate})

        if not found_tau and error_rate <= target_error:
            selected_tau = float(threshold)
            found_tau = True

    cov_at_target = float(np.mean(confidences >= selected_tau))
    return selected_tau, cov_at_target, curve
