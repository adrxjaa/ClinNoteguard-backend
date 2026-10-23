import torch
from verify.model.baselines import MajorityClassBaseline, MLPBaselineModel
from verify.eval.metrics import compute_verification_metrics

def test_majority_class_baseline():
    a0 = MajorityClassBaseline()
    a0.fit(["HISTORY_SUPPORTED", "HISTORY_SUPPORTED", "STALE"])
    preds = a0.predict(3)
    assert preds == ["HISTORY_SUPPORTED", "HISTORY_SUPPORTED", "HISTORY_SUPPORTED"]

def test_mlp_baseline_forward():
    model = MLPBaselineModel(in_features=772, hidden_dim=64, num_classes=6)
    dummy_input = torch.randn(4, 772)
    out = model(dummy_input)
    assert out.shape == (4, 6)

def test_compute_verification_metrics():
    y_true = ["HISTORY_SUPPORTED", "STALE", "HALLUCINATED"]
    y_pred = ["HISTORY_SUPPORTED", "HISTORY_SUPPORTED", "HALLUCINATED"]
    metrics = compute_verification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 2.0 / 3.0
    assert metrics["stale_as_supported_rate"] == 1.0 # STALE was predicted as HISTORY_SUPPORTED
