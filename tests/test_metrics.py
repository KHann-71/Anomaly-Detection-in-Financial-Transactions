import numpy as np
import pytest

from src.model import compute_metrics
from src.train import _find_best_threshold

def test_compute_metrics_simple_case():
    y_true = np.array([0,1,1,0])
    y_prob = np.array([0.1,0.6,0.9,0.4])  # threshold 0.5 -> perfect
    m = compute_metrics(y_true, y_prob)
    # with default threshold 0.5 inside compute_metrics we compute using y_prob >= 0.5
    # Expect precision/recall/f1_macro all 1.0, auc also > 0.9
    assert pytest.approx(m["precision"], rel=0, abs=1e-8) == 1.0
    assert pytest.approx(m["recall"], rel=0, abs=1e-8) == 1.0
    assert pytest.approx(m["f1_macro"], rel=0, abs=1e-8) == 1.0
    assert m["auc"] > 0.9

def test_find_best_threshold_scans_and_returns_metrics():
    y_true = np.array([0,0,1,1,1,0,1,0])
    y_prob = np.array([0.1,0.2,0.7,0.8,0.6,0.3,0.9,0.4])
    t, m = _find_best_threshold(y_true, y_prob)
    assert 0.0 < t < 1.0
    for k in ("precision","recall","f1_macro","auc"):
        assert k in m
