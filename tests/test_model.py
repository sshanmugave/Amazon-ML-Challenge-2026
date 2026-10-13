"""
Unit Tests for Model Classifier & Threshold Optimization
Amazon ML Challenge 2026 - Business Entity Resolution
"""

import os
import pytest
import numpy as np
from src.entity_resolution.model import (
    EntityResolutionClassifier,
    optimize_macro_f05_threshold,
)
from src.entity_resolution.features import NUM_FEATURES


def test_model_training_and_serialization(tmp_path):
    rng = np.random.RandomState(42)
    N = 200
    X_train = rng.randn(N, NUM_FEATURES).astype(np.float32)
    # Target is correlated with feature 0 (exact match) and feature 1 (ratio)
    y_train = ((X_train[:, 0] + X_train[:, 1]) > 0).astype(np.int8)
    
    clf = EntityResolutionClassifier(n_estimators=20, num_leaves=15)
    clf.fit(X_train, y_train)
    
    probs = clf.predict_proba(X_train[:10])
    assert len(probs) == 10
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    
    # Save and load
    model_path = os.path.join(tmp_path, "model.joblib")
    clf.save(model_path)
    assert os.path.exists(model_path)
    
    loaded = EntityResolutionClassifier.load(model_path)
    loaded_probs = loaded.predict_proba(X_train[:10])
    np.testing.assert_allclose(probs, loaded_probs, rtol=1e-5)


def test_optimize_macro_f05_threshold():
    # Setup 3 entities:
    # E1: true={T1}, candidates=[(T1, 0.9), (T2, 0.3)]
    # E2: true={T3}, candidates=[(T3, 0.8), (T4, 0.7)]
    # E3: singleton (true={}), candidates=[(T5, 0.6)]
    pair_metadata = [
        ("E1", "T1"),
        ("E1", "T2"),
        ("E2", "T3"),
        ("E2", "T4"),
        ("E3", "T5"),
    ]
    probabilities = np.array([0.9, 0.3, 0.8, 0.7, 0.6])
    ground_truth = {
        "E1": {"T1"},
        "E2": {"T3"},
        "E3": set(),
    }
    all_s1_ids = ["E1", "E2", "E3"]
    
    best_thresh, best_f05, sweep = optimize_macro_f05_threshold(
        pair_metadata=pair_metadata,
        probabilities=probabilities,
        ground_truth=ground_truth,
        all_s1_entity_ids=all_s1_ids,
        threshold_range=(0.4, 0.8, 0.1),
    )
    
    # At threshold 0.75:
    # E1: {T1} -> perfect (F0.5=1.0)
    # E2: {T3} -> perfect (F0.5=1.0)
    # E3: {} -> singleton empty -> perfect (F0.5=1.0)
    # Macro F0.5 should be 1.0 at thresh >= 0.71 and <= 0.8!
    assert best_f05 == 1.0
    assert best_thresh >= 0.7
