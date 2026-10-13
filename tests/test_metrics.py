"""
Unit Tests for Competition Evaluation Harness
Amazon ML Challenge 2026
"""

import pytest
import math
import numpy as np

from src.entity_resolution.metrics import (
    compute_entity_f05,
    compute_macro_f05,
    evaluate_predictions,
)


def test_official_readme_worked_example():
    """
    Direct test of the official README worked example:
    - S1-00001 matches true: [S2-00047, S3-00812]
    - Model predicts: [S2-00047, S2-00193, S3-00812]
    - Precision = 2/3, Recall = 1.0
    - Expected F0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = (5/6) / (7/6) = 5/7 ~= 0.7142857
    """
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}
    
    score = compute_entity_f05(true_set, pred_set)
    expected = 5.0 / 7.0
    assert math.isclose(score, expected, rel_tol=1e-6)
    assert f"{score:.3f}" == "0.714"


def test_perfect_prediction():
    true_set = {"S2-101", "S3-202", "S2-303"}
    pred_set = {"S2-101", "S3-202", "S2-303"}
    score = compute_entity_f05(true_set, pred_set)
    assert score == 1.0


def test_zero_overlap_prediction():
    true_set = {"S2-101"}
    pred_set = {"S2-999"}
    score = compute_entity_f05(true_set, pred_set)
    assert score == 0.0


def test_correct_singleton():
    """Singletons correctly predicted with an empty set must score 1.0."""
    true_set = set()
    pred_set = set()
    score = compute_entity_f05(true_set, pred_set)
    assert score == 1.0


def test_false_merge_on_singleton():
    """Singletons predicted with any candidate must score 0.0 (penalty)."""
    true_set = set()
    pred_set = {"S2-00001"}
    score = compute_entity_f05(true_set, pred_set)
    assert score == 0.0


def test_missed_matched_entity():
    """Matched entity predicted with empty set must score 0.0."""
    true_set = {"S2-00001"}
    pred_set = set()
    score = compute_entity_f05(true_set, pred_set)
    assert score == 0.0


def test_precision_recall_asymmetry():
    """
    F0.5 weights precision 2x over recall.
    Test that high precision (P=1.0, R=0.5) scores higher than high recall (P=0.5, R=1.0).
    """
    # High precision: P=1.0, R=0.5 (predicted 1 out of 2 true matches, 0 false positives)
    # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 ~= 0.8333
    score_high_p = compute_entity_f05({"A", "B"}, {"A"})
    expected_high_p = 5.0 / 6.0
    assert math.isclose(score_high_p, expected_high_p, rel_tol=1e-6)
    
    # High recall: P=0.5, R=1.0 (predicted 1 true match + 1 false positive)
    # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 ~= 0.5556
    score_high_r = compute_entity_f05({"A"}, {"A", "B"})
    expected_high_r = 5.0 / 9.0
    assert math.isclose(score_high_r, expected_high_r, rel_tol=1e-6)
    
    assert score_high_p > score_high_r


def test_macro_f05_multi_entity():
    """Verifies macro-averaging across varied entity scenarios."""
    gt = {
        "E1": {"T1"},                     # Perfect -> 1.0
        "E2": {"S2-00047", "S3-00812"},   # Official example -> 5/7
        "E3": set(),                      # Correct singleton -> 1.0
        "E4": set(),                      # False singleton -> 0.0
    }
    preds = {
        "E1": {"T1"},
        "E2": {"S2-00047", "S2-00193", "S3-00812"},
        "E3": set(),
        "E4": {"T_WRONG"},
    }
    
    macro_score = compute_macro_f05(gt, preds)
    expected_macro = (1.0 + (5.0 / 7.0) + 1.0 + 0.0) / 4.0
    assert math.isclose(macro_score, expected_macro, rel_tol=1e-6)


def test_evaluate_predictions_detailed_summary():
    """Verifies MetricSummary computation and diagnostic formatting."""
    gt = {
        "S1-1": {"T1", "T2"},
        "S1-2": set(),
        "S1-3": {"T3"},
    }
    preds = {
        "S1-1": {"T1", "T2"},
        "S1-2": set(),
        "S1-3": {"T_WRONG"},
    }
    countries = {
        "S1-1": "US",
        "S1-2": "US",
        "S1-3": "India",
    }
    
    summary = evaluate_predictions(gt, preds, countries)
    
    assert summary.total_entities == 3
    assert summary.matched_entities == 2
    assert summary.singleton_entities == 1
    assert summary.singleton_false_positives == 0
    assert summary.singleton_fp_rate == 0.0
    
    # S1-1 score=1.0, S1-2 score=1.0, S1-3 score=0.0 -> mean = 2/3
    assert math.isclose(summary.macro_f05, 2.0 / 3.0, rel_tol=1e-6)
    
    # Check US country slice: S1-1 (1.0) and S1-2 (1.0) -> macro = 1.0
    assert "US" in summary.country_metrics
    assert summary.country_metrics["US"].macro_f05 == 1.0
    
    # Check India slice: S1-3 (0.0) -> macro = 0.0
    assert "India" in summary.country_metrics
    assert summary.country_metrics["India"].macro_f05 == 0.0
    
    # Ensure report formats cleanly without exception
    rep = summary.report()
    assert "Official Leaderboard Metric" in rep
    assert "Per-Country Breakdown" in rep


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
