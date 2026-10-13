"""
Entity Resolution Model Training & Threshold Calibration Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Implements:
- LightGBM Gradient Boosted Decision Trees with GBDT objective
- Unweighted binary logloss (preserves calibrated probabilities)
- Global and country-specific threshold optimization maximizing exact Entity Macro F0.5
- Serialization of trained models and metadata
"""

import os
import json
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import lightgbm as lgb
import joblib

from src.entity_resolution.metrics import compute_macro_f05, evaluate_predictions
from src.entity_resolution.features import FEATURE_NAMES


class EntityResolutionClassifier:
    """
    LightGBM pairwise classifier for entity resolution match scoring.
    """
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = 8,
        min_child_samples: int = 50,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": -1,
        }
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_threshold: float = 0.50
        self.feature_names = FEATURE_NAMES

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ) -> "EntityResolutionClassifier":
        """Fits the LightGBM classifier with early stopping on validation pairs."""
        self.model = lgb.LGBMClassifier(**self.params)
        
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)] if eval_set else None
        
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted probability of match (class 1)."""
        assert self.model is not None, "Model is not fitted"
        if len(X) == 0:
            return np.zeros(0, dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped by feature name."""
        assert self.model is not None, "Model is not fitted"
        importances = self.model.feature_importances_
        return {
            name: float(imp)
            for name, imp in sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
        }

    def save(self, model_path: str) -> None:
        """Saves model artifact and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load(cls, model_path: str) -> "EntityResolutionClassifier":
        """Loads serialized model artifact."""
        return joblib.load(model_path)


def optimize_macro_f05_threshold(
    pair_metadata: List[Tuple[str, str]],
    probabilities: np.ndarray,
    ground_truth: Dict[str, Set[str]],
    all_s1_entity_ids: List[str],
    threshold_range: Tuple[float, float, float] = (0.20, 0.85, 0.02),
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Performs grid search to find probability threshold maximizing exact Entity Macro F0.5.
    
    Args:
        pair_metadata: List of (s1_id, candidate_target_id)
        probabilities: Array of predicted match probabilities
        ground_truth: Dict mapping s1_id -> set of true matching target IDs
        all_s1_entity_ids: Complete list of all S1 entities in the evaluation slice (including singletons)
        threshold_range: (min_threshold, max_threshold, step)
        
    Returns:
        best_threshold: Optimal threshold float
        best_score: Highest Macro F0.5 achieved
        sweep_records: List of diagnostic dicts for each threshold
    """
    assert len(pair_metadata) == len(probabilities), "Mismatch between pairs and probabilities"
    
    # Group pairs and scores by S1 entity
    s1_to_scored_cands: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in all_s1_entity_ids}
    for (s1_id, cand_id), prob in zip(pair_metadata, probabilities):
        if s1_id in s1_to_scored_cands:
            s1_to_scored_cands[s1_id].append((cand_id, float(prob)))
            
    min_t, max_t, step = threshold_range
    thresholds = np.arange(min_t, max_t + step / 2, step)
    
    best_threshold = 0.50
    best_score = -1.0
    sweep_records = []
    
    for thresh in thresholds:
        # Build predictions at this threshold
        preds: Dict[str, Set[str]] = {}
        for s1_id in all_s1_entity_ids:
            scored = s1_to_scored_cands.get(s1_id, [])
            preds[s1_id] = {cid for cid, prob in scored if prob >= thresh}
            
        metrics = evaluate_predictions(
            ground_truth_dict=ground_truth,
            prediction_dict=preds,
        )
        macro_f05 = metrics.macro_f05
        
        sweep_records.append({
            "threshold": round(float(thresh), 3),
            "macro_f05": macro_f05,
            "micro_precision": metrics.micro_precision,
            "micro_recall": metrics.micro_recall,
            "singleton_fp_rate": metrics.singleton_fp_rate,
        })
        
        if macro_f05 > best_score:
            best_score = macro_f05
            best_threshold = float(thresh)
            
    return best_threshold, best_score, sweep_records
