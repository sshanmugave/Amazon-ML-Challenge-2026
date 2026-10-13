"""
Evaluation Metrics & Competition Harness
Amazon ML Challenge 2026 - Business Entity Resolution

Implements the official precision-heavy Macro F0.5 metric:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Calculated per Source 1 entity, then macro-averaged across ALL Source 1 entities.
Singletons (entities with 0 true matches) are scored as:
    - 1.0 if prediction is empty
    - 0.0 if prediction is non-empty (false merge penalty)
"""

from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> float:
    """
    Computes F0.5 for a single Source 1 entity.
    
    Args:
        true_set: Set of true matching target entity IDs (empty for singletons).
        pred_set: Set of predicted matching target entity IDs (empty for abstention).
        
    Returns:
        float score in [0.0, 1.0].
    """
    # Case 1: Ground-truth singleton (no true matches)
    if not true_set:
        return 1.0 if not pred_set else 0.0
    
    # Case 2: Ground-truth matched entity, but model predicted empty
    if not pred_set:
        return 0.0
    
    # Case 3: Ground-truth matched entity, model predicted candidates
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    
    denom = 0.25 * prec + rec
    if denom == 0.0:
        return 0.0
        
    return (1.25 * prec * rec) / denom


def compute_macro_f05(
    ground_truth_dict: Dict[str, Set[str]],
    prediction_dict: Dict[str, Set[str]],
) -> float:
    """
    Computes exact Macro F0.5 as defined by the Amazon ML Challenge 2026 rules:
    Calculated per Source 1 entity, then averaged across ALL Source 1 entities.
    
    Args:
        ground_truth_dict: Mapping {s1_id: set(true_target_ids)} for ALL evaluation entities.
        prediction_dict: Mapping {s1_id: set(predicted_target_ids)}.
                         Entities missing from prediction_dict are treated as empty sets.
                         
    Returns:
        float: Macro F0.5 across all entities in ground_truth_dict.
    """
    if not ground_truth_dict:
        return 0.0
        
    scores = [
        compute_entity_f05(true_set, prediction_dict.get(s1_id, set()))
        for s1_id, true_set in ground_truth_dict.items()
    ]
    return float(np.mean(scores))


@dataclass
class CountryMetricSummary:
    country: str
    total_entities: int
    matched_entities: int
    singleton_entities: int
    singleton_false_positives: int
    singleton_fp_rate: float
    macro_f05: float
    micro_precision: float
    micro_recall: float
    micro_f05: float


@dataclass
class MetricSummary:
    """Detailed evaluation summary across the evaluation cohort."""
    total_entities: int
    matched_entities: int
    singleton_entities: int
    singleton_false_positives: int
    singleton_fp_rate: float
    
    macro_f05: float
    micro_precision: float
    micro_recall: float
    micro_f05: float
    
    total_true_pairs: int
    total_pred_pairs: int
    total_tp_pairs: int
    
    perfect_score_count: int
    zero_score_count: int
    
    score_p25: float
    score_p50: float
    score_p75: float
    score_p90: float
    score_p99: float
    
    country_metrics: Dict[str, CountryMetricSummary] = field(default_factory=dict)
    
    def report(self) -> str:
        """Generates a formatted diagnostic string report."""
        lines = [
            "=" * 78,
            "                   COMPETITION EVALUATION SUMMARY REPORT                   ",
            "=" * 78,
            f"  Macro F0.5 (Official Leaderboard Metric) : {self.macro_f05:.6f}",
            "-" * 78,
            f"  Total S1 Entities Evaluated              : {self.total_entities:,}",
            f"    - Matched Entities                     : {self.matched_entities:,} ({self.matched_entities / self.total_entities * 100:.2f}%)",
            f"    - Singleton Entities                   : {self.singleton_entities:,} ({self.singleton_entities / self.total_entities * 100:.2f}%)",
            f"    - Singleton False Merges (FP)          : {self.singleton_false_positives:,} ({self.singleton_fp_rate * 100:.2f}%)",
            "-" * 78,
            f"  Pair-Level Micro Metrics:",
            f"    - Total True Pairs                     : {self.total_true_pairs:,}",
            f"    - Total Predicted Pairs                : {self.total_pred_pairs:,}",
            f"    - True Positives (TP)                  : {self.total_tp_pairs:,}",
            f"    - Micro Precision                      : {self.micro_precision:.6f}",
            f"    - Micro Recall                         : {self.micro_recall:.6f}",
            f"    - Micro F0.5                           : {self.micro_f05:.6f}",
            "-" * 78,
            f"  Per-Entity Score Distribution:",
            f"    - Perfect Score (1.0) Entities         : {self.perfect_score_count:,} ({self.perfect_score_count / self.total_entities * 100:.2f}%)",
            f"    - Zero Score (0.0) Entities            : {self.zero_score_count:,} ({self.zero_score_count / self.total_entities * 100:.2f}%)",
            f"    - Percentiles: P25={self.score_p25:.4f} | P50={self.score_p50:.4f} | P75={self.score_p75:.4f} | P90={self.score_p90:.4f} | P99={self.score_p99:.4f}",
        ]
        
        if self.country_metrics:
            lines.extend([
                "-" * 78,
                "  Per-Country Breakdown:",
                "  Country  |  N Ent  |  Macro F05 | Micro Prec | Micro Rec  | Micro F05  | Sing FP %",
                "  " + "-" * 74
            ])
            for c, cm in self.country_metrics.items():
                lines.append(
                    f"  {c:<8} | {cm.total_entities:>7,} |   {cm.macro_f05:.6f} |   {cm.micro_precision:.6f} |   {cm.micro_recall:.6f} |   {cm.micro_f05:.6f} |   {cm.singleton_fp_rate*100:>5.2f}%"
                )
        lines.append("=" * 78)
        return "\n".join(lines)


def evaluate_predictions(
    ground_truth_dict: Dict[str, Set[str]],
    prediction_dict: Dict[str, Set[str]],
    entity_countries: Optional[Dict[str, str]] = None,
) -> MetricSummary:
    """
    Computes a comprehensive diagnostic evaluation across the entire cohort.
    
    Args:
        ground_truth_dict: {s1_id: set(target_ids)} for ALL true entities in evaluation split.
        prediction_dict: {s1_id: set(target_ids)} for predictions.
        entity_countries: Optional {s1_id: country_name} for country slicing.
        
    Returns:
        MetricSummary containing overall and sliced diagnostics.
    """
    total_entities = len(ground_truth_dict)
    if total_entities == 0:
        raise ValueError("ground_truth_dict cannot be empty.")
        
    scores = []
    matched_count = 0
    singleton_count = 0
    singleton_fp_count = 0
    
    total_true = 0
    total_pred = 0
    total_tp = 0
    
    country_groups: Dict[str, Dict[str, Any]] = {}
    if entity_countries:
        for c in set(entity_countries.values()):
            country_groups[c] = {
                "scores": [],
                "matched": 0,
                "singleton": 0,
                "sing_fp": 0,
                "true_pairs": 0,
                "pred_pairs": 0,
                "tp_pairs": 0,
            }
            
    for s1_id, true_set in ground_truth_dict.items():
        pred_set = prediction_dict.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)
        
        country = entity_countries.get(s1_id) if entity_countries else None
        if country and country in country_groups:
            country_groups[country]["scores"].append(score)
            
        is_singleton = (len(true_set) == 0)
        if is_singleton:
            singleton_count += 1
            if country and country in country_groups:
                country_groups[country]["singleton"] += 1
            if pred_set:
                singleton_fp_count += 1
                if country and country in country_groups:
                    country_groups[country]["sing_fp"] += 1
        else:
            matched_count += 1
            if country and country in country_groups:
                country_groups[country]["matched"] += 1
                
        tp = len(true_set & pred_set)
        total_tp += tp
        total_true += len(true_set)
        total_pred += len(pred_set)
        
        if country and country in country_groups:
            country_groups[country]["tp_pairs"] += tp
            country_groups[country]["true_pairs"] += len(true_set)
            country_groups[country]["pred_pairs"] += len(pred_set)

    # Micro metrics
    micro_prec = (total_tp / total_pred) if total_pred > 0 else 0.0
    micro_rec = (total_tp / total_true) if total_true > 0 else 0.0
    denom = 0.25 * micro_prec + micro_rec
    micro_f05 = (1.25 * micro_prec * micro_rec / denom) if denom > 0.0 else 0.0
    
    macro_f05 = float(np.mean(scores))
    sing_fp_rate = (singleton_fp_count / singleton_count) if singleton_count > 0 else 0.0
    
    # Country breakdown
    country_metrics = {}
    if entity_countries:
        for c, cg in country_groups.items():
            c_scores = cg["scores"]
            c_tp = cg["tp_pairs"]
            c_pred = cg["pred_pairs"]
            c_true = cg["true_pairs"]
            c_prec = (c_tp / c_pred) if c_pred > 0 else 0.0
            c_rec = (c_tp / c_true) if c_true > 0 else 0.0
            c_denom = 0.25 * c_prec + c_rec
            c_micro_f05 = (1.25 * c_prec * c_rec / c_denom) if c_denom > 0 else 0.0
            c_sing = cg["singleton"]
            c_sing_fp = cg["sing_fp"]
            c_sing_rate = (c_sing_fp / c_sing) if c_sing > 0 else 0.0
            
            country_metrics[c] = CountryMetricSummary(
                country=c,
                total_entities=len(c_scores),
                matched_entities=cg["matched"],
                singleton_entities=c_sing,
                singleton_false_positives=c_sing_fp,
                singleton_fp_rate=c_sing_rate,
                macro_f05=float(np.mean(c_scores)) if c_scores else 0.0,
                micro_precision=c_prec,
                micro_recall=c_rec,
                micro_f05=c_micro_f05,
            )
            
    scores_arr = np.array(scores)
    return MetricSummary(
        total_entities=total_entities,
        matched_entities=matched_count,
        singleton_entities=singleton_count,
        singleton_false_positives=singleton_fp_count,
        singleton_fp_rate=sing_fp_rate,
        macro_f05=macro_f05,
        micro_precision=micro_prec,
        micro_recall=micro_rec,
        micro_f05=micro_f05,
        total_true_pairs=total_true,
        total_pred_pairs=total_pred,
        total_tp_pairs=total_tp,
        perfect_score_count=int(np.sum(scores_arr == 1.0)),
        zero_score_count=int(np.sum(scores_arr == 0.0)),
        score_p25=float(np.percentile(scores_arr, 25)),
        score_p50=float(np.percentile(scores_arr, 50)),
        score_p75=float(np.percentile(scores_arr, 75)),
        score_p90=float(np.percentile(scores_arr, 90)),
        score_p99=float(np.percentile(scores_arr, 99)),
        country_metrics=country_metrics,
    )
