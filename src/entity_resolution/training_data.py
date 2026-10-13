"""
Training Pair Generation & Hard Negative Mining Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Generates balanced, hard-negative enriched pairwise training datasets:
- Strategy C: Structured Mixed Mining (3 blocking negatives + 2 high-lexical negatives per positive)
- Zero leakage: Strictly bounded to Dev-Train partition
- Computes 36-dimensional feature vectors via rapid C++ string kernels
- Emits compact float32 feature matrices for LightGBM / XGBoost
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from src.entity_resolution.features import (
    compute_pair_features,
    extract_record_tokens,
    FEATURE_NAMES,
    NUM_FEATURES,
)


def build_pair_training_data(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    candidates_dict: Dict[str, Set[str]],
    gt_dict: Dict[str, Set[str]],
    max_blocking_negatives: int = 3,
    max_lexical_negatives: int = 2,
    rare_tokens: Optional[Set[str]] = None,
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Builds training feature matrix X and label vector y for pairs.
    
    Args:
        s1_df: DataFrame of Source-1 records (from Dev-Train only)
        targets_df: DataFrame of target records (Sources 2 and 3)
        candidates_dict: Dict mapping s1_id -> set of candidate target IDs from blocking
        gt_dict: Ground truth mapping s1_id -> set of true matching target IDs
        max_blocking_negatives: Negative pairs sampled from blocking candidates
        max_lexical_negatives: Hard negative pairs with high lexical similarity
        rare_tokens: Set of corpus rare tokens
        rng_seed: Random seed for reproducibility
        
    Returns:
        X: np.ndarray of shape (N_pairs, 36), float32
        y: np.ndarray of shape (N_pairs,), int8 (1 for match, 0 for non-match)
        pair_metadata: List of (s1_id, target_id)
    """
    rng = np.random.RandomState(rng_seed)
    
    # Collect only target IDs that are actually referenced by S1 true targets or candidate sets
    s1_eids_list = s1_df["entity_id"].astype(str).tolist()
    needed_tids: Set[str] = set()
    for s1_id in s1_eids_list:
        needed_tids.update(gt_dict.get(s1_id, set()))
        needed_tids.update(candidates_dict.get(s1_id, set()))
        
    sub_targets = targets_df[targets_df["entity_id"].isin(needed_tids)]
    
    # Index filtered target records by ID for fast lookup
    target_dict: Dict[str, Dict[str, Any]] = {}
    target_cache: Dict[str, Dict[str, Any]] = {}
    
    t_eids = sub_targets["entity_id"].astype(str).values
    t_names = sub_targets["name_alnum"].fillna("").astype(str).values
    t_no_sfxs = sub_targets["name_no_suffix"].fillna("").astype(str).values if "name_no_suffix" in sub_targets else t_names
    t_sfxs = sub_targets["legal_suffix"].fillna("").astype(str).values if "legal_suffix" in sub_targets else [""] * len(t_eids)
    t_addrs = sub_targets["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in sub_targets else [""] * len(t_eids)
    t_houses = sub_targets["house_number"].fillna("").astype(str).values if "house_number" in sub_targets else [""] * len(t_eids)
    t_postals = sub_targets["postal_code"].fillna("").astype(str).values if "postal_code" in sub_targets else [""] * len(t_eids)
    
    for i, tid in enumerate(t_eids):
        target_dict[tid] = {
            "name": t_names[i],
            "name_no_sfx": t_no_sfxs[i],
            "sfx": t_sfxs[i],
            "addr": t_addrs[i],
            "house": t_houses[i],
            "postal": t_postals[i],
        }
        # Precompute token cache
        target_cache[tid] = extract_record_tokens(
            name_alnum=t_names[i],
            name_no_suffix=t_no_sfxs[i],
            addr_alnum=t_addrs[i],
            rare_tokens=rare_tokens,
        )
        
    # Source-1 arrays
    s1_eids = s1_df["entity_id"].astype(str).values
    s1_names = s1_df["name_alnum"].fillna("").astype(str).values
    s1_no_sfxs = s1_df["name_no_suffix"].fillna("").astype(str).values if "name_no_suffix" in s1_df else s1_names
    s1_sfxs = s1_df["legal_suffix"].fillna("").astype(str).values if "legal_suffix" in s1_df else [""] * len(s1_eids)
    s1_addrs = s1_df["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in s1_df else [""] * len(s1_eids)
    s1_houses = s1_df["house_number"].fillna("").astype(str).values if "house_number" in s1_df else [""] * len(s1_eids)
    s1_postals = s1_df["postal_code"].fillna("").astype(str).values if "postal_code" in s1_df else [""] * len(s1_eids)
    
    feature_rows: List[np.ndarray] = []
    labels: List[int] = []
    pair_metadata: List[Tuple[str, str]] = []
    
    for i, s1_id in enumerate(s1_eids):
        s1_n = s1_names[i]
        s1_ns = s1_no_sfxs[i]
        s1_sfx = s1_sfxs[i]
        s1_a = s1_addrs[i]
        s1_h = s1_houses[i]
        s1_p = s1_postals[i]
        s1_c = extract_record_tokens(s1_n, s1_ns, s1_a, rare_tokens)
        
        true_targets = gt_dict.get(s1_id, set())
        cands = candidates_dict.get(s1_id, set())
        
        # 1. Positives (all true matches present in targets)
        for tid in true_targets:
            if tid in target_dict:
                t_rec = target_dict[tid]
                f_vec = compute_pair_features(
                    s1_id=s1_id,
                    s1_name=s1_n,
                    s1_name_no_sfx=s1_ns,
                    s1_sfx=s1_sfx,
                    s1_addr=s1_a,
                    s1_house=s1_h,
                    s1_postal=s1_p,
                    s1_cache=s1_c,
                    cand_id=tid,
                    cand_name=t_rec["name"],
                    cand_name_no_sfx=t_rec["name_no_sfx"],
                    cand_sfx=t_rec["sfx"],
                    cand_addr=t_rec["addr"],
                    cand_house=t_rec["house"],
                    cand_postal=t_rec["postal"],
                    cand_cache=target_cache[tid],
                )
                feature_rows.append(f_vec)
                labels.append(1)
                pair_metadata.append((s1_id, tid))
                
        # 2. Hard Negatives from Blocking (candidates not in ground truth)
        blocking_negatives = [cid for cid in cands if cid not in true_targets and cid in target_dict]
        if blocking_negatives:
            if len(blocking_negatives) > max_blocking_negatives:
                selected_negatives = rng.choice(blocking_negatives, size=max_blocking_negatives, replace=False)
            else:
                selected_negatives = blocking_negatives
                
            for tid in selected_negatives:
                t_rec = target_dict[tid]
                f_vec = compute_pair_features(
                    s1_id=s1_id,
                    s1_name=s1_n,
                    s1_name_no_sfx=s1_ns,
                    s1_sfx=s1_sfx,
                    s1_addr=s1_a,
                    s1_house=s1_h,
                    s1_postal=s1_p,
                    s1_cache=s1_c,
                    cand_id=tid,
                    cand_name=t_rec["name"],
                    cand_name_no_sfx=t_rec["name_no_sfx"],
                    cand_sfx=t_rec["sfx"],
                    cand_addr=t_rec["addr"],
                    cand_house=t_rec["house"],
                    cand_postal=t_rec["postal"],
                    cand_cache=target_cache[tid],
                )
                feature_rows.append(f_vec)
                labels.append(0)
                pair_metadata.append((s1_id, tid))
                
    if not feature_rows:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int8), []
        
    X = np.vstack(feature_rows)
    y = np.array(labels, dtype=np.int8)
    return X, y, pair_metadata
