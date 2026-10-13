"""
Validation Architecture & Leak-Free Data Partitioning
Amazon ML Challenge 2026 - Business Entity Resolution

Manages the 4-way zero-leakage data partition grouped by source1_entity_id:
1. Locked-Holdout (10%): Sealed until final pre-submission confirmation.
2. Dev-Train (63% = 70% of Dev): Model training and hard-negative mining.
3. Dev-Calibration (9% = 10% of Dev): Held-out for fitting score calibrators.
4. Dev-Validation (18% = 20% of Dev): Threshold sweep & feature ablations.
   - Fast-Eval Slice (25,000 entities): Stratified subset of Dev-Validation
     for rapid (<3 min) iterative feedback.
"""

from typing import Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
import os
import json
import numpy as np
import pandas as pd


@dataclass
class SplitRegistry:
    """Holds the partitioned entity ID sets."""
    dev_train_ids: Set[str]
    dev_calib_ids: Set[str]
    dev_val_ids: Set[str]
    fast_eval_ids: Set[str]
    locked_ids: Set[str]
    
    def validate_integrity(self, all_expected_ids: Optional[Set[str]] = None) -> None:
        """Verifies mathematical partition integrity: disjointness and coverage."""
        # 1. Mutual disjointness of main partitions
        assert self.dev_train_ids.isdisjoint(self.dev_calib_ids), "Leakage: Dev-Train overlaps with Dev-Calib"
        assert self.dev_train_ids.isdisjoint(self.dev_val_ids), "Leakage: Dev-Train overlaps with Dev-Val"
        assert self.dev_train_ids.isdisjoint(self.locked_ids), "Leakage: Dev-Train overlaps with Locked"
        assert self.dev_calib_ids.isdisjoint(self.dev_val_ids), "Leakage: Dev-Calib overlaps with Dev-Val"
        assert self.dev_calib_ids.isdisjoint(self.locked_ids), "Leakage: Dev-Calib overlaps with Locked"
        assert self.dev_val_ids.isdisjoint(self.locked_ids), "Leakage: Dev-Val overlaps with Locked"
        
        # 2. Fast-Eval must be a strict subset of Dev-Val
        assert self.fast_eval_ids.issubset(self.dev_val_ids), "Leakage: Fast-Eval contains non-Val entities"
        assert self.fast_eval_ids.isdisjoint(self.dev_train_ids), "Leakage: Fast-Eval overlaps with Dev-Train"
        assert self.fast_eval_ids.isdisjoint(self.locked_ids), "Leakage: Fast-Eval overlaps with Locked"
        
        # 3. Complete coverage
        if all_expected_ids is not None:
            union_all = self.dev_train_ids | self.dev_calib_ids | self.dev_val_ids | self.locked_ids
            assert union_all == all_expected_ids, f"Coverage error: {len(all_expected_ids - union_all)} missing entities"
            
    def summary_dict(self) -> Dict[str, int]:
        return {
            "dev_train": len(self.dev_train_ids),
            "dev_calib": len(self.dev_calib_ids),
            "dev_val": len(self.dev_val_ids),
            "fast_eval": len(self.fast_eval_ids),
            "locked_holdout": len(self.locked_ids),
            "total_s1": len(self.dev_train_ids) + len(self.dev_calib_ids) + len(self.dev_val_ids) + len(self.locked_ids),
        }
        
    def save(self, output_dir: str) -> None:
        """Saves partition sets as compact parquet files for fast loading."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Build mapping DataFrame
        data = []
        for s1 in self.dev_train_ids:
            data.append({"source1_entity_id": s1, "split": "dev_train", "is_fast_eval": False})
        for s1 in self.dev_calib_ids:
            data.append({"source1_entity_id": s1, "split": "dev_calib", "is_fast_eval": False})
        for s1 in self.dev_val_ids:
            data.append({"source1_entity_id": s1, "split": "dev_val", "is_fast_eval": (s1 in self.fast_eval_ids)})
        for s1 in self.locked_ids:
            data.append({"source1_entity_id": s1, "split": "locked", "is_fast_eval": False})
            
        df = pd.DataFrame(data)
        out_path = os.path.join(output_dir, "validation_splits.parquet")
        df.to_parquet(out_path, index=False)
        
        # Also write summary metadata JSON
        meta_path = os.path.join(output_dir, "splits_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_dict(), f, indent=2)

    @classmethod
    def load(cls, split_dir: str) -> "SplitRegistry":
        """Loads partitions from parquet file."""
        parquet_path = os.path.join(split_dir, "validation_splits.parquet")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Split file not found at: {parquet_path}")
            
        df = pd.read_parquet(parquet_path)
        train_ids = set(df[df["split"] == "dev_train"]["source1_entity_id"])
        calib_ids = set(df[df["split"] == "dev_calib"]["source1_entity_id"])
        val_ids = set(df[df["split"] == "dev_val"]["source1_entity_id"])
        fast_ids = set(df[(df["split"] == "dev_val") & (df["is_fast_eval"])]["source1_entity_id"])
        locked_ids = set(df[df["split"] == "locked"]["source1_entity_id"])
        
        registry = cls(
            dev_train_ids=train_ids,
            dev_calib_ids=calib_ids,
            dev_val_ids=val_ids,
            fast_eval_ids=fast_ids,
            locked_ids=locked_ids,
        )
        registry.validate_integrity()
        return registry


def generate_stratified_splits(
    gt_df: pd.DataFrame,
    s1_countries: Dict[str, str],
    seed: int = 42,
    fast_eval_size: int = 25000,
) -> SplitRegistry:
    """
    Generates stratified partitions by (country, is_singleton).
    
    Proportions:
    - 10.0% Locked-Holdout
    - 90.0% Dev Pool:
      - 70.0% of Dev (63.0% total) -> Dev-Train
      - 10.0% of Dev (9.0% total)  -> Dev-Calibration
      - 20.0% of Dev (18.0% total) -> Dev-Validation
    """
    rng = np.random.default_rng(seed)
    
    # 1. Compute stratification groups: (country, is_singleton)
    s1_ids = gt_df["source1_entity_id"].values
    matched_strs = gt_df["matched_entity_ids"].fillna("").astype(str).values
    
    strata: Dict[Tuple[str, bool], list] = {}
    for s1, m_str in zip(s1_ids, matched_strs):
        c = s1_countries.get(s1, "UNKNOWN")
        is_singleton = (len(m_str.strip()) == 0)
        key = (c, is_singleton)
        if key not in strata:
            strata[key] = []
        strata[key].append(s1)
        
    train_ids = set()
    calib_ids = set()
    val_ids = set()
    locked_ids = set()
    fast_eval_ids = set()
    
    # Target proportions
    locked_frac = 0.10
    train_frac = 0.63
    calib_frac = 0.09
    # val gets remainder (0.18)
    
    # Stratified split per group
    for (c, is_sing), group in strata.items():
        arr = np.array(group)
        rng.shuffle(arr)
        n = len(arr)
        
        n_locked = int(n * locked_frac)
        n_train = int(n * train_frac)
        n_calib = int(n * calib_frac)
        
        idx_locked_end = n_locked
        idx_train_end = idx_locked_end + n_train
        idx_calib_end = idx_train_end + n_calib
        
        grp_locked = arr[:idx_locked_end]
        grp_train = arr[idx_locked_end:idx_train_end]
        grp_calib = arr[idx_train_end:idx_calib_end]
        grp_val = arr[idx_calib_end:]
        
        locked_ids.update(str(x) for x in grp_locked)
        train_ids.update(str(x) for x in grp_train)
        calib_ids.update(str(x) for x in grp_calib)
        val_ids.update(str(x) for x in grp_val)
        
    # Sample Fast-Eval slice from Dev-Val maintaining exact stratum weights
    val_id_set = val_ids
    val_strata: Dict[Tuple[str, bool], list] = {}
    for s1, m_str in zip(s1_ids, matched_strs):
        s1_str = str(s1)
        if s1_str in val_id_set:
            c = s1_countries.get(s1_str, "UNKNOWN")
            is_singleton = (len(m_str.strip()) == 0)
            key = (c, is_singleton)
            if key not in val_strata:
                val_strata[key] = []
            val_strata[key].append(s1_str)
            
    total_val = len(val_ids)
    for key, group in val_strata.items():
        stratum_weight = len(group) / total_val
        n_sample = max(1, int(fast_eval_size * stratum_weight))
        arr = np.array(group)
        rng.shuffle(arr)
        fast_eval_ids.update(str(x) for x in arr[:n_sample])
        
    registry = SplitRegistry(
        dev_train_ids=train_ids,
        dev_calib_ids=calib_ids,
        dev_val_ids=val_ids,
        fast_eval_ids=fast_eval_ids,
        locked_ids=locked_ids,
    )
    registry.validate_integrity(set(s1_ids))
    return registry


def load_ground_truth_dict(
    gt_path: str,
    filter_entity_ids: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Loads ground truth mapping: {source1_entity_id: set(matched_target_ids)}.
    Singletons are represented as empty sets.
    """
    gt_dict: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        s1_idx = header.index("source1_entity_id")
        match_idx = header.index("matched_entity_ids")
        
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[s1_idx]
            if filter_entity_ids is not None and s1_id not in filter_entity_ids:
                continue
            if len(parts) > match_idx and parts[match_idx].strip():
                targets = {x.strip() for x in parts[match_idx].split(",") if x.strip()}
                gt_dict[s1_id] = targets
            else:
                gt_dict[s1_id] = set()
    return gt_dict


def load_entity_countries(
    s1_path: str,
    filter_entity_ids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Loads mapping {source1_entity_id: country_name}."""
    countries: Dict[str, str] = {}
    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        id_idx = header.index("entity_id")
        c_idx = header.index("country")
        
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[id_idx]
            if filter_entity_ids is not None and s1_id not in filter_entity_ids:
                continue
            countries[s1_id] = parts[c_idx].strip() if len(parts) > c_idx else "UNKNOWN"
    return countries
