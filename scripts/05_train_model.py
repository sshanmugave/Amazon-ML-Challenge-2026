"""
Script 05: Pairwise Classifier Training & Threshold Optimization
Amazon ML Challenge 2026 - Business Entity Resolution

Trains the production LightGBM model on Dev-Train and validates on Fast-Eval:
- Leakage-free: Dev-Train pairs for training, Fast-Eval slice for threshold calibration
- Structured Mixed Negative Mining (blocking negatives + high-lexical negatives)
- Computes 36-dimensional feature matrix
- Fits unweighted LightGBM with binary logloss
- Optimizes probability threshold maximizing exact Entity Macro F0.5
- Emits comprehensive diagnostic evaluation report
"""

import os
import sys
import time
import argparse
from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.validation import SplitRegistry, load_ground_truth_dict
from src.entity_resolution.blocking import InvertedIndexBlocker, find_rare_tokens
from src.entity_resolution.training_data import build_pair_training_data
from src.entity_resolution.model import EntityResolutionClassifier, optimize_macro_f05_threshold
from src.entity_resolution.features import (
    compute_pair_features,
    extract_record_tokens,
    FEATURE_NAMES,
)
from src.entity_resolution.metrics import evaluate_predictions

DEFAULT_CACHE_DIR = r"cache\normalized"
DEFAULT_SPLIT_DIR = r"cache\splits"
DEFAULT_GT_PATH = os.environ.get(
    "AMAZON_ML_GT_PATH",
    r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train\train_ground_truth.tsv"
    if os.path.exists(r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train\train_ground_truth.tsv")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train", "train_ground_truth.tsv")
)
DEFAULT_MODEL_DIR = r"models" if os.path.exists(r"models") else r"cache\models"

BLOCKING_COLUMNS = [
    "entity_id",
    "country",
    "name_alnum",
    "name_no_suffix",
    "legal_suffix",
    "addr_alnum",
    "house_number",
    "postal_code",
]


def main():
    parser = argparse.ArgumentParser(description="Entity Resolution Model Training")
    parser.add_argument("--country", choices=["all", "US", "India"], default="US", help="Country partition")
    parser.add_argument("--train-sample", type=int, default=15000, help="Number of Dev-Train S1 entities to train on")
    parser.add_argument("--val-sample", type=int, default=2000, help="Number of Fast-Eval S1 entities to validate on")
    parser.add_argument("--n-estimators", type=int, default=300, help="Number of LightGBM trees")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Normalized parquet cache directory")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Validation splits directory")
    parser.add_argument("--gt-path", default=DEFAULT_GT_PATH, help="Train ground truth TSV path")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Output directory for trained models")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("      STAGE 3.6: ENTITY RESOLUTION MODEL TRAINING & CALIBRATION       ")
    print("=" * 80)
    print(f"Target Country                   : {args.country}")
    print(f"Train S1 Entities (Dev-Train)    : {args.train_sample:,}")
    print(f"Val S1 Entities (Fast-Eval)      : {args.val_sample:,}")
    print("-" * 80, flush=True)

    # 1. Load Partitions and Ground Truth
    print("Loading partition registry and ground truth ...", flush=True)
    registry = SplitRegistry.load(args.split_dir)
    dev_train_ids = registry.dev_train_ids
    fast_eval_ids = registry.fast_eval_ids
    gt_dict = load_ground_truth_dict(args.gt_path)

    # 2. Load Normalized Source 1
    print("Loading normalized Source 1 records ...", flush=True)
    s1_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source1.parquet"), columns=BLOCKING_COLUMNS)
    
    if args.country != "all":
        s1_df = s1_df[s1_df["country"].str.upper() == args.country.upper()]
        
    s1_train = s1_df[s1_df["entity_id"].isin(dev_train_ids)]
    s1_val = s1_df[s1_df["entity_id"].isin(fast_eval_ids)]
    
    if args.train_sample > 0 and len(s1_train) > args.train_sample:
        s1_train = s1_train.sample(n=args.train_sample, random_state=42).reset_index(drop=True)
    if args.val_sample > 0 and len(s1_val) > args.val_sample:
        s1_val = s1_val.sample(n=args.val_sample, random_state=42).reset_index(drop=True)
        
    print(f"Sampled {len(s1_train):,} train S1 entities and {len(s1_val):,} validation S1 entities.", flush=True)

    # 3. Load Targets (S2 + S3)
    print("Loading normalized target records (S2 + S3) ...", flush=True)
    t0 = time.time()
    s2_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source2.parquet"), columns=BLOCKING_COLUMNS)
    s3_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source3.parquet"), columns=BLOCKING_COLUMNS)
    
    if args.country != "all":
        targets_df = pd.concat([
            s2_df[s2_df["country"].str.upper() == args.country.upper()],
            s3_df[s3_df["country"].str.upper() == args.country.upper()],
        ], ignore_index=True)
    else:
        targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    print(f"Loaded {len(targets_df):,} target records in {time.time() - t0:.1f}s.", flush=True)

    # 4. Build Multi-Pass Deterministic Blocker
    print("\nBuilding deterministic inverted index ...", flush=True)
    t_idx = time.time()
    blocker = InvertedIndexBlocker(bucket_cap=200)
    
    t_names_no_sfx = targets_df["name_no_suffix"].fillna("").astype(str).values
    sample_tokens = [name.split() for name in t_names_no_sfx[:500000]]
    rare_tokens = find_rare_tokens(sample_tokens, min_freq=5, max_freq=500)
    print(f"Identified {len(rare_tokens):,} rare tokens in {time.time() - t_idx:.1f}s.", flush=True)
    
    t_eids = targets_df["entity_id"].astype(str).values
    t_names = targets_df["name_alnum"].fillna("").astype(str).values
    t_houses = targets_df["house_number"].fillna("").astype(str).values
    t_postals = targets_df["postal_code"].fillna("").astype(str).values
    
    t_fill = time.time()
    for eid, n, ns, h, p in zip(t_eids, t_names, t_names_no_sfx, t_houses, t_postals):
        blocker.add_record(
            entity_id=eid,
            country=args.country,
            name_alnum=n,
            name_no_suffix=ns,
            house_number=h,
            postal_code=p,
            rare_tokens=rare_tokens,
        )
    print(f"Indexed {len(t_eids):,} targets in {time.time() - t_fill:.1f}s.", flush=True)

    # 5. Mine Candidates for Training Entities
    print("\nMining blocking candidates for Dev-Train sample ...", flush=True)
    t_mine = time.time()
    train_cands: Dict[str, Set[str]] = {}
    
    tr_eids = s1_train["entity_id"].astype(str).values
    tr_names = s1_train["name_alnum"].fillna("").astype(str).values
    tr_no_sfxs = s1_train["name_no_suffix"].fillna("").astype(str).values
    tr_houses = s1_train["house_number"].fillna("").astype(str).values
    tr_postals = s1_train["postal_code"].fillna("").astype(str).values
    
    for eid, n, ns, h, p in zip(tr_eids, tr_names, tr_no_sfxs, tr_houses, tr_postals):
        cands = blocker.retrieve_candidates(
            country=args.country,
            name_alnum=n,
            name_no_suffix=ns,
            house_number=h,
            postal_code=p,
            rare_tokens=rare_tokens,
            max_candidates=100,
        )
        train_cands[eid] = cands
    print(f"Mined candidates for {len(tr_eids):,} training entities in {time.time() - t_mine:.1f}s.", flush=True)

    # 6. Generate Training Features with Hard Negative Mining
    print("\nGenerating training features & hard negative pairs (Strategy C) ...", flush=True)
    t_feat = time.time()
    X_train, y_train, train_pairs = build_pair_training_data(
        s1_df=s1_train,
        targets_df=targets_df,
        candidates_dict=train_cands,
        gt_dict=gt_dict,
        max_blocking_negatives=3,
        rare_tokens=rare_tokens,
    )
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    print(f"Built training set: {len(y_train):,} pairs (Positives: {n_pos:,}, Negatives: {n_neg:,}, Ratio: {n_neg/max(1,n_pos):.1f}:1) in {time.time() - t_feat:.1f}s.", flush=True)

    # 7. Mine Candidates for Validation Entities
    print("\nMining blocking candidates for Fast-Eval validation sample ...", flush=True)
    val_cands: Dict[str, Set[str]] = {}
    val_eids = s1_val["entity_id"].astype(str).values
    val_names = s1_val["name_alnum"].fillna("").astype(str).values
    val_no_sfxs = s1_val["name_no_suffix"].fillna("").astype(str).values
    val_houses = s1_val["house_number"].fillna("").astype(str).values
    val_postals = s1_val["postal_code"].fillna("").astype(str).values
    
    for eid, n, ns, h, p in zip(val_eids, val_names, val_no_sfxs, val_houses, val_postals):
        cands = blocker.retrieve_candidates(
            country=args.country,
            name_alnum=n,
            name_no_suffix=ns,
            house_number=h,
            postal_code=p,
            rare_tokens=rare_tokens,
            max_candidates=100,
        )
        val_cands[eid] = cands

    # Build validation pairs
    print("Generating validation pairs & features ...", flush=True)
    t_val_f = time.time()
    X_val, y_val, val_pairs = build_pair_training_data(
        s1_df=s1_val,
        targets_df=targets_df,
        candidates_dict=val_cands,
        gt_dict=gt_dict,
        max_blocking_negatives=5,
        rare_tokens=rare_tokens,
    )
    print(f"Built validation set: {len(y_val):,} pairs in {time.time() - t_val_f:.1f}s.", flush=True)

    # 8. Train LightGBM Classifier
    print("\nFitting LightGBM Classifier (unweighted binary logloss) ...", flush=True)
    t_fit = time.time()
    clf = EntityResolutionClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=63,
        max_depth=8,
    )
    clf.fit(X_train, y_train, X_val=X_val, y_val=y_val, early_stopping_rounds=25)
    print(f"Model training completed in {time.time() - t_fit:.1f}s.", flush=True)

    # Top Feature Importances
    importances = clf.get_feature_importances()
    print("\nTop 10 Most Important Features:")
    for i, (name, imp) in enumerate(list(importances.items())[:10], 1):
        print(f"  {i:2d}. {name:<28} : {imp:.1f}")

    # 9. Evaluate Predictions on Full Validation Cohort & Sweep Threshold
    print("\nScoring all candidate pairs on Fast-Eval cohort ...", flush=True)
    # Filter targets only to those needed by validation candidates
    needed_tids = set().union(*val_cands.values())
    print(f"  Filtering {len(needed_tids):,} candidate target records for feature caching ...", flush=True)
    val_targets_df = targets_df[targets_df["entity_id"].isin(needed_tids)].copy()
    
    t_dict = dict(zip(val_targets_df["entity_id"].astype(str), val_targets_df.to_dict(orient="records")))
    t_cache = {}
    for tid, rec in t_dict.items():
        t_cache[tid] = extract_record_tokens(
            name_alnum=str(rec.get("name_alnum", "")),
            name_no_suffix=str(rec.get("name_no_suffix", "")),
            addr_alnum=str(rec.get("addr_alnum", "")),
            rare_tokens=rare_tokens,
        )

    all_val_pairs: List[Tuple[str, str]] = []
    all_val_feats: List[np.ndarray] = []
    
    val_sfxs = s1_val["legal_suffix"].fillna("").astype(str).values if "legal_suffix" in s1_val else [""] * len(val_eids)
    val_addrs = s1_val["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in s1_val else [""] * len(val_eids)
    
    for i, s1_id in enumerate(val_eids):
        s1_n = val_names[i]
        s1_ns = val_no_sfxs[i]
        s1_sfx = val_sfxs[i]
        s1_a = val_addrs[i]
        s1_h = val_houses[i]
        s1_p = val_postals[i]
        s1_c = extract_record_tokens(s1_n, s1_ns, s1_a, rare_tokens)
        
        cands = val_cands.get(s1_id, set())
        for cid in cands:
            if cid in t_dict:
                t_rec = t_dict[cid]
                f_vec = compute_pair_features(
                    s1_id=s1_id,
                    s1_name=s1_n,
                    s1_name_no_sfx=s1_ns,
                    s1_sfx=s1_sfx,
                    s1_addr=s1_a,
                    s1_house=s1_h,
                    s1_postal=s1_p,
                    s1_cache=s1_c,
                    cand_id=cid,
                    cand_name=str(t_rec.get("name_alnum", "")),
                    cand_name_no_sfx=str(t_rec.get("name_no_suffix", "")),
                    cand_sfx=str(t_rec.get("legal_suffix", "")),
                    cand_addr=str(t_rec.get("addr_alnum", "")),
                    cand_house=str(t_rec.get("house_number", "")),
                    cand_postal=str(t_rec.get("postal_code", "")),
                    cand_cache=t_cache[cid],
                )
                all_val_pairs.append((s1_id, cid))
                all_val_feats.append(f_vec)

    print(f"Scoring {len(all_val_pairs):,} candidate validation pairs ...", flush=True)
    val_X_matrix = np.vstack(all_val_feats) if all_val_feats else np.zeros((0, 36), dtype=np.float32)
    val_probs = clf.predict_proba(val_X_matrix)

    # 10. Optimize Probability Threshold for Exact Entity Macro F0.5
    print("\nOptimizing probability threshold for exact Macro F0.5 ...", flush=True)
    val_gt = {k: gt_dict.get(k, set()) for k in val_eids}
    best_thresh, best_macro_f05, sweep = optimize_macro_f05_threshold(
        pair_metadata=all_val_pairs,
        probabilities=val_probs,
        ground_truth=val_gt,
        all_s1_entity_ids=list(val_eids),
        threshold_range=(0.70, 0.99, 0.01),
    )

    print("\n" + "=" * 80)
    print("                     THRESHOLD OPTIMIZATION RESULTS                           ")
    print("=" * 80)
    print("  Threshold |  Macro F0.5 | Micro Prec | Micro Rec  | Singleton FP %")
    print("  " + "-" * 66)
    for rec in sweep:
        print(f"    {rec['threshold']:5.2f}   |   {rec['macro_f05']:.6f} |   {rec['micro_precision']:.6f} |   {rec['micro_recall']:.6f} |   {rec['singleton_fp_rate']*100:6.2f}%")
    print("-" * 80)
    print(f"  OPTIMAL THRESHOLD : {best_thresh:.2f}")
    print(f"  PEAK MACRO F0.5   : {best_macro_f05:.6f}")
    print("=" * 80)

    # Save model
    model_save_path = os.path.join(args.model_dir, f"lgbm_model_{args.country.lower()}.joblib")
    clf.best_threshold = best_thresh
    clf.save(model_save_path)
    print(f"\nModel saved successfully to: {model_save_path}")
    print(f"Total script runtime: {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
