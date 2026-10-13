"""
Quickstart Demo Pipeline: Amazon ML Challenge 2026 - Entity Resolution
Runs the complete end-to-end entity resolution workflow on sample data:
  1. Multi-View Text Normalization (Indic transliteration, French diacritics, suffix extraction)
  2. Deterministic Multi-Pass Blocking (Inverted index with country isolation)
  3. 36-Dimensional C++ Pairwise Feature Engineering (RapidFuzz kernels)
  4. LightGBM Scoring & Threshold Calibration (tau* = 0.98)
  5. Official Macro F0.5 Evaluation Harness
  6. Final Competition TSV Export
"""

import os
import sys
import time
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from src.entity_resolution.normalization import normalize_record, NormalizedRecord
from src.entity_resolution.blocking import InvertedIndexBlocker
from src.entity_resolution.features import compute_pair_features, extract_record_tokens
from src.entity_resolution.model import EntityResolutionClassifier
from src.entity_resolution.metrics import evaluate_predictions, compute_macro_f05


def run_demo():
    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE DEMO  ")
    print("=" * 80)
    
    sample_dir = os.path.join(BASE_DIR, "sample_data")
    s1_path = os.path.join(sample_dir, "sample_source1.tsv")
    s2_path = os.path.join(sample_dir, "sample_source2.tsv")
    s3_path = os.path.join(sample_dir, "sample_source3.tsv")
    gt_path = os.path.join(sample_dir, "sample_ground_truth.tsv")
    
    # 1. Load Data
    print("\n[Step 1/6] Loading Sample Datasets...")
    df_s1 = pd.read_csv(s1_path, sep="\t")
    df_s2 = pd.read_csv(s2_path, sep="\t")
    df_s3 = pd.read_csv(s3_path, sep="\t")
    df_gt = pd.read_csv(gt_path, sep="\t", keep_default_na=False)
    
    print(f"  Source 1 queries: {len(df_s1)} records")
    print(f"  Source 2 targets: {len(df_s2)} records")
    print(f"  Source 3 targets: {len(df_s3)} records")
    print(f"  Ground Truth:     {len(df_gt)} entities")
    
    # 2. Multi-View Normalization
    print("\n[Step 2/6] Running Multi-View Normalization Pipeline...")
    t0 = time.time()
    
    s1_norm = [
        normalize_record(row["entity_id"], row["name"], row["address"], row["country"])
        for _, row in df_s1.iterrows()
    ]
    s2_norm = [
        normalize_record(row["entity_id"], row["name"], row["address"], row["country"])
        for _, row in df_s2.iterrows()
    ]
    s3_norm = [
        normalize_record(row["entity_id"], row["name"], row["address"], row["country"])
        for _, row in df_s3.iterrows()
    ]
    target_norm = s2_norm + s3_norm
    print(f"  Normalized {len(s1_norm) + len(target_norm)} total records in {time.time() - t0:.3f}s.")
    
    # 3. Deterministic Inverted Index Blocking
    print("\n[Step 3/6] Indexing Targets & Generating Multi-Pass Candidate Pairs...")
    t0 = time.time()
    blocker = InvertedIndexBlocker()
    
    for rec in target_norm:
        blocker.add_record(
            entity_id=rec.entity_id,
            country=rec.country,
            name_alnum=rec.name_alnum,
            name_no_suffix=rec.name_no_suffix,
            house_number=rec.house_number,
            postal_code=rec.postal_code,
        )
        
    candidates_by_query: Dict[str, Set[str]] = {}
    total_candidates = 0
    for q in s1_norm:
        cands = blocker.retrieve_candidates(
            country=q.country,
            name_alnum=q.name_alnum,
            name_no_suffix=q.name_no_suffix,
            house_number=q.house_number,
            postal_code=q.postal_code,
            max_candidates=100,
        )
        candidates_by_query[q.entity_id] = cands
        total_candidates += len(cands)
        
    print(f"  Generated {total_candidates} candidate pairs across {len(s1_norm)} queries in {time.time() - t0:.3f}s.")
    
    # 4. Feature Extraction & Model Inference
    print("\n[Step 4/6] Computing 36 Pairwise Features & Running Model Scoring...")
    target_dict = {rec.entity_id: rec for rec in target_norm}
    s1_dict = {rec.entity_id: rec for rec in s1_norm}
    
    # Precompute caches
    target_cache = {
        rec.entity_id: extract_record_tokens(rec.name_alnum, rec.name_no_suffix, rec.addr_alnum)
        for rec in target_norm
    }
    s1_cache = {
        rec.entity_id: extract_record_tokens(rec.name_alnum, rec.name_no_suffix, rec.addr_alnum)
        for rec in s1_norm
    }

    # Check for pre-trained models
    model_dir = os.path.join(BASE_DIR, "models")
    predictions: Dict[str, List[str]] = {}
    
    for country in ["US", "India", "France"]:
        model_file = os.path.join(model_dir, f"lgbm_model_{country.lower()}.joblib")
        country_queries = [q for q in s1_norm if q.country.upper() == country.upper()]
        if not country_queries:
            continue
            
        has_pretrained = os.path.exists(model_file)
        if has_pretrained:
            clf = EntityResolutionClassifier.load(model_file)
            print(f"  Loaded pre-trained {country.upper()} LightGBM model: {model_file}")
        else:
            print(f"  [Notice] Pretrained model not found at {model_file}. Using heuristic baseline.")
            clf = None

        for q in country_queries:
            q_cands = candidates_by_query.get(q.entity_id, set())
            if not q_cands:
                predictions[q.entity_id] = []
                continue
                
            pair_features = []
            cand_list = list(q_cands)
            
            for c_id in cand_list:
                t_rec = target_dict[c_id]
                feats = compute_pair_features(
                    s1_id=q.entity_id,
                    s1_name=q.name_alnum,
                    s1_name_no_sfx=q.name_no_suffix,
                    s1_sfx=q.legal_suffix,
                    s1_addr=q.addr_alnum,
                    s1_house=q.house_number,
                    s1_postal=q.postal_code,
                    s1_cache=s1_cache[q.entity_id],
                    cand_id=t_rec.entity_id,
                    cand_name=t_rec.name_alnum,
                    cand_name_no_sfx=t_rec.name_no_suffix,
                    cand_sfx=t_rec.legal_suffix,
                    cand_addr=t_rec.addr_alnum,
                    cand_house=t_rec.house_number,
                    cand_postal=t_rec.postal_code,
                    cand_cache=target_cache[c_id],
                )
                pair_features.append(feats)
                
            X_mat = np.array(pair_features, dtype=np.float32)
            if clf is not None:
                probs = clf.predict_proba(X_mat)
                # Calibrated threshold tau* = 0.98
                matched = [c_id for c_id, p in zip(cand_list, probs) if p >= 0.98]
            else:
                # Heuristic fallback based on composite feature score
                matched = [c_id for c_id, f in zip(cand_list, X_mat) if f[1] >= 80.0]
                
            predictions[q.entity_id] = sorted(matched)
            
    # 5. Evaluate Against Ground Truth
    print("\n[Step 5/6] Evaluating Predictions Against Competition Ground Truth...")
    gt_dict: Dict[str, Set[str]] = {}
    for _, row in df_gt.iterrows():
        eid = str(row["source1_entity_id"])
        m_str = str(row["matched_entity_ids"]).strip()
        gt_dict[eid] = set(m_str.split(",")) if m_str else set()
        
    pred_set_dict = {k: set(v) for k, v in predictions.items()}
    metrics = evaluate_predictions(pred_set_dict, gt_dict)
    
    print(metrics.report())
    
    # 6. Save formatted submission
    print("\n[Step 6/6] Exporting Final TSV Submission...")
    out_dir = os.path.join(BASE_DIR, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "demo_submission.tsv")
    
    submission_rows = []
    for q in s1_norm:
        matched_str = ",".join(predictions.get(q.entity_id, []))
        submission_rows.append({"source1_entity_id": q.entity_id, "matched_entity_ids": matched_str})
        
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(out_file, sep="\t", index=False)
    print(f"  Successfully generated: {out_file} ({len(sub_df)} rows)")
    print("\n[SUCCESS] Pipeline completed successfully!")


if __name__ == "__main__":
    run_demo()
