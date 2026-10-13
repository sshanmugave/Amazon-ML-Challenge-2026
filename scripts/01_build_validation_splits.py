"""
Script 01: Build and Seal Leakage-Free Validation Partitions
Amazon ML Challenge 2026 - Business Entity Resolution

Generates stratified 4-way partitions grouped by source1_entity_id:
- Locked-Holdout (10%)
- Dev-Train (63%)
- Dev-Calibration (9%)
- Dev-Validation (18%)
  - Fast-Eval Slice (25,000 entities)

Saves the partition registry to cache/splits/validation_splits.parquet
"""

import os
import sys
import time
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.validation import (
    generate_stratified_splits,
    load_entity_countries,
)

import argparse

DEFAULT_DATA_DIR = os.environ.get(
    "AMAZON_ML_DATA_DIR",
    r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train" 
    if os.path.exists(r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train") 
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train")
)
DEFAULT_OUTPUT_DIR = r"cache\splits"

def main():
    parser = argparse.ArgumentParser(description="Build and Seal Leakage-Free Validation Partitions")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to raw train dataset folder")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to save split parquets")
    parser.add_argument("--fast-eval-size", type=int, default=25000, help="Number of entities in fast-eval slice")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("  STAGE 3.2: GENERATING & SEALING LEAKAGE-FREE VALIDATION PARTITIONS  ")
    print("=" * 78)
    
    gt_path = os.path.join(args.data_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(args.data_dir, "train_source1.tsv")
    
    print(f"Loading Ground Truth from: {gt_path} ...", flush=True)
    gt_df = pd.read_csv(gt_path, sep="\t")
    print(f"  Total S1 entities in Ground Truth: {len(gt_df):,}", flush=True)
    
    print(f"Loading S1 countries from: {s1_path} ...", flush=True)
    countries = load_entity_countries(s1_path)
    print(f"  Total S1 countries loaded: {len(countries):,}", flush=True)
    
    print("\nGenerating stratified partitions (seed=42, fast_eval=25,000) ...", flush=True)
    registry = generate_stratified_splits(
        gt_df=gt_df,
        s1_countries=countries,
        seed=42,
        fast_eval_size=25000,
    )
    
    print("\nValidating mathematical integrity...", flush=True)
    registry.validate_integrity(set(gt_df["source1_entity_id"]))
    print("  [PASSED] All 4 main partitions are strictly pairwise disjoint.")
    print("  [PASSED] Union of partitions covers 100.0% of Source-1 entities.")
    print("  [PASSED] Fast-Eval slice is a strict subset of Dev-Validation with ZERO leakage into Train.")
    
    print(f"\nSaving partitions to {args.output_dir} ...", flush=True)
    registry.save(args.output_dir)
    
    # Stratification Verification Report
    print("\n" + "=" * 78)
    print("                      PARTITION SUMMARY & STRATIFICATION AUDIT                 ")
    print("=" * 78)
    summary = registry.summary_dict()
    for name, count in summary.items():
        pct = (count / summary["total_s1"]) * 100 if name != "total_s1" else 100.0
        print(f"  {name:<18} : {count:>9,} entities ({pct:>6.2f}%)")
        
    print("-" * 78)
    print(f"Execution completed in {time.time() - t0:.2f} seconds.")
    print("=" * 78)

if __name__ == "__main__":
    main()
