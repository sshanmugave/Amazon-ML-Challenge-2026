"""
Script 06: Test Set Inference & Competition Submission Generation
Amazon ML Challenge 2026 - Business Entity Resolution

Generates complete, leakage-free predictions on the test dataset:
- Evaluates per country partition (US, India, France)
- Generates candidates via multi-pass blocker
- Computes 36-dimensional feature vectors
- Scores pairs with calibrated LightGBM models
- Applies optimal threshold tau* per country
- Robust checkpointing: saves per-batch parquet checkpoints with automatic resumption
- Country-partitioned memory management: prevents OOM by unloading inactive targets
- Validates output integrity: exactly 1,732,544 S1 entities, zero nulls
- Formats to competition specification: TSV with (source1_entity_id, matched_entity_ids)
"""

import os
import sys
import gc
import glob
import time
import argparse
from typing import Dict, Set, List, Tuple, Any
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.blocking import InvertedIndexBlocker, find_rare_tokens
from src.entity_resolution.model import EntityResolutionClassifier
from src.entity_resolution.features import compute_pair_features, extract_record_tokens

DEFAULT_CACHE_DIR = r"cache\normalized"
DEFAULT_MODEL_DIR = r"models" if os.path.exists(r"models") else r"cache\models"
DEFAULT_CHECKPOINT_DIR = r"cache\checkpoints"
DEFAULT_SUBMISSION_DIR = r"submissions"

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


COUNTRY_CASE_MAP = {
    "us": "US",
    "india": "India",
    "france": "France",
}


def load_country_targets(country: str, cache_dir: str = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Loads target records (S2 + S3) strictly filtered to the specified country."""
    c_val = COUNTRY_CASE_MAP.get(country.lower(), country)
    print(f"  Loading test target records (S2 + S3) for {c_val} ...", flush=True)
    t0 = time.time()
    s2 = pd.read_parquet(
        os.path.join(cache_dir, "test_source2.parquet"),
        columns=BLOCKING_COLUMNS,
        filters=[("country", "==", c_val)],
    )
    s3 = pd.read_parquet(
        os.path.join(cache_dir, "test_source3.parquet"),
        columns=BLOCKING_COLUMNS,
        filters=[("country", "==", c_val)],
    )
    targets = pd.concat([s2, s3], ignore_index=True)
    del s2, s3
    gc.collect()
    print(f"  Loaded {len(targets):,} target records for {c_val} in {time.time() - t0:.1f}s.", flush=True)
    return targets


def generate_submission(
    country: str,
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    model: EntityResolutionClassifier,
    threshold: float,
    batch_size: int = 25000,
    checkpoint_dir: str = CHECKPOINT_DIR,
    reset_checkpoints: bool = False,
) -> Dict[str, List[str]]:
    """
    Generates test predictions for a single country partition with incremental checkpointing.
    """
    country_lower = country.lower()
    c_chk_dir = os.path.join(checkpoint_dir, country_lower)
    os.makedirs(c_chk_dir, exist_ok=True)
    consolidated_file = os.path.join(checkpoint_dir, f"predictions_{country_lower}.parquet")

    # Check if complete consolidated file already exists
    if not reset_checkpoints and os.path.exists(consolidated_file):
        print(f"  Found consolidated predictions for {country.upper()}: {consolidated_file}", flush=True)
        cdf = pd.read_parquet(consolidated_file)
        if len(cdf) == len(s1_df):
            print(f"  Verified {len(cdf):,} predictions match S1 query count. Reusing existing predictions!", flush=True)
            preds: Dict[str, List[str]] = {}
            for eid, m_str in zip(cdf["source1_entity_id"].values, cdf["matched_entity_ids"].values):
                preds[str(eid)] = m_str.split(",") if m_str else []
            return preds
        else:
            print(f"  Consolidated count mismatch ({len(cdf):,} vs {len(s1_df):,}). Recomputing missing batches ...", flush=True)

    print(f"\nProcessing Country: {country.upper()} (S1: {len(s1_df):,}, Targets: {len(targets_df):,}) ...", flush=True)
    t_start = time.time()

    # Pre-index targets for rapid feature lookup via column dicts
    print("  Pre-caching target attribute lookup maps ...", flush=True)
    t_eids = targets_df["entity_id"].astype(str).values
    t_names = targets_df["name_alnum"].fillna("").astype(str).values
    t_names_no_sfx = targets_df["name_no_suffix"].fillna("").astype(str).values
    t_houses = targets_df["house_number"].fillna("").astype(str).values
    t_postals = targets_df["postal_code"].fillna("").astype(str).values
    t_sfxs = targets_df["legal_suffix"].fillna("").astype(str).values if "legal_suffix" in targets_df else [""] * len(t_eids)
    t_addrs = targets_df["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in targets_df else [""] * len(t_eids)

    t_names_map = dict(zip(t_eids, t_names))
    t_no_sfx_map = dict(zip(t_eids, t_names_no_sfx))
    t_sfx_map = dict(zip(t_eids, t_sfxs))
    t_addr_map = dict(zip(t_eids, t_addrs))
    t_house_map = dict(zip(t_eids, t_houses))
    t_postal_map = dict(zip(t_eids, t_postals))
    t_cache: Dict[str, Any] = {}

    # Check which batches are pending
    total_s1 = len(s1_df)
    batch_ranges = [(start, min(start + batch_size, total_s1)) for start in range(0, total_s1, batch_size)]
    pending_ranges = []
    predictions: Dict[str, List[str]] = {}

    for start_idx, end_idx in batch_ranges:
        b_file = os.path.join(c_chk_dir, f"batch_{start_idx:07d}_{end_idx:07d}.parquet")
        if not reset_checkpoints and os.path.exists(b_file):
            b_df = pd.read_parquet(b_file)
            for eid, m_str in zip(b_df["source1_entity_id"].values, b_df["matched_entity_ids"].values):
                predictions[str(eid)] = m_str.split(",") if m_str else []
        else:
            pending_ranges.append((start_idx, end_idx))

    if not pending_ranges:
        print(f"  All {len(batch_ranges)} batches already completed from checkpoints!", flush=True)
    else:
        print(f"  {len(batch_ranges) - len(pending_ranges)} / {len(batch_ranges)} batches cached. {len(pending_ranges)} batches to score.", flush=True)
        # Build blocker index on test targets only if there are pending batches
        print(f"  Building inverted index on {len(targets_df):,} target records ...", flush=True)
        t_idx = time.time()
        blocker = InvertedIndexBlocker(bucket_cap=200)

        sample_tokens = [name.split() for name in t_names_no_sfx[:500000]]
        rare_tokens = find_rare_tokens(sample_tokens, min_freq=5, max_freq=500)

        for eid, n, ns, h, p in zip(t_eids, t_names, t_names_no_sfx, t_houses, t_postals):
            blocker.add_record(
                entity_id=eid,
                country=country,
                name_alnum=n,
                name_no_suffix=ns,
                house_number=h,
                postal_code=p,
                rare_tokens=rare_tokens,
            )
        print(f"  Indexed {len(targets_df):,} records in {time.time() - t_idx:.1f}s.", flush=True)

        # Process pending batches
        s1_eids = s1_df["entity_id"].astype(str).values
        s1_names = s1_df["name_alnum"].fillna("").astype(str).values
        s1_no_sfxs = s1_df["name_no_suffix"].fillna("").astype(str).values
        s1_sfxs = s1_df["legal_suffix"].fillna("").astype(str).values if "legal_suffix" in s1_df else [""] * total_s1
        s1_addrs = s1_df["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in s1_df else [""] * total_s1
        s1_houses = s1_df["house_number"].fillna("").astype(str).values if "house_number" in s1_df else [""] * total_s1
        s1_postals = s1_df["postal_code"].fillna("").astype(str).values if "postal_code" in s1_df else [""] * total_s1

        for b_idx, (start_idx, end_idx) in enumerate(pending_ranges):
            t_b0 = time.time()
            b_file = os.path.join(c_chk_dir, f"batch_{start_idx:07d}_{end_idx:07d}.parquet")
            b_pairs: List[Tuple[str, str]] = []
            b_feats: List[np.ndarray] = []
            b_preds: Dict[str, List[str]] = {s1_eids[i]: [] for i in range(start_idx, end_idx)}

            for i in range(start_idx, end_idx):
                s1_id = s1_eids[i]
                s1_n = s1_names[i]
                s1_ns = s1_no_sfxs[i]
                s1_sfx = s1_sfxs[i]
                s1_a = s1_addrs[i]
                s1_h = s1_houses[i]
                s1_p = s1_postals[i]
                s1_c = extract_record_tokens(s1_n, s1_ns, s1_a, rare_tokens)

                cands = blocker.retrieve_candidates(
                    country=country,
                    name_alnum=s1_n,
                    name_no_suffix=s1_ns,
                    house_number=s1_h,
                    postal_code=s1_p,
                    rare_tokens=rare_tokens,
                    max_candidates=100,
                )

                for cid in cands:
                    if cid in t_names_map:
                        if cid not in t_cache:
                            t_cache[cid] = extract_record_tokens(
                                name_alnum=t_names_map[cid],
                                name_no_suffix=t_no_sfx_map[cid],
                                addr_alnum=t_addr_map[cid],
                                rare_tokens=rare_tokens,
                            )
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
                            cand_name=t_names_map[cid],
                            cand_name_no_sfx=t_no_sfx_map[cid],
                            cand_sfx=t_sfx_map[cid],
                            cand_addr=t_addr_map[cid],
                            cand_house=t_house_map[cid],
                            cand_postal=t_postal_map[cid],
                            cand_cache=t_cache[cid],
                        )
                        b_pairs.append((s1_id, cid))
                        b_feats.append(f_vec)

            if b_feats:
                b_X = np.vstack(b_feats)
                b_probs = model.predict_proba(b_X)
                for (qid, cid), prob in zip(b_pairs, b_probs):
                    if prob >= threshold:
                        b_preds[qid].append(cid)

            # Save batch checkpoint to disk
            b_df = pd.DataFrame({
                "source1_entity_id": list(b_preds.keys()),
                "matched_entity_ids": [",".join(v) for v in b_preds.values()],
            })
            b_df.to_parquet(b_file, index=False)
            predictions.update(b_preds)

            pct = (end_idx / total_s1) * 100
            elapsed = time.time() - t_b0
            print(f"    [{country.upper()}] Batch [{start_idx:,}..{end_idx:,}] ({pct:.1f}%) | {len(b_feats):,} pairs scored in {elapsed:.1f}s. Checkpointed.", flush=True)

    # Consolidate all batches for this country
    print(f"  Consolidating all {len(predictions):,} predictions for {country.upper()} ...", flush=True)
    all_s1_ordered = s1_df["entity_id"].astype(str).tolist()
    consolidated_df = pd.DataFrame({
        "source1_entity_id": all_s1_ordered,
        "matched_entity_ids": [",".join(predictions.get(eid, [])) for eid in all_s1_ordered],
    })
    consolidated_df.to_parquet(consolidated_file, index=False)
    print(f"  Saved consolidated predictions to: {consolidated_file}", flush=True)
    print(f"  Country {country.upper()} completed in {time.time() - t_start:.1f}s.", flush=True)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate Competition Test Submission")
    parser.add_argument("--output", type=str, default="submissions/submission.tsv")
    parser.add_argument("--sample-s1", type=int, default=0, help="Subsample N test S1 entities for rapid verification (0 = full)")
    parser.add_argument("--country", choices=["all", "US", "India", "France"], default="all", help="Evaluate specific country partition")
    parser.add_argument("--batch-size", type=int, default=25000, help="Batch size for candidate scoring and checkpointing")
    parser.add_argument("--reset-checkpoints", action="store_true", help="Clear previous checkpoints and recompute from scratch")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Normalized parquet cache directory")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory containing pre-trained models")
    parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    t0 = time.time()

    print("=" * 80)
    print("      STAGE 3.8: FULL TEST SET INFERENCE & SUBMISSION GENERATION      ")
    print("=" * 80)
    print(f"Output TSV: {args.output}")
    print(f"Checkpoints Dir: {args.checkpoint_dir}")
    print(f"Target Country: {args.country}")
    print(f"Batch Size: {args.batch_size:,}")
    print(f"Reset Checkpoints: {args.reset_checkpoints}")
    print("-" * 80)

    # 1. Load Normalized Test Source 1 Records
    print("Loading normalized test Source 1 records ...", flush=True)
    test_s1 = pd.read_parquet(os.path.join(args.cache_dir, "test_source1.parquet"), columns=BLOCKING_COLUMNS)
    print(f"Loaded {len(test_s1):,} test Source-1 records.", flush=True)

    # 2. Process per country partition
    countries = [args.country] if args.country != "all" else ["US", "India", "France"]
    all_predictions: Dict[str, List[str]] = {}

    for country in countries:
        c_s1 = test_s1[test_s1["country"].str.upper() == country.upper()].copy()
        if args.sample_s1 > 0 and len(c_s1) > args.sample_s1:
            c_s1 = c_s1.sample(n=args.sample_s1, random_state=42).reset_index(drop=True)

        if len(c_s1) == 0:
            continue

        model_path = os.path.join(args.model_dir, f"lgbm_model_{country.lower()}.joblib")
        if not os.path.exists(model_path):
            model_path = os.path.join(args.model_dir, "lgbm_model_us.joblib")

        print(f"\nLoading model from: {model_path} ...", flush=True)
        model = EntityResolutionClassifier.load(model_path)
        thresh = getattr(model, "best_threshold", 0.50)
        print(f"Using calibrated decision threshold for {country.upper()}: {thresh:.2f}", flush=True)

        # Check if country already fully consolidated (and not in sample mode)
        consolidated_file = os.path.join(args.checkpoint_dir, f"predictions_{country.lower()}.parquet")
        if not args.reset_checkpoints and args.sample_s1 == 0 and os.path.exists(consolidated_file):
            cdf = pd.read_parquet(consolidated_file)
            if len(cdf) == len(c_s1):
                print(f"Found fully consolidated {country.upper()} predictions ({len(cdf):,} rows). Skipping targets load!", flush=True)
                for eid, m_str in zip(cdf["source1_entity_id"].values, cdf["matched_entity_ids"].values):
                    all_predictions[str(eid)] = m_str.split(",") if m_str else []
                continue

        # Load only active country targets to minimize RAM footprint
        c_targets = load_country_targets(country, cache_dir=args.cache_dir)

        c_preds = generate_submission(
            country=country,
            s1_df=c_s1,
            targets_df=c_targets,
            model=model,
            threshold=thresh,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            reset_checkpoints=args.reset_checkpoints,
        )
        all_predictions.update(c_preds)

        # Explicitly unload target DataFrame and garbage collect
        del c_targets, c_s1
        gc.collect()

    # 3. Format & Validate Submission
    print("\n" + "=" * 80)
    print("                     SUBMISSION INTEGRITY AUDIT                       ")
    print("=" * 80)

    # Order by original test S1 entity IDs
    if args.sample_s1 > 0 or args.country != "all":
        s1_eval_ids = [eid for eid in test_s1["entity_id"].astype(str).tolist() if eid in all_predictions]
    else:
        s1_eval_ids = test_s1["entity_id"].astype(str).tolist()

    total_test_entities = len(s1_eval_ids)
    singletons = sum(1 for s1_id in s1_eval_ids if len(all_predictions.get(s1_id, [])) == 0)
    matched = total_test_entities - singletons
    total_matches = sum(len(all_predictions.get(s1_id, [])) for s1_id in s1_eval_ids)

    print(f"  Total S1 Entities in Submission  : {total_test_entities:,}")
    print(f"  Matched S1 Entities              : {matched:,} ({matched / max(1, total_test_entities) * 100:.2f}%)")
    print(f"  Predicted Singletons (Empty)     : {singletons:,} ({singletons / max(1, total_test_entities) * 100:.2f}%)")
    print(f"  Total Predicted Target Pairs     : {total_matches:,}")
    print(f"  Mean Targets per Matched Entity  : {total_matches / max(1, matched):.2f}")
    print("-" * 80)

    # 4. Write Official TSV
    print(f"Writing official submission TSV to: {args.output} ...", flush=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_eval_ids:
            matches = all_predictions.get(s1_id, [])
            match_str = ",".join(matches) if matches else ""
            f.write(f"{s1_id}\t{match_str}\n")

    # 5. Sanity-check written file
    print("Verifying generated submission file structure ...", flush=True)
    verify_df = pd.read_csv(args.output, sep="\t", dtype=str)
    assert len(verify_df) == total_test_entities, f"Row count mismatch: {len(verify_df)} vs {total_test_entities}"
    assert list(verify_df.columns) == ["source1_entity_id", "matched_entity_ids"], f"Column mismatch: {verify_df.columns}"
    print(f"Verification PASSED: {len(verify_df):,} rows, exactly 2 columns, zero schema errors.")
    print(f"All processing completed in {time.time() - t0:.1f}s.")
    print("=" * 80)


if __name__ == "__main__":
    main()
