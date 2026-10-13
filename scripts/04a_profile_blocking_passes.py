"""
Script 04a: Multi-Pass Blocking Diagnostic & Ablation Analysis
Amazon ML Challenge 2026 - Business Entity Resolution

Measures:
1. Exact recall contribution of each individual deterministic pass (Pass 1, 2, 3, 4, 7, 8)
2. Cumulative deterministic recall and mean candidate volume
3. Name TF-IDF incremental recall across K in [10, 20, 30, 50]
4. Address TF-IDF incremental recall across K in [10, 20, 30]
5. Effect of TF-IDF max_df filtering (removing ultra-frequent noise n-grams)
"""

import os
import sys
import time
import argparse
from typing import Dict, Set, List, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.validation import SplitRegistry, load_ground_truth_dict
from src.entity_resolution.blocking import find_rare_tokens
import itertools

DEFAULT_CACHE_DIR = r"cache\normalized"
DEFAULT_SPLIT_DIR = r"cache\splits"
DEFAULT_GT_PATH = os.environ.get(
    "AMAZON_ML_GT_PATH",
    r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train\train_ground_truth.tsv"
    if os.path.exists(r"e:\KCT\amzon ml hackthon\dataset\student_resource\dataset\train\train_ground_truth.tsv")
    else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "train", "train_ground_truth.tsv")
)

BLOCKING_COLUMNS = [
    "entity_id",
    "country",
    "name_alnum",
    "name_no_suffix",
    "addr_alnum",
    "house_number",
    "postal_code",
]


def generate_pass_keys(
    country: str,
    name_alnum: str,
    name_no_suffix: str,
    house_number: str,
    postal_code: str,
    rare_tokens: Set[str],
) -> Dict[str, List[str]]:
    """Generates keys partitioned by pass name."""
    c = country.strip().upper()
    keys_by_pass = defaultdict(list)
    if not c or not name_alnum:
        return keys_by_pass

    tokens = name_no_suffix.split() if name_no_suffix else name_alnum.split()
    n_tokens = len(tokens)

    # Pass 1: Exact normalized name
    keys_by_pass["pass_1_exact"].append(f"{c}|N1:{name_alnum}")

    # Pass 2: Rare tokens
    if rare_tokens and tokens:
        for tok in tokens:
            if tok in rare_tokens:
                keys_by_pass["pass_2_rare"].append(f"{c}|RT:{tok}")

    # Pass 3: Token bigrams
    if n_tokens >= 2:
        top_tokens = tokens[:4]
        for t1, t2 in itertools.combinations(top_tokens, 2):
            if t1 != t2:
                b_min, b_max = (t1, t2) if t1 < t2 else (t2, t1)
                keys_by_pass["pass_3_bigram"].append(f"{c}|BG:{b_min}_{b_max}")

    # Pass 4: Name prefix 4 + house number
    if house_number and len(name_alnum) >= 4:
        pfx4 = name_alnum[:4]
        keys_by_pass["pass_4_pfx4_house"].append(f"{c}|NH:{pfx4}|{house_number}")

    # Pass 7: Token 0 + postal code
    if postal_code and tokens:
        keys_by_pass["pass_7_tok_postal"].append(f"{c}|NP:{tokens[0]}|{postal_code}")

    # Pass 8: House number + name prefix 3
    if house_number and len(name_alnum) >= 3:
        pfx3 = name_alnum[:3]
        keys_by_pass["pass_8_house_pfx3"].append(f"{c}|AN:{house_number}|{pfx3}")

    return keys_by_pass


def main():
    parser = argparse.ArgumentParser(description="Multi-Pass Blocking Ablation Study")
    parser.add_argument("--country", choices=["US", "India"], default="US")
    parser.add_argument("--sample-s1", type=int, default=500, help="Number of S1 entities to evaluate")
    parser.add_argument("--bucket-cap", type=int, default=200, help="Inverted index bucket cap")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Normalized parquet cache directory")
    parser.add_argument("--split-dir", default=DEFAULT_SPLIT_DIR, help="Validation splits directory")
    parser.add_argument("--gt-path", default=DEFAULT_GT_PATH, help="Train ground truth TSV path")
    args = parser.parse_args()

    print("=" * 80)
    print("        STAGE 3.4: BLOCKING PASS ABLATION & EFFICIENCY PROFILING        ")
    print("=" * 80)
    
    # 1. Load registry and S1 entities
    registry = SplitRegistry.load(args.split_dir)
    fast_eval_ids = registry.fast_eval_ids
    gt_dict = load_ground_truth_dict(args.gt_path)

    print(f"Loading normalized S1 records for {args.country} ...")
    s1_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source1.parquet"), columns=BLOCKING_COLUMNS)
    c_s1 = s1_df[(s1_df["country"].str.upper() == args.country.upper()) & (s1_df["entity_id"].isin(fast_eval_ids))]
    
    if args.sample_s1 > 0 and len(c_s1) > args.sample_s1:
        c_s1 = c_s1.sample(n=args.sample_s1, random_state=42).reset_index(drop=True)
    eval_s1_ids = set(c_s1["entity_id"])
    c_gt = {k: v for k, v in gt_dict.items() if k in eval_s1_ids}
    total_true_pairs = sum(len(v) for v in c_gt.values())
    total_matched_s1 = sum(1 for v in c_gt.values() if len(v) > 0)
    
    print(f"Evaluation sample: {len(c_s1):,} S1 entities ({total_matched_s1:,} matched, {total_true_pairs:,} true pairs)")

    # 2. Load targets
    print(f"Loading normalized targets (S2 + S3) for {args.country} ...")
    t0 = time.time()
    s2_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source2.parquet"), columns=BLOCKING_COLUMNS)
    s3_df = pd.read_parquet(os.path.join(args.cache_dir, "train_source3.parquet"), columns=BLOCKING_COLUMNS)
    targets_df = pd.concat([
        s2_df[s2_df["country"].str.upper() == args.country.upper()],
        s3_df[s3_df["country"].str.upper() == args.country.upper()],
    ], ignore_index=True)
    del s2_df, s3_df
    print(f"Loaded {len(targets_df):,} targets in {time.time() - t0:.1f}s.")

    # 3. Mine rare tokens
    t_rt = time.time()
    target_names_no_sfx = targets_df["name_no_suffix"].fillna("").astype(str).values
    sample_tokens = [name.split() for name in target_names_no_sfx[:500000]]
    rare_tokens = find_rare_tokens(sample_tokens, min_freq=5, max_freq=500)
    print(f"Identified {len(rare_tokens):,} rare tokens in {time.time() - t_rt:.1f}s.")

    # 4. Build inverted indexes per pass
    pass_names = [
        "pass_1_exact",
        "pass_2_rare",
        "pass_3_bigram",
        "pass_4_pfx4_house",
        "pass_7_tok_postal",
        "pass_8_house_pfx3",
    ]
    pass_indexes: Dict[str, Dict[str, List[str]]] = {p: defaultdict(list) for p in pass_names}

    print("Building separate inverted indexes per deterministic pass ...")
    t_idx = time.time()
    t_eids = targets_df["entity_id"].astype(str).values
    t_names = targets_df["name_alnum"].fillna("").astype(str).values
    t_houses = targets_df["house_number"].fillna("").astype(str).values
    t_postals = targets_df["postal_code"].fillna("").astype(str).values

    for eid, n, ns, h, p in zip(t_eids, t_names, target_names_no_sfx, t_houses, t_postals):
        keys_by_pass = generate_pass_keys(args.country, n, ns, h, p, rare_tokens)
        for pname, keys in keys_by_pass.items():
            idx = pass_indexes[pname]
            for k in keys:
                bucket = idx[k]
                if len(bucket) < args.bucket_cap:
                    bucket.append(eid)
    print(f"Built all pass indexes in {time.time() - t_idx:.1f}s.")
    for pname in pass_names:
        print(f"  - {pname}: {len(pass_indexes[pname]):,} unique keys")

    # 5. Measure Individual & Cumulative Deterministic Passes
    print("\n" + "=" * 80)
    print("           INDIVIDUAL & CUMULATIVE DETERMINISTIC PASS ABLATION           ")
    print("=" * 80)
    
    s1_eids = c_s1["entity_id"].astype(str).values
    s1_names = c_s1["name_alnum"].fillna("").astype(str).values
    s1_no_sfxs = c_s1["name_no_suffix"].fillna("").astype(str).values
    s1_houses = c_s1["house_number"].fillna("").astype(str).values
    s1_postals = c_s1["postal_code"].fillna("").astype(str).values

    # Pre-generate query keys
    s1_pass_keys = []
    for eid, n, ns, h, p in zip(s1_eids, s1_names, s1_no_sfxs, s1_houses, s1_postals):
        s1_pass_keys.append(generate_pass_keys(args.country, n, ns, h, p, rare_tokens))

    # Test each pass individually
    pass_candidates: Dict[str, Dict[str, Set[str]]] = {p: {} for p in pass_names}
    for pname in pass_names:
        idx = pass_indexes[pname]
        hits = 0
        cand_counts = []
        for i, qid in enumerate(s1_eids):
            qkeys = s1_pass_keys[i].get(pname, [])
            cands = set()
            for k in qkeys:
                if k in idx:
                    cands.update(idx[k])
            pass_candidates[pname][qid] = cands
            cand_counts.append(len(cands))
            true_set = c_gt.get(qid, set())
            if true_set:
                hits += len(true_set & cands)
        rec = (hits / total_true_pairs) * 100 if total_true_pairs > 0 else 0.0
        print(f"  {pname:<22} : Pair Recall = {rec:6.2f}% | Mean Cands = {np.mean(cand_counts):6.1f} | Max Cands = {np.max(cand_counts)}")

    # Test cumulative addition
    print("-" * 80)
    print("  Cumulative Pass Combination:")
    cumulative_cands: Dict[str, Set[str]] = {qid: set() for qid in s1_eids}
    for pname in pass_names:
        for qid in s1_eids:
            cumulative_cands[qid].update(pass_candidates[pname][qid])
        hits = sum(len(c_gt.get(qid, set()) & cumulative_cands[qid]) for qid in s1_eids)
        rec = (hits / total_true_pairs) * 100 if total_true_pairs > 0 else 0.0
        c_counts = [len(cumulative_cands[qid]) for qid in s1_eids]
        print(f"  + {pname:<20} : Pair Recall = {rec:6.2f}% | Mean Cands = {np.mean(c_counts):6.1f} | Max = {np.max(c_counts)}")

    # 6. TF-IDF Ablation with max_df=0.15
    print("\n" + "=" * 80)
    print("          TF-IDF RETRIEVAL ABLATION (max_df=0.15, sublinear_tf)          ")
    print("=" * 80)
    print("Fitting TF-IDF on target names (char_wb 3-5, max_df=0.15) ...", flush=True)
    t_tfidf = time.time()
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.15,
        sublinear_tf=True,
        dtype=np.float32,
    )
    t_mat = vec.fit_transform(t_names)
    print(f"Fitted TF-IDF target matrix: {t_mat.shape} (nnz={t_mat.nnz:,}) in {time.time() - t_tfidf:.1f}s.")

    q_mat = vec.transform(s1_names)
    print("Computing sparse cosine similarity ...", flush=True)
    t_dot = time.time()
    sim_mat = q_mat.dot(t_mat.T)
    print(f"Sparse dot product completed in {time.time() - t_dot:.2f}s.")

    # Sweep top-K in [10, 20, 30, 50]
    for top_k in [10, 20, 30, 50]:
        tfidf_cands = defaultdict(set)
        for r_idx in range(sim_mat.shape[0]):
            qid = s1_eids[r_idx]
            row = sim_mat.getrow(r_idx)
            if row.nnz > 0:
                d = row.data
                idx = row.indices
                mask = d >= 0.15
                if np.any(mask):
                    f_d = d[mask]
                    f_i = idx[mask]
                    if len(f_d) > top_k:
                        part_idx = np.argpartition(f_d, -top_k)[-top_k:]
                        top_ids = f_i[part_idx]
                    else:
                        top_ids = f_i
                    tfidf_cands[qid] = set(t_eids[top_ids])
        
        # Standalone TF-IDF recall
        tf_hits = sum(len(c_gt.get(qid, set()) & tfidf_cands[qid]) for qid in s1_eids)
        tf_rec = (tf_hits / total_true_pairs) * 100 if total_true_pairs > 0 else 0.0
        
        # Combined with deterministic passes
        combined_hits = 0
        comb_counts = []
        for qid in s1_eids:
            comb = cumulative_cands[qid] | tfidf_cands[qid]
            comb_counts.append(len(comb))
            true_set = c_gt.get(qid, set())
            if true_set:
                combined_hits += len(true_set & comb)
        comb_rec = (combined_hits / total_true_pairs) * 100 if total_true_pairs > 0 else 0.0
        
        print(f"  Top-K = {top_k:2d} | Standalone Recall = {tf_rec:5.2f}% | Combined Recall = {comb_rec:5.2f}% | Mean Cands = {np.mean(comb_counts):5.1f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
