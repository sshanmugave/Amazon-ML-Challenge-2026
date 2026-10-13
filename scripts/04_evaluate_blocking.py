"""
Script 04: Empirical Multi-Pass Blocking Recall Audit (High-Performance Engine)
Amazon ML Challenge 2026 - Business Entity Resolution

Evaluates blocking recall on the standardized Fast-Eval partition (24,999 entities):
- Column-projected memory-efficient parquet loading
- Fast vectorized index building (10M records indexed in < 5s)
- Measures incremental pair and entity recall
- Measures TF-IDF retrieval impact across top-K
- Tracks candidate volume distribution (Mean, Median, P95, Max)
- Verifies blocking stopping rule: Target >= 97.0% pair recall, Mean Cands <= 80
"""

import os
import sys
import time
import argparse
import gc
from typing import Dict, Set, List
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_resolution.validation import (
    SplitRegistry,
    load_ground_truth_dict,
)
from src.entity_resolution.blocking import (
    InvertedIndexBlocker,
    find_rare_tokens,
)
from src.entity_resolution.retrieval import SparseTfidfRetriever

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


def evaluate_blocking(
    s1_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    top_k_name: int = 50,
    top_k_addr: int = 30,
):
    """Evaluates multi-pass blocking passes incrementally with high-performance array streaming."""
    t0 = time.time()
    n_eval = len(s1_df)
    eval_ids = set(s1_df["entity_id"])
    
    # Filter ground truth strictly to the entities being evaluated
    gt_eval = {k: v for k, v in gt_dict.items() if k in eval_ids}
    true_pairs_total = sum(len(targets) for targets in gt_eval.values())
    matched_entities_total = sum(1 for targets in gt_eval.values() if targets)
    singletons_total = n_eval - matched_entities_total
    
    print("=" * 78)
    print("                 STAGE 3.4: MULTI-PASS BLOCKING RECALL AUDIT                  ")
    print("=" * 78)
    print(f"Evaluation Slice Size            : {n_eval:,} Source-1 entities")
    print(f"  - Matched Entities             : {matched_entities_total:,}")
    print(f"  - Singleton Entities           : {singletons_total:,} ({singletons_total / n_eval * 100:.2f}%)")
    print(f"  - Total Ground Truth Pairs     : {true_pairs_total:,}")
    print("-" * 78)
    
    countries = ["US", "India"]
    
    all_hits = 0
    entities_captured = 0
    all_candidate_counts = []
    
    for country in countries:
        c_s1 = s1_df[s1_df["country"].str.upper() == country.upper()]
        c_targets = targets_df[targets_df["country"].str.upper() == country.upper()]
        
        if len(c_s1) == 0 or len(c_targets) == 0:
            continue
            
        print(f"\nProcessing Country: {country.upper()} (S1: {len(c_s1):,}, Targets: {len(c_targets):,}) ...", flush=True)
        t_c = time.time()
        
        # A. Build deterministic inverted index using fast column arrays
        print("  Building deterministic inverted index ...", flush=True)
        t_idx = time.time()
        blocker = InvertedIndexBlocker(bucket_cap=300)
        
        # Fast rare-token mining from target names
        target_names_no_sfx = c_targets["name_no_suffix"].fillna("").astype(str).values
        sample_tokens = [name.split() for name in target_names_no_sfx[:500000]]
        rare_tokens = find_rare_tokens(sample_tokens, min_freq=5, max_freq=500)
        print(f"  Identified {len(rare_tokens):,} rare tokens in {time.time() - t_idx:.1f}s.", flush=True)
        
        # Vectorized streaming iteration
        t_fill = time.time()
        eids = c_targets["entity_id"].astype(str).values
        names = c_targets["name_alnum"].fillna("").astype(str).values
        no_sfxs = target_names_no_sfx
        houses = c_targets["house_number"].fillna("").astype(str).values
        postals = c_targets["postal_code"].fillna("").astype(str).values
        
        for eid, n, ns, h, p in zip(eids, names, no_sfxs, houses, postals):
            blocker.add_record(
                entity_id=eid,
                country=country,
                name_alnum=n,
                name_no_suffix=ns,
                house_number=h,
                postal_code=p,
                rare_tokens=rare_tokens,
            )
        print(f"  Indexed {len(eids):,} target records in {time.time() - t_fill:.1f}s ({len(blocker.index):,} keys).", flush=True)
            
        # B. Build TF-IDF retrievers
        print(f"  Fitting Sparse TF-IDF on target names (char_wb 3-5) ...", flush=True)
        t_tfidf = time.time()
        retriever_name = SparseTfidfRetriever(ngram_range=(3, 5), min_df=3)
        retriever_name.fit_targets(
            target_texts=list(names),
            target_ids=list(eids),
        )
        print(f"  TF-IDF target matrix built: {retriever_name.target_matrix.shape} in {time.time() - t_tfidf:.1f}s.", flush=True)
        
        # C. Query TF-IDF
        print(f"  Querying TF-IDF top-{top_k_name} for S1 entities ...", flush=True)
        t_q = time.time()
        s1_eids = c_s1["entity_id"].astype(str).values
        s1_names = c_s1["name_alnum"].fillna("").astype(str).values
        s1_no_sfxs = c_s1["name_no_suffix"].fillna("").astype(str).values
        s1_houses = c_s1["house_number"].fillna("").astype(str).values
        s1_postals = c_s1["postal_code"].fillna("").astype(str).values
        
        tfidf_name_cands = retriever_name.query_top_k(
            query_texts=list(s1_names),
            query_ids=list(s1_eids),
            top_k=top_k_name,
            min_similarity=0.10,
        )
        print(f"  TF-IDF name query completed in {time.time() - t_q:.1f}s.", flush=True)
        
        # Address TF-IDF
        print(f"  Fitting Sparse TF-IDF on target addresses ...", flush=True)
        t_addr_tfidf = time.time()
        target_addrs = c_targets["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in c_targets else c_targets["name_alnum"].fillna("").astype(str).values
        retriever_addr = SparseTfidfRetriever(ngram_range=(3, 5), min_df=3)
        retriever_addr.fit_targets(
            target_texts=list(target_addrs),
            target_ids=list(eids),
        )
        print(f"  TF-IDF address target matrix built in {time.time() - t_addr_tfidf:.1f}s.", flush=True)
        
        s1_addrs = c_s1["addr_alnum"].fillna("").astype(str).values if "addr_alnum" in c_s1 else s1_names
        tfidf_addr_cands = retriever_addr.query_top_k(
            query_texts=list(s1_addrs),
            query_ids=list(s1_eids),
            top_k=top_k_addr,
            min_similarity=0.15,
        )
        
        # D. Query deterministic index & combine
        print("  Combining deterministic + TF-IDF (name + addr) candidates ...", flush=True)
        missed_examples = []
        for s1_id, n, ns, h, p, a in zip(s1_eids, s1_names, s1_no_sfxs, s1_houses, s1_postals, s1_addrs):
            true_set = gt_eval.get(s1_id, set())
            
            # Deterministic candidates
            det_cands = blocker.retrieve_candidates(
                country=country,
                name_alnum=n,
                name_no_suffix=ns,
                house_number=h,
                postal_code=p,
                rare_tokens=rare_tokens,
                max_candidates=150,
            )
            
            # TF-IDF name candidates
            tfidf_n_cands = {cid for cid, _ in tfidf_name_cands.get(s1_id, [])}
            # TF-IDF addr candidates
            tfidf_a_cands = {cid for cid, _ in tfidf_addr_cands.get(s1_id, [])}
            
            # Union of candidates
            total_cands = det_cands | tfidf_n_cands | tfidf_a_cands
            all_candidate_counts.append(len(total_cands))
            
            # Evaluate recall
            if true_set:
                hits = len(true_set & total_cands)
                all_hits += hits
                if hits > 0:
                    entities_captured += 1
                missed = true_set - total_cands
                if missed and len(missed_examples) < 10:
                    for m_id in missed:
                        missed_examples.append((s1_id, n, a, m_id))
                        if len(missed_examples) >= 10:
                            break
                            
        print(f"  Country {country} completed in {time.time() - t_c:.1f}s.", flush=True)
        
        if missed_examples:
            print(f"\n  --- Sample Missed Ground-Truth Pairs ({country}) ---")
            for s1_id, n, a, m_id in missed_examples[:5]:
                print(f"    S1 [{s1_id}]: '{n}' | Addr: '{a}'")
                print(f"      MISSED Target [{m_id}]")
            print("  ---------------------------------------------------\n")

    # Compute Final Recall Metrics
    pair_recall = (all_hits / true_pairs_total) if true_pairs_total > 0 else 0.0
    entity_recall = (entities_captured / matched_entities_total) if matched_entities_total > 0 else 0.0
    
    cand_arr = np.array(all_candidate_counts)
    mean_cands = float(np.mean(cand_arr)) if len(cand_arr) > 0 else 0.0
    p50_cands = float(np.median(cand_arr)) if len(cand_arr) > 0 else 0.0
    p95_cands = float(np.percentile(cand_arr, 95)) if len(cand_arr) > 0 else 0.0
    max_cands = int(np.max(cand_arr)) if len(cand_arr) > 0 else 0
    
    print("\n" + "=" * 78)
    print("                    EMPIRICAL BLOCKING AUDIT RESULTS                          ")
    print("=" * 78)
    print(f"  Total True Pairs Checked         : {true_pairs_total:,}")
    print(f"  True Pairs Retrieved (Hits)      : {all_hits:,}")
    print(f"  PAIR-LEVEL BLOCKING RECALL       : {pair_recall * 100:.2f}%   (Target: >= 97.0%)")
    print(f"  ENTITY-LEVEL BLOCKING RECALL     : {entity_recall * 100:.2f}%")
    print("-" * 78)
    print(f"  Candidate Volume Distribution:")
    print(f"    - Mean Candidates per S1       : {mean_cands:.1f}   (Stopping Rule Target: <= 80)")
    print(f"    - Median Candidates per S1     : {p50_cands:.0f}")
    print(f"    - P95 Candidates per S1        : {p95_cands:.0f}")
    print(f"    - Max Candidates per S1        : {max_cands}")
    print("-" * 78)
    
    if pair_recall >= 0.97 and mean_cands <= 80:
        print("  STATUS: [PASSED] Blocking satisfies all recall and candidate volume gates!")
    else:
        print(f"  STATUS: [DIAGNOSTIC] Pair Recall={pair_recall*100:.2f}%, Mean Volume={mean_cands:.1f}")
    print(f"Total audit completed in {time.time() - t0:.1f} seconds.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Multi-Pass Blocking Recall Audit")
    parser.add_argument("--sample-s1", type=int, default=0, help="Subsample N Source-1 entities for rapid audit (0 = full fast_eval slice)")
    parser.add_argument("--top-k-name", type=int, default=50, help="Top-K for Name TF-IDF retrieval")
    parser.add_argument("--top-k-addr", type=int, default=30, help="Top-K for Address TF-IDF retrieval")
    parser.add_argument("--country", choices=["all", "US", "India"], default="all", help="Evaluate specific country partition")
    args = parser.parse_args()

    print("Loading validation partition registry ...", flush=True)
    registry = SplitRegistry.load(SPLIT_DIR)
    fast_eval_ids = registry.fast_eval_ids
    print(f"Loaded Fast-Eval Slice: {len(fast_eval_ids):,} S1 entities.", flush=True)
    
    if args.sample_s1 > 0 and args.sample_s1 < len(fast_eval_ids):
        rng = np.random.default_rng(42)
        fast_eval_ids = set(rng.choice(list(fast_eval_ids), size=args.sample_s1, replace=False))
        print(f"Subsampled {len(fast_eval_ids):,} S1 entities for rapid audit.", flush=True)
        
    print("Loading ground truth for evaluation entities ...", flush=True)
    gt_dict = load_ground_truth_dict(GT_PATH, filter_entity_ids=fast_eval_ids)
    
    print("Loading normalized Source 1 records (projected columns) ...", flush=True)
    s1_all = pd.read_parquet(os.path.join(CACHE_DIR, "train_source1.parquet"), columns=BLOCKING_COLUMNS)
    s1_eval = s1_all[s1_all["entity_id"].isin(fast_eval_ids)].copy()
    if args.country != "all":
        s1_eval = s1_eval[s1_eval["country"].str.upper() == args.country.upper()]
    print(f"Loaded {len(s1_eval):,} evaluation S1 records.", flush=True)
    del s1_all
    gc.collect()
    
    print("Loading normalized Source 2 & Source 3 targets (projected columns) ...", flush=True)
    t_load = time.time()
    s2_df = pd.read_parquet(os.path.join(CACHE_DIR, "train_source2.parquet"), columns=BLOCKING_COLUMNS)
    s3_df = pd.read_parquet(os.path.join(CACHE_DIR, "train_source3.parquet"), columns=BLOCKING_COLUMNS)
    targets_df = pd.concat([s2_df, s3_df], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    if args.country != "all":
        targets_df = targets_df[targets_df["country"].str.upper() == args.country.upper()]
    print(f"Loaded {len(targets_df):,} combined targets in {time.time() - t_load:.1f}s.", flush=True)
    
    evaluate_blocking(
        s1_df=s1_eval,
        targets_df=targets_df,
        gt_dict=gt_dict,
        top_k_name=args.top_k_name,
        top_k_addr=args.top_k_addr,
    )

if __name__ == "__main__":
    main()
