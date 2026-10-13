"""
Feature Engineering Engine for Pairwise Entity Resolution
Amazon ML Challenge 2026 - Business Entity Resolution

Implements the 36 Audited Pairwise Features (28 Core + 8 Experimental):
- RapidFuzz high-throughput string metrics (ratio, token sort, jaro-winkler, partial)
- TF-IDF character and word sparse cosine dot products
- Jaccard token and rare-token set overlaps
- Address numeric, house number, and postal code verifications
- Contextual metadata (source indicator, string lengths, legal suffixes)
- Non-linear cross features (name x addr, high-name + exact postal/house)
"""

from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from rapidfuzz import fuzz, distance


FEATURE_NAMES: List[str] = [
    # --- Name Features (Core: 1-12, Exp: 13-14) ---
    "name_exact_match",          # 1
    "name_ratio",                # 2
    "name_token_sort_ratio",     # 3
    "name_jaro_winkler",         # 4
    "name_char_tfidf_cosine",    # 5
    "name_word_tfidf_cosine",    # 6
    "name_token_jaccard",        # 7
    "name_rare_token_overlap",   # 8
    "name_len_ratio",            # 9
    "name_token_count_diff",     # 10
    "name_no_suffix_ratio",      # 11
    "name_suffix_match",         # 12
    "name_partial_ratio",        # 13 (Exp)
    "name_token_set_ratio",      # 14 (Exp)
    
    # --- Address Features (Core: 15-23, Exp: 24-25) ---
    "addr_exact_match",          # 15
    "addr_ratio",                # 16
    "addr_char_tfidf_cosine",    # 17
    "addr_token_jaccard",        # 18
    "addr_numeric_overlap",      # 19
    "addr_house_number_eq",      # 20
    "addr_postal_eq",            # 21
    "addr_len_ratio",            # 22
    "is_addr_missing",           # 23
    "addr_token_set_ratio",      # 24 (Exp)
    "addr_postal_prefix",        # 25 (Exp)
    
    # --- Context & Source Features (Core: 26-32) ---
    "is_source2",                # 26
    "is_source3",                # 27
    "name_len",                  # 28
    "addr_len",                  # 29
    "name_token_count",          # 30
    "name_has_legal_suffix",     # 31
    "candidate_retrieval_score", # 32
    
    # --- Cross Features (Core: 33, Exp: 34-36) ---
    "name_x_addr",               # 33
    "high_name_exact_postal",    # 34 (Exp)
    "high_name_exact_house",     # 35 (Exp)
    "rare_name_rare_addr",       # 36 (Exp)
]

NUM_FEATURES = len(FEATURE_NAMES)
assert NUM_FEATURES == 36, f"Expected 36 features, got {NUM_FEATURES}"


def extract_record_tokens(
    name_alnum: str,
    name_no_suffix: str,
    addr_alnum: str,
    rare_tokens: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Precomputes token sets and numeric sets for a single record to avoid
    re-tokenizing repeatedly during pairwise comparison.
    """
    n_toks = set(name_alnum.split()) if name_alnum else set()
    ns_toks = set(name_no_suffix.split()) if name_no_suffix else n_toks
    a_toks = set(addr_alnum.split()) if addr_alnum else set()
    
    rare_toks = (ns_toks & rare_tokens) if rare_tokens else set()
    
    # Extract numerics from address
    nums = {w for w in a_toks if any(c.isdigit() for c in w)}
    
    return {
        "name_tokens": n_toks,
        "name_no_suffix_tokens": ns_toks,
        "addr_tokens": a_toks,
        "rare_tokens": rare_toks,
        "addr_numerics": nums,
    }


def compute_pair_features(
    # Source-1 record fields
    s1_id: str,
    s1_name: str,
    s1_name_no_sfx: str,
    s1_sfx: str,
    s1_addr: str,
    s1_house: str,
    s1_postal: str,
    s1_cache: Dict[str, Any],
    
    # Candidate target record fields
    cand_id: str,
    cand_name: str,
    cand_name_no_sfx: str,
    cand_sfx: str,
    cand_addr: str,
    cand_house: str,
    cand_postal: str,
    cand_cache: Dict[str, Any],
    
    # Optional precomputed TF-IDF / retrieval similarities
    name_tfidf_sim: float = 0.0,
    word_tfidf_sim: float = 0.0,
    addr_tfidf_sim: float = 0.0,
    retrieval_score: float = 0.0,
) -> np.ndarray:
    """
    Computes a 36-dimensional feature vector for a single (Source-1, Target) pair.
    Returns a float32 NumPy array of length 36.
    """
    features = np.zeros(NUM_FEATURES, dtype=np.float32)
    
    # Cached token sets
    s1_n_toks = s1_cache.get("name_tokens", set())
    c_n_toks = cand_cache.get("name_tokens", set())
    s1_rare = s1_cache.get("rare_tokens", set())
    c_rare = cand_cache.get("rare_tokens", set())
    s1_a_toks = s1_cache.get("addr_tokens", set())
    c_a_toks = cand_cache.get("addr_tokens", set())
    s1_nums = s1_cache.get("addr_numerics", set())
    c_nums = cand_cache.get("addr_numerics", set())
    
    # -------------------------------------------------------------
    # 1. NAME FEATURES (1-14)
    # -------------------------------------------------------------
    exact_name = (s1_name == cand_name) and bool(s1_name)
    features[0] = 1.0 if exact_name else 0.0
    
    if exact_name:
        ratio = 1.0
        token_sort = 1.0
        jw = 1.0
        partial = 1.0
        token_set = 1.0
        no_sfx_ratio = 1.0
    else:
        ratio = fuzz.ratio(s1_name, cand_name) / 100.0
        token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        jw = distance.JaroWinkler.similarity(s1_name, cand_name)
        partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
        token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        no_sfx_ratio = fuzz.ratio(s1_name_no_sfx, cand_name_no_sfx) / 100.0 if (s1_name_no_sfx and cand_name_no_sfx) else ratio
        
    features[1] = ratio
    features[2] = token_sort
    features[3] = jw
    features[4] = name_tfidf_sim
    features[5] = word_tfidf_sim
    
    # Token Jaccard
    union_n = len(s1_n_toks | c_n_toks)
    features[6] = (len(s1_n_toks & c_n_toks) / union_n) if union_n > 0 else 0.0
    
    # Rare Token overlap
    union_r = len(s1_rare | c_rare)
    features[7] = (len(s1_rare & c_rare) / union_r) if union_r > 0 else 0.0
    
    # Length Ratio
    len_s1_n = len(s1_name)
    len_c_n = len(cand_name)
    max_n_len = max(len_s1_n, len_c_n)
    features[8] = (min(len_s1_n, len_c_n) / max_n_len) if max_n_len > 0 else 0.0
    
    # Token count diff
    features[9] = float(abs(len(s1_n_toks) - len(c_n_toks)))
    features[10] = no_sfx_ratio
    
    # Suffix match
    if s1_sfx and cand_sfx:
        features[11] = 1.0 if s1_sfx == cand_sfx else 0.0
    elif not s1_sfx and not cand_sfx:
        features[11] = 0.5
    else:
        features[11] = 0.0
        
    features[12] = partial
    features[13] = token_set
    
    # -------------------------------------------------------------
    # 2. ADDRESS FEATURES (15-25)
    # -------------------------------------------------------------
    has_s1_addr = bool(s1_addr)
    has_cand_addr = bool(cand_addr)
    
    if has_s1_addr and has_cand_addr:
        exact_addr = (s1_addr == cand_addr)
        features[14] = 1.0 if exact_addr else 0.0
        features[15] = 1.0 if exact_addr else (fuzz.ratio(s1_addr, cand_addr) / 100.0)
        
        # Token Jaccard
        union_a = len(s1_a_toks | c_a_toks)
        features[17] = (len(s1_a_toks & c_a_toks) / union_a) if union_a > 0 else 0.0
        
        # Numeric overlap
        union_nums = len(s1_nums | c_nums)
        features[18] = (len(s1_nums & c_nums) / union_nums) if union_nums > 0 else 0.0
        
        # Length ratio
        len_s1_a = len(s1_addr)
        len_c_a = len(cand_addr)
        max_a_len = max(len_s1_a, len_c_a)
        features[21] = (min(len_s1_a, len_c_a) / max_a_len) if max_a_len > 0 else 0.0
        features[23] = 1.0 if exact_addr else (fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0)
    else:
        features[14] = 0.0
        features[15] = 0.0
        features[17] = 0.0
        features[18] = 0.0
        features[21] = 0.0
        features[23] = 0.0
        
    features[16] = addr_tfidf_sim
    
    # House number match
    if s1_house and cand_house:
        features[19] = 1.0 if s1_house == cand_house else 0.0
    elif not s1_house and not cand_house:
        features[19] = 0.5
    else:
        features[19] = 0.0
        
    # Postal code match
    if s1_postal and cand_postal:
        features[20] = 1.0 if s1_postal == cand_postal else 0.0
        features[24] = 1.0 if (len(s1_postal) >= 3 and len(cand_postal) >= 3 and s1_postal[:3] == cand_postal[:3]) else 0.0
    elif not s1_postal and not cand_postal:
        features[20] = 0.5
        features[24] = 0.5
    else:
        features[20] = 0.0
        features[24] = 0.0
        
    features[22] = 0.0 if has_cand_addr else 1.0
    
    # -------------------------------------------------------------
    # 3. CONTEXT & SOURCE FEATURES (26-32)
    # -------------------------------------------------------------
    features[25] = 1.0 if cand_id.startswith("S2") else 0.0
    features[26] = 1.0 if cand_id.startswith("S3") else 0.0
    features[27] = float(len_s1_n)
    features[28] = float(len(s1_addr))
    features[29] = float(len(s1_n_toks))
    features[30] = 1.0 if bool(s1_sfx) else 0.0
    features[31] = retrieval_score
    
    # -------------------------------------------------------------
    # 4. CROSS FEATURES (33-36)
    # -------------------------------------------------------------
    features[32] = name_tfidf_sim * addr_tfidf_sim
    features[33] = 1.0 if (ratio > 0.85 and features[20] == 1.0) else 0.0
    features[34] = 1.0 if (ratio > 0.85 and features[19] == 1.0) else 0.0
    features[35] = 1.0 if (features[7] > 0.0 and features[17] > 0.0) else 0.0
    
    return features
