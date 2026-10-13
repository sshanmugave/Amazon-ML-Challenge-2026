"""
Unit Tests for Pairwise Feature Engineering
Amazon ML Challenge 2026 - Business Entity Resolution
"""

import pytest
import numpy as np
from src.entity_resolution.features import (
    compute_pair_features,
    extract_record_tokens,
    FEATURE_NAMES,
    NUM_FEATURES,
)


def test_feature_vector_dimension():
    assert len(FEATURE_NAMES) == 36
    assert NUM_FEATURES == 36


def test_exact_match_pair():
    s1_cache = extract_record_tokens("target corporation", "target", "1000 nicollet mall minneapolis mn")
    cand_cache = extract_record_tokens("target corporation", "target", "1000 nicollet mall minneapolis mn")
    
    feats = compute_pair_features(
        s1_id="S1-1",
        s1_name="target corporation",
        s1_name_no_sfx="target",
        s1_sfx="corp",
        s1_addr="1000 nicollet mall minneapolis mn",
        s1_house="1000",
        s1_postal="55403",
        s1_cache=s1_cache,
        cand_id="S2-100",
        cand_name="target corporation",
        cand_name_no_sfx="target",
        cand_sfx="corp",
        cand_addr="1000 nicollet mall minneapolis mn",
        cand_house="1000",
        cand_postal="55403",
        cand_cache=cand_cache,
        name_tfidf_sim=1.0,
        word_tfidf_sim=1.0,
        addr_tfidf_sim=1.0,
        retrieval_score=0.95,
    )
    
    assert len(feats) == 36
    assert not np.isnan(feats).any()
    assert feats[0] == 1.0  # name_exact_match
    assert feats[1] == 1.0  # name_ratio
    assert feats[14] == 1.0 # addr_exact_match
    assert feats[19] == 1.0 # addr_house_number_eq
    assert feats[20] == 1.0 # addr_postal_eq
    assert feats[25] == 1.0 # is_source2
    assert feats[26] == 0.0 # is_source3


def test_disjoint_pair():
    s1_cache = extract_record_tokens("apple inc", "apple", "1 infinite loop cupertino ca")
    cand_cache = extract_record_tokens("microsoft corp", "microsoft", "1 microsoft way redmond wa")
    
    feats = compute_pair_features(
        s1_id="S1-1",
        s1_name="apple inc",
        s1_name_no_sfx="apple",
        s1_sfx="inc",
        s1_addr="1 infinite loop cupertino ca",
        s1_house="1",
        s1_postal="95014",
        s1_cache=s1_cache,
        cand_id="S3-500",
        cand_name="microsoft corp",
        cand_name_no_sfx="microsoft",
        cand_sfx="corp",
        cand_addr="100 microsoft way redmond wa",
        cand_house="100",
        cand_postal="98052",
        cand_cache=cand_cache,
    )
    
    assert feats[0] == 0.0  # name_exact_match
    assert feats[1] < 0.4   # name_ratio
    assert feats[14] == 0.0 # addr_exact_match
    assert feats[19] == 0.0 # house mismatch
    assert feats[20] == 0.0 # postal mismatch
    assert feats[25] == 0.0 # is_source2
    assert feats[26] == 1.0 # is_source3
    assert not np.isnan(feats).any()


def test_missing_address_handling():
    s1_cache = extract_record_tokens("google llc", "google", "")
    cand_cache = extract_record_tokens("google", "google", "")
    
    feats = compute_pair_features(
        s1_id="S1-1",
        s1_name="google llc",
        s1_name_no_sfx="google",
        s1_sfx="llc",
        s1_addr="",
        s1_house="",
        s1_postal="",
        s1_cache=s1_cache,
        cand_id="S2-99",
        cand_name="google",
        cand_name_no_sfx="google",
        cand_sfx="",
        cand_addr="",
        cand_house="",
        cand_postal="",
        cand_cache=cand_cache,
    )
    
    assert len(feats) == 36
    assert not np.isnan(feats).any()
    assert feats[22] == 1.0 # is_addr_missing
