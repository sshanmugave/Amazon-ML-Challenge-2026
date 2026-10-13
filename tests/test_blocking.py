"""
Unit Tests for Blocking and Sparse TF-IDF Retrieval
Amazon ML Challenge 2026
"""

import pytest
import numpy as np

from src.entity_resolution.blocking import (
    generate_blocking_keys,
    find_rare_tokens,
    InvertedIndexBlocker,
)
from src.entity_resolution.retrieval import SparseTfidfRetriever


def test_generate_blocking_keys():
    keys = generate_blocking_keys(
        country="US",
        name_alnum="blue dart express inc",
        name_no_suffix="blue dart express",
        house_number="123",
        postal_code="94107",
        rare_tokens={"dart"},
    )
    
    # Verify Pass 1
    assert "US|N1:blue dart express inc" in keys
    # Verify Pass 2 (rare token)
    assert "US|RT:dart" in keys
    # Verify Pass 3 (sorted bigrams)
    assert "US|BG:blue_dart" in keys
    assert "US|BG:dart_express" in keys
    # Verify Pass 4 (prefix + house number)
    assert "US|NH:blue|123" in keys
    # Verify Pass 7 (token + postal)
    assert "US|NP:blue|94107" in keys


def test_inverted_index_blocker_roundtrip():
    blocker = InvertedIndexBlocker(bucket_cap=10)
    
    # Add target records
    blocker.add_record(
        entity_id="S2-001",
        country="US",
        name_alnum="starbucks coffee company",
        name_no_suffix="starbucks coffee",
        house_number="100",
        postal_code="98101",
    )
    blocker.add_record(
        entity_id="S3-002",
        country="US",
        name_alnum="starbucks reserve roastery",
        name_no_suffix="starbucks reserve roastery",
        house_number="1124",
        postal_code="98101",
    )
    
    # Query with S1 record
    cands = blocker.retrieve_candidates(
        country="US",
        name_alnum="starbucks coffee",
        name_no_suffix="starbucks coffee",
        house_number="100",
        postal_code="98101",
        max_candidates=50,
    )
    
    assert "S2-001" in cands


def test_sparse_tfidf_retriever():
    retriever = SparseTfidfRetriever(ngram_range=(3, 5), min_df=1)
    
    target_texts = [
        "microsoft corporation technology",
        "apple computer incorporation",
        "google alphabet headquarters",
        "amazon web services cloud",
    ]
    target_ids = ["T-MSFT", "T-AAPL", "T-GOOG", "T-AMZN"]
    
    retriever.fit_targets(target_texts, target_ids)
    
    # Query with a typo: 'amazn web servic'
    queries = ["amazn web servic", "appl computr"]
    q_ids = ["Q1", "Q2"]
    
    results = retriever.query_top_k(queries, q_ids, top_k=2, min_similarity=0.2)
    
    assert "Q1" in results
    assert len(results["Q1"]) >= 1
    top_cand_q1, score_q1 = results["Q1"][0]
    assert top_cand_q1 == "T-AMZN"
    assert score_q1 > 0.5
    
    assert "Q2" in results
    assert len(results["Q2"]) >= 1
    top_cand_q2, score_q2 = results["Q2"][0]
    assert top_cand_q2 == "T-AAPL"
    assert score_q2 > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
