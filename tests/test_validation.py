"""
Unit Tests for Validation Architecture & Partitioning
Amazon ML Challenge 2026
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.entity_resolution.validation import (
    SplitRegistry,
    generate_stratified_splits,
)


@pytest.fixture
def synthetic_data():
    """Generates synthetic S1 ground-truth and country mappings."""
    n_entities = 1000
    rng = np.random.default_rng(123)
    
    s1_ids = [f"S1-{i:05d}" for i in range(n_entities)]
    countries = {s1: ("US" if rng.random() < 0.6 else "India") for s1 in s1_ids}
    
    # ~5.6% singletons
    matches = []
    for s1 in s1_ids:
        if rng.random() < 0.056:
            matches.append("")
        else:
            n_m = rng.integers(1, 6)
            m_list = [f"S2-{rng.integers(1000, 9999)}" for _ in range(n_m)]
            matches.append(",".join(m_list))
            
    df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "matched_entity_ids": matches,
    })
    return df, countries, set(s1_ids)


def test_stratified_splits_properties(synthetic_data):
    df, countries, all_ids = synthetic_data
    registry = generate_stratified_splits(df, countries, seed=42, fast_eval_size=50)
    
    # Check total counts (within stratum floor rounding tolerance)
    total = len(all_ids)
    assert abs(len(registry.locked_ids) - int(total * 0.10)) <= 4
    assert abs(len(registry.dev_train_ids) - int(total * 0.63)) <= 4
    assert abs(len(registry.dev_calib_ids) - int(total * 0.09)) <= 4
    assert len(registry.dev_val_ids) == total - len(registry.locked_ids) - len(registry.dev_train_ids) - len(registry.dev_calib_ids)
    
    # Zero leakage checks
    assert registry.dev_train_ids.isdisjoint(registry.dev_calib_ids)
    assert registry.dev_train_ids.isdisjoint(registry.dev_val_ids)
    assert registry.dev_train_ids.isdisjoint(registry.locked_ids)
    assert registry.dev_calib_ids.isdisjoint(registry.dev_val_ids)
    assert registry.dev_calib_ids.isdisjoint(registry.locked_ids)
    assert registry.dev_val_ids.isdisjoint(registry.locked_ids)
    
    # Full coverage
    union = registry.dev_train_ids | registry.dev_calib_ids | registry.dev_val_ids | registry.locked_ids
    assert union == all_ids
    
    # Fast eval properties
    assert registry.fast_eval_ids.issubset(registry.dev_val_ids)
    assert registry.fast_eval_ids.isdisjoint(registry.dev_train_ids)
    assert len(registry.fast_eval_ids) <= 50


def test_save_and_load_roundtrip(synthetic_data):
    df, countries, all_ids = synthetic_data
    registry = generate_stratified_splits(df, countries, seed=42, fast_eval_size=50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        registry.save(tmpdir)
        loaded = SplitRegistry.load(tmpdir)
        
        assert loaded.dev_train_ids == registry.dev_train_ids
        assert loaded.dev_calib_ids == registry.dev_calib_ids
        assert loaded.dev_val_ids == registry.dev_val_ids
        assert loaded.fast_eval_ids == registry.fast_eval_ids
        assert loaded.locked_ids == registry.locked_ids


def test_integrity_failure_on_leakage():
    s1 = {"S1-1", "S1-2"}
    s2 = {"S1-2", "S1-3"}  # Overlap on S1-2
    s3 = {"S1-4"}
    locked = {"S1-5"}
    fast = {"S1-4"}
    
    reg = SplitRegistry(
        dev_train_ids=s1,
        dev_calib_ids=s2,
        dev_val_ids=s3,
        fast_eval_ids=fast,
        locked_ids=locked,
    )
    with pytest.raises(AssertionError, match="Leakage"):
        reg.validate_integrity()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
