"""
Deterministic Multi-Pass Blocking Engine
Amazon ML Challenge 2026 - Business Entity Resolution

Generates complementary deterministic blocking passes:
- Pass 1: Exact normalized name match
- Pass 2: Rare token index (corpus frequency between min_freq and max_freq)
- Pass 3: Sorted significant-token bigrams
- Pass 4: Name prefix (4 chars) + house number
- Pass 7: Name primary token + postal code
- Pass 8: Address numerics + name prefix (3 chars)

All passes operate strictly within country partitions.
"""

from typing import Dict, List, Set, Tuple, Optional, Iterable, Iterator
from collections import defaultdict, Counter
import itertools


def generate_blocking_keys(
    country: str,
    name_alnum: str,
    name_no_suffix: str,
    house_number: str,
    postal_code: str,
    rare_tokens: Optional[Set[str]] = None,
) -> List[str]:
    """
    Generates all deterministic blocking keys for a single record.
    Keys are prefixed with country to enforce hard country partitioning.
    """
    keys: List[str] = []
    c = country.strip().upper()
    if not c or not name_alnum:
        return keys
        
    # Pass 1: Exact normalized alphanumeric name (High Precision)
    keys.append(f"{c}|N1:{name_alnum}")
    
    tokens = name_no_suffix.split() if name_no_suffix else name_alnum.split()
    n_tokens = len(tokens)
    
    # Pass 4: Name prefix (4 chars) + house number (Ultra-high recall, ~5 cands)
    if house_number and len(name_alnum) >= 4:
        pfx4 = name_alnum[:4]
        keys.append(f"{c}|NH:{pfx4}|{house_number}")
        
    # Pass 8: Address house number + name prefix (3 chars) (Ultra-high recall, ~7 cands)
    if house_number and len(name_alnum) >= 3:
        pfx3 = name_alnum[:3]
        keys.append(f"{c}|AN:{house_number}|{pfx3}")
        
    # Pass 7: Name primary token + postal code
    if postal_code and tokens:
        keys.append(f"{c}|NP:{tokens[0]}|{postal_code}")
        
    # Pass 3: Sorted significant-token bigrams (first 4 tokens to avoid explosion)
    if n_tokens >= 2:
        top_tokens = tokens[:4]
        for t1, t2 in itertools.combinations(top_tokens, 2):
            if t1 != t2:
                b_min, b_max = (t1, t2) if t1 < t2 else (t2, t1)
                keys.append(f"{c}|BG:{b_min}_{b_max}")
                
    # Pass 2: Rare token keys (Back-fill)
    if rare_tokens and tokens:
        for tok in tokens:
            if tok in rare_tokens:
                keys.append(f"{c}|RT:{tok}")
                
    return keys


def find_rare_tokens(
    token_sequences: Iterable[List[str]],
    min_freq: int = 5,
    max_freq: int = 500,
) -> Set[str]:
    """
    Identifies rare, highly discriminative tokens within the target corpus.
    Tokens appearing < min_freq are excluded (typos/noise);
    Tokens appearing > max_freq are excluded (generic terms).
    """
    counts: Counter = Counter()
    for seq in token_sequences:
        # Count unique tokens per document to measure document frequency
        counts.update(set(seq))
        
    rare = {tok for tok, freq in counts.items() if min_freq <= freq <= max_freq}
    return rare


class InvertedIndexBlocker:
    """
    High-performance in-memory inverted index for deterministic multi-pass blocking.
    """
    def __init__(self, bucket_cap: int = 500):
        self.bucket_cap = bucket_cap
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.total_records = 0
        
    def add_record(
        self,
        entity_id: str,
        country: str,
        name_alnum: str,
        name_no_suffix: str,
        house_number: str,
        postal_code: str,
        rare_tokens: Optional[Set[str]] = None,
    ) -> None:
        """Indexes keys for a target record."""
        keys = generate_blocking_keys(
            country=country,
            name_alnum=name_alnum,
            name_no_suffix=name_no_suffix,
            house_number=house_number,
            postal_code=postal_code,
            rare_tokens=rare_tokens,
        )
        for k in keys:
            bucket = self.index[k]
            if len(bucket) < self.bucket_cap:
                bucket.append(entity_id)
        self.total_records += 1

    def retrieve_candidates(
        self,
        country: str,
        name_alnum: str,
        name_no_suffix: str,
        house_number: str,
        postal_code: str,
        rare_tokens: Optional[Set[str]] = None,
        max_candidates: int = 150,
    ) -> Set[str]:
        """Retrieves union of candidate IDs for a Source-1 query record."""
        keys = generate_blocking_keys(
            country=country,
            name_alnum=name_alnum,
            name_no_suffix=name_no_suffix,
            house_number=house_number,
            postal_code=postal_code,
            rare_tokens=rare_tokens,
        )
        ordered_candidates: Dict[str, None] = {}
        for k in keys:
            if k in self.index:
                bucket = self.index[k]
                for cid in bucket:
                    ordered_candidates[cid] = None
                    if len(ordered_candidates) >= max_candidates:
                        break
            if len(ordered_candidates) >= max_candidates:
                break
        return set(ordered_candidates.keys())
