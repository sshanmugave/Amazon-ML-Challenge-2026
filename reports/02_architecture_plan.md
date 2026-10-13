# Architecture Plan — Amazon ML Challenge 2026
## Business Entity Resolution: Production-Grade System Design

**Stage 2 Output | Version 1.0 | September 2026**

---

## 1. System Architecture

### 1.1 High-Level Pipeline

```
                         RAW DATA (TSV)
                              │
                              ▼
                    ┌─────────────────────┐
                    │   DATA VALIDATION   │
                    │  Schema check       │
                    │  Encoding check     │
                    │  Missing value audit│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │ MULTI-VIEW          │
                    │ NORMALIZATION       │
                    │                     │
                    │ ├─ raw text         │
                    │ ├─ ascii (anyascii) │
                    │ ├─ alnum only       │
                    │ ├─ tokens           │
                    │ ├─ tokens no suffix │
                    │ ├─ legal suffix     │
                    │ ├─ name phonetic    │
                    │ ├─ address tokens   │
                    │ ├─ numeric tokens   │
                    │ ├─ postal code      │
                    │ └─ house number     │
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  COUNTRY PARTITION  │
                    │  (hard constraint)  │
                    │  US | India | France│
                    └─────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  INDEX BUILDING     │
                    │                     │
                    │ ├─ Inverted indexes │
                    │ ├─ TF-IDF matrices  │
                    │ │  (name char ngram)│
                    │ │  (name word)      │
                    │ │  (addr char ngram)│
                    │ └─ Cached to disk   │
                    └─────────┬───────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   MULTI-PASS BLOCKING        │
               │                              │
               │ Pass 1: Exact name match     │
               │ Pass 2: Rare name token      │
               │ Pass 3: Sorted name bigram   │
               │ Pass 4: Name prefix + house# │
               │ Pass 5: TF-IDF name (top-K)  │◄── NEW CRITICAL
               │ Pass 6: TF-IDF addr (top-K)  │◄── NEW CRITICAL
               │ Pass 7: Name + postal code   │
               │ Pass 8: Addr num + name pfx  │
               │                              │
               │         ALL UNIONED          │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   BLOCKING RECALL AUDIT      │
               │                              │
               │ Target: ≥97% pair recall     │
               │ Track: entity recall          │
               │ Track: candidate volume       │
               │                              │
               │ IF <97%: STOP, FIX BLOCKING  │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   PAIRWISE FEATURES          │
               │   (~45 features)             │
               │                              │
               │ Name features (16)           │
               │ Address features (12)        │
               │ Structural features (10)     │
               │ Cross features (5-7)         │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   HARD NEGATIVE MINING       │
               │                              │
               │ Blocking negatives (50%)     │
               │ High-similarity negatives    │
               │ Same-postal negatives        │
               │ Random negatives (remaining) │
               │ Ratio: 5:1 neg:pos           │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   LightGBM MATCHER           │
               │                              │
               │ Objective: binary logloss    │
               │ eval_metric: AUC             │
               │ Trees: 500-1000              │
               │ Max depth: 7-8               │
               │ GroupKFold validation         │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   OPTIONAL CALIBRATION       │
               │                              │
               │ Isotonic regression           │
               │ On held-out calibration set  │
               │ Keep ONLY if improves F0.5   │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   THRESHOLD OPTIMIZATION     │
               │                              │
               │ Score all validation pairs   │
               │ Sweep 0.01 to 0.99           │
               │ Generate entity-level preds  │
               │ Calculate exact macro F0.5   │
               │ Choose argmax threshold      │
               │                              │
               │ Optional: per-country sweep  │
               └──────────────┬───────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   ENTITY-LEVEL DECISION      │
               │                              │
               │ Score > threshold → MATCH    │
               │ No candidate > thresh → []   │
               │ Variable cardinality output  │
               │                              │
               │ Optional: margin gating      │
               │ (top-1 - top-2 gap analysis) │
               └──────────────┬───────────────┘
                              │
                    ┌─────────┴──────────┐
                    │                    │
                 MATCHES              ABSTAIN
                    │                    │
                    └─────────┬──────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   OUTPUT GENERATION          │
               │                              │
               │ matching_results.tsv         │
               │ candidate_pairs.tsv          │
               │                              │
               │ Official validator check     │
               └──────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Training Data Flow

```
Train S1 (2.2M) ──┐
Train S2 (5.0M) ──┤──► Normalization ──► Index Building ──► Blocking ──► Feature Gen ──► Model Training
Train S3 (5.3M) ──┘                                                                         │
Train GT (2.2M) ─────────────────────────────────────────────────────────────────────► Label Assignment
                                                                                             │
                                                                                     ┌───────┴───────┐
                                                                                     │               │
                                                                             Dev Validation    Locked Validation
                                                                              (90% entities)    (10% entities)
```

### 2.2 Test Inference Flow

```
Test S1 (1.7M) ──┐
Test S2 (4.9M) ──┤──► Normalization ──► Index Building ──► Blocking ──► Feature Gen ──► Model Predict ──► Threshold ──► Output
Test S3 (5.1M) ──┘
```

### 2.3 Intermediate Artifacts

All intermediate results should be saved to disk as Parquet or pickle files:

| Artifact | Format | Size Estimate | Purpose |
|----------|--------|---------------|---------|
| Normalized records | Parquet | ~2-3 GB | Avoid re-normalization |
| TF-IDF matrices | Scipy sparse (npz) | ~1-2 GB per matrix | Reusable for blocking + features |
| Inverted indexes | Pickle/Dict | ~500 MB | Fast candidate lookup |
| Candidate pairs | Parquet | ~1-5 GB | Input to feature generation |
| Feature matrix | Numpy/Parquet | ~2-4 GB | Input to model |
| Model | LightGBM file | ~10-50 MB | Trained model |
| Scores | Numpy array | ~100 MB | Model output for threshold sweep |

---

## 3. Validation Architecture

### 3.1 Validation Strategy

```
All 2,206,821 training S1 entities
              │
              ▼
      Random shuffle (seed=42)
              │
       ┌──────┴──────┐
       │              │
  Dev Set (90%)   Locked Set (10%)
  ~1,986,139       ~220,682
       │              │
       │              │ NEVER TOUCH until
       │              │ final confirmation
       │              │
  Used for:       Used for:
  - Blocking      - Final score
    recall eval     confirmation
  - Feature       - Leaderboard
    development     correlation
  - Model         - One-time
    training        validation
  - Threshold
    optimization
  - Error
    analysis
```

### 3.2 Development Validation (within Dev Set)

For rapid iteration, use a subsample of the Dev Set:

- **Quick validation:** 50,000 entities — fast iteration, ~15 min per experiment
- **Full dev validation:** ~200,000 entities — thorough evaluation, ~1-2 hours
- **Complete dev validation:** All ~1.98M entities — pre-submission, ~6-12 hours

### 3.3 Validation Split by Source1 Entity

```python
from sklearn.model_selection import GroupKFold

# Split entities, not pairs
all_s1_ids = sorted(ground_truth['source1_entity_id'].unique())
np.random.seed(42)
np.random.shuffle(all_s1_ids)

n_locked = int(len(all_s1_ids) * 0.10)
locked_s1 = set(all_s1_ids[:n_locked])
dev_s1 = set(all_s1_ids[n_locked:])

# Within dev, use train/val split
dev_s1_list = sorted(dev_s1)
n_val = int(len(dev_s1_list) * 0.15)
val_s1 = set(dev_s1_list[:n_val])
train_s1 = set(dev_s1_list[n_val:])
```

### 3.4 Evaluation Harness (MUST BUILD FIRST)

```python
def evaluate_macro_f05(ground_truth_dict, prediction_dict):
    """
    Exact competition metric.
    
    Args:
        ground_truth_dict: {s1_id: set(matched_ids)} for all S1 entities
        prediction_dict: {s1_id: set(predicted_ids)} for all S1 entities
    
    Returns:
        dict with: overall_f05, us_f05, india_f05, precision, recall,
                   singleton_fp_rate, entity_recall, per_entity_scores
    """
```

This evaluator must be **implemented and tested before any model is built**.

---

## 4. Blocking Plan

### 4.1 Blocking Passes (In Priority Order)

| Pass | Key Structure | Purpose | Expected Recall Contribution |
|------|--------------|---------|------------------------------|
| 1 | Exact normalized name | Catch clean duplicates | ~40-50% of true pairs |
| 2 | Rare name token (IDF > threshold) | Catch entities sharing distinctive words | ~15-20% additional |
| 3 | Sorted first-two significant tokens | Order-invariant bigram match | ~5-10% additional |
| 4 | Name prefix (4 chars) + house number | Address-anchored name match | ~3-5% additional |
| 5 | **TF-IDF char n-gram top-50** (name) | **Fuzzy name matching** | **~15-25% additional** |
| 6 | **TF-IDF char n-gram top-30** (address) | **Fuzzy address matching** | **~5-10% additional** |
| 7 | Name (first sig token) + postal code | Location-anchored name match | ~2-3% additional |
| 8 | Address numerics + name prefix | Physical co-location | ~1-2% additional |

### 4.2 TF-IDF Blocking Implementation

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Build TF-IDF vectors for name matching
name_tfidf = TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=(3, 5),
    max_features=500000,
    sublinear_tf=True,
    dtype=np.float32
)

# Fit on S2+S3 records (target corpus)
target_matrix = name_tfidf.fit_transform(target_names)  # sparse matrix

# Transform S1 records (query corpus)
query_matrix = name_tfidf.transform(s1_names)  # sparse matrix

# For each S1 entity, find top-K most similar targets
# Use batched sparse matrix multiplication for efficiency
for batch_start in range(0, n_s1, batch_size):
    batch_queries = query_matrix[batch_start:batch_start+batch_size]
    similarity = batch_queries.dot(target_matrix.T)  # sparse × sparse.T
    # Extract top-K from each row
```

### 4.3 Blocking Volume Control

Target metrics:
- **Mean candidates per S1:** 50-150
- **P95 candidates per S1:** <500
- **P99 candidates per S1:** <1000
- **Maximum candidates per S1:** <2000 (hard cap)
- **Total candidate pairs:** ~150M-300M

If candidate volume is too high, reduce top-K in TF-IDF passes. If blocking recall is too low, increase top-K or add more passes.

### 4.4 Block Size Capping

For deterministic passes (token blocking), cap individual block contributions:
- Single-token blocks: cap at 200 candidates
- If a block exceeds the cap, keep only the top-K by quick TF-IDF similarity

For TF-IDF passes, the top-K parameter naturally controls volume.

---

## 5. Feature Plan

### 5.1 Name Features (16 features)

| # | Feature | Computation | Signal |
|---|---------|-------------|--------|
| 1 | name_exact_match | n1 == n2 | Highest signal |
| 2 | name_ratio | fuzz.ratio(n1, n2) / 100 | Overall edit similarity |
| 3 | name_partial_ratio | fuzz.partial_ratio(n1, n2) / 100 | Substring containment |
| 4 | name_token_sort_ratio | fuzz.token_sort_ratio(n1, n2) / 100 | Order-invariant |
| 5 | name_token_set_ratio | fuzz.token_set_ratio(n1, n2) / 100 | Subset match |
| 6 | name_jaro_winkler | JaroWinkler.similarity(n1, n2) | Prefix-rewarding |
| 7 | name_levenshtein_norm | 1 - lev_dist / max(len1, len2) | Normalized edit distance |
| 8 | name_char_tfidf_cosine | cosine(tfidf_vec1, tfidf_vec2) | **IDF-weighted char similarity** |
| 9 | name_word_tfidf_cosine | cosine(word_tfidf1, word_tfidf2) | **IDF-weighted word similarity** |
| 10 | name_token_jaccard | |tokens1 ∩ tokens2| / |tokens1 ∪ tokens2| | Token overlap |
| 11 | name_rare_token_overlap | |rare1 ∩ rare2| / |rare1 ∪ rare2| | Discriminative token overlap |
| 12 | name_len_ratio | min(len1, len2) / max(len1, len2) | Length similarity |
| 13 | name_len_diff | abs(len1 - len2) | Length difference |
| 14 | name_token_count_diff | abs(|t1| - |t2|) | Token count difference |
| 15 | name_no_suffix_ratio | fuzz.ratio(ns1, ns2) / 100 | Name without legal suffix |
| 16 | name_suffix_match | suffix1 == suffix2 | Legal suffix agreement |

### 5.2 Address Features (12 features)

| # | Feature | Computation | Signal |
|---|---------|-------------|--------|
| 17 | addr_exact_match | a1 == a2 | Highest signal |
| 18 | addr_token_set_ratio | fuzz.token_set_ratio(a1, a2) / 100 | Token overlap |
| 19 | addr_ratio | fuzz.ratio(a1, a2) / 100 | Edit similarity |
| 20 | addr_jaro_winkler | JaroWinkler.similarity(a1, a2) | Prefix similarity |
| 21 | addr_char_tfidf_cosine | cosine(addr_tfidf1, addr_tfidf2) | **IDF-weighted similarity** |
| 22 | addr_token_jaccard | |t1 ∩ t2| / |t1 ∪ t2| | Token overlap |
| 23 | addr_numeric_overlap | |nums1 ∩ nums2| / |nums1 ∪ nums2| | Number agreement |
| 24 | addr_house_number_eq | house1 == house2 | Exact house number |
| 25 | addr_postal_eq | postal1 == postal2 | Exact postal code |
| 26 | addr_postal_prefix | postal1[:3] == postal2[:3] | Approximate postal |
| 27 | addr_len_ratio | min(len1, len2) / max(len1, len2) | Address length similarity |
| 28 | is_addr_missing | 1 if addr2 is empty | Missing data indicator |

### 5.3 Structural Features (10 features)

| # | Feature | Computation | Signal |
|---|---------|-------------|--------|
| 29 | country_india | 1 if India | Country context |
| 30 | country_us | 1 if US | Country context |
| 31 | country_france | 1 if France | Country context |
| 32 | is_source2 | 1 if candidate from S2 | Source context |
| 33 | is_source3 | 1 if candidate from S3 | Source context |
| 34 | name_len | len(n1) | Name complexity |
| 35 | addr_len | len(a1) | Address complexity |
| 36 | name_token_count | |tokens(n1)| | Name token count |
| 37 | name_has_legal_suffix | 1 if legal suffix present | Business structure |
| 38 | candidate_retrieval_score | TF-IDF similarity from blocking | Blocking confidence |

### 5.4 Cross Features (5-7 features)

| # | Feature | Computation | Signal |
|---|---------|-------------|--------|
| 39 | name_x_addr | name_tfidf_cos × addr_tfidf_cos | Combined confidence |
| 40 | high_name_exact_postal | name_ratio > 0.9 AND postal_eq | Strong combined signal |
| 41 | high_name_exact_house | name_ratio > 0.9 AND house_eq | Strong combined signal |
| 42 | high_addr_mod_name | addr_tfidf > 0.8 AND name_tfidf > 0.3 | Address-anchored match |
| 43 | rare_name_rare_addr | rare_name_overlap > 0 AND addr_jaccard > 0.5 | Discriminative combination |

**Total: ~43-45 features**

---

## 6. Negative Sampling Plan

### 6.1 Negative Types

| Type | Source | Proportion | Difficulty |
|------|--------|------------|------------|
| Blocking negatives | Retrieved by blocking, not in GT | 30% | Hard |
| High-similarity negatives | TF-IDF name sim > 0.5, not in GT | 20% | Very Hard |
| Same-postal negatives | Same postal code, different entity | 15% | Hard |
| Near-miss negatives | Top-K TF-IDF but low rank | 15% | Medium |
| Random within-country | Random from same country | 20% | Easy |

### 6.2 Sampling Ratio

- **Target ratio:** 5:1 (negatives:positives)
- **Per S1 entity:** All positive pairs + sampled negatives
- **Cap negative samples per S1:** 20 negatives maximum
- **Ensure diversity:** At least 2 different negative types per S1

### 6.3 Hard Negative Mining Pipeline

```
For each S1 entity in training:
    1. Retrieve all blocking candidates
    2. Identify positives (in GT)
    3. Score remaining candidates by TF-IDF name similarity
    4. Top-scored non-positives = hard negatives
    5. Sample from different difficulty tiers
    6. Add controlled random negatives
```

---

## 7. Model Plan

### 7.1 Primary Model: LightGBM

```python
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,        # leaf-wise growth
    'max_depth': -1,           # controlled by num_leaves
    'learning_rate': 0.05,
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}
```

### 7.2 Baseline Model: XGBoost

```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 600,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1
}
```

### 7.3 Model Comparison Protocol

1. Train both models on identical training data
2. Evaluate on identical validation data
3. Compare:
   - Pair-level AUC
   - **Macro F0.5 (at respective optimal thresholds)**
   - Runtime (training + inference)
4. Choose the model with higher measured F0.5

---

## 8. Calibration Plan

### 8.1 Protocol

1. Hold out 15% of dev training data as calibration set
2. Train model on remaining 85%
3. Score calibration set
4. Fit isotonic regression on (scores, labels) from calibration set
5. Compare threshold-optimized F0.5 with and without calibration
6. Keep calibration ONLY if F0.5 improves

### 8.2 Calibration Implementation

```python
from sklearn.isotonic import IsotonicRegression

calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(raw_scores_cal, labels_cal)
calibrated_scores = calibrator.predict(raw_scores_val)
```

---

## 9. Threshold Plan

### 9.1 Global Threshold Optimization

```python
best_threshold = 0.5
best_f05 = 0.0

for threshold in np.arange(0.01, 0.99, 0.01):
    predictions = {}
    for s1_id in val_s1_entities:
        matched = [cid for cid, score in candidates[s1_id] if score >= threshold]
        predictions[s1_id] = set(matched)
    
    f05 = calculate_macro_f05(ground_truth, predictions)
    
    if f05 > best_f05:
        best_f05 = f05
        best_threshold = threshold
```

### 9.2 Per-Country Threshold (Optional)

Only if validation shows >0.5% improvement:

```python
for country in ['US', 'India']:
    country_entities = val_entities_by_country[country]
    # Sweep threshold for this country's entities only
    # Choose threshold that maximizes country-specific F0.5
```

For France: use France-like stress subset threshold.

### 9.3 Margin-Based Gating (Optional)

Investigate whether adding a margin gate improves F0.5:

```python
for s1_id in val_entities:
    scores = sorted(candidate_scores[s1_id], reverse=True)
    if len(scores) >= 2:
        margin = scores[0] - scores[1]
        if margin < margin_threshold:
            # Uncertain — consider abstaining or being more conservative
```

Only implement if validation shows improvement.

---

## 10. Singleton Plan

### 10.1 Core Principle

A singleton prediction (empty match list) is correct when the S1 entity has no true matches. It scores 1.0. A false merge on a true singleton scores 0.0. Therefore, **false positives on singletons are maximally expensive**.

### 10.2 Implementation

```
IF no candidate scores above threshold:
    prediction = [] (empty / abstain)
    → Scores 1.0 if truly singleton, 0.0 if missed true matches

IF some candidates score above threshold:
    prediction = set(above-threshold candidates)
    → Scores F0.5 based on precision and recall
```

### 10.3 Singleton Rate Target

Training singleton rate: 5.58%
Expected test singleton rate: similar (5-6%)

If predicted singleton rate deviates significantly (e.g., <3% or >10%), investigate — likely threshold issue.

### 10.4 Singleton False Positive Analysis

After each experiment, report:
- Number of true singletons in validation
- Number of true singletons with false match predictions
- Singleton false positive rate
- Impact on overall F0.5

---

## 11. France Plan

### 11.1 Strategy: Generalization, Not Special-Casing

The model should generalize to France through:

1. **TF-IDF character n-gram features** — language-agnostic
2. **Word TF-IDF features** — automatically down-weight common French words
3. **Legal suffix normalization** — include SARL, SAS, SASU, EURL, SCI, SA
4. **Diacritic normalization** — anyascii handles French accents
5. **Street descriptor handling** — treat "Rue", "Boulevard" like "Street", "Road"

### 11.2 French Normalization Additions

```python
FRENCH_LEGAL_SUFFIXES = {
    'sarl', 'sas', 'sasu', 'eurl', 'sci', 'sa', 
    'societe', 'association', 'groupe', 'cie'
}

FRENCH_STREET_DESCRIPTORS = {
    'rue', 'boulevard', 'blvd', 'avenue', 'ave',
    'chemin', 'impasse', 'allee', 'place', 'passage',
    'route', 'cours', 'quai'
}
```

### 11.3 France Stress Test

Create a "France-like" validation subset from training data:

```python
france_like = []
for s1_id in dev_entities:
    name, addr, country = s1_data[s1_id]
    # Select entities with France-like characteristics:
    # - Short name (< 20 chars)
    # - Legal suffix present
    # - Dense address (multiple components)
    # - Low rare token density
    # - High ambiguity (name appears similar to many others)
    if is_france_like(name, addr):
        france_like.append(s1_id)
```

Evaluate model on this subset to estimate France robustness.

### 11.4 France Threshold

- Start with the global threshold
- If France-like validation shows significantly different optimal threshold, consider per-country
- **Never manually set France threshold without validation evidence**

---

## 12. Experiment Plan

### 12.1 Experiment Sequence

| ID | Experiment | Purpose | Success Criterion |
|----|-----------|---------|-------------------|
| E01 | Build evaluator + validation infrastructure | Foundation | Evaluator matches competition formula exactly |
| E02 | Data profiling + normalization | Understanding | Complete profile of data characteristics |
| E03 | Baseline blocking (deterministic only) | Blocking floor | Measure pair recall, candidate volume |
| E04 | + TF-IDF name blocking | Critical improvement | Blocking recall ≥95% |
| E05 | + TF-IDF address blocking | Further improvement | Blocking recall ≥97% |
| E06 | + Additional blocking passes | Completeness | Blocking recall ≥98% |
| E07 | Basic features (18, like previous) + LightGBM | Model baseline | F0.5 with proper threshold sweep |
| E08 | Extended features (45) + LightGBM | Feature impact | F0.5 improvement measured |
| E09 | Hard negative mining | Precision improvement | F0.5 improvement measured |
| E10 | XGBoost comparison | Model selection | Compare F0.5, choose winner |
| E11 | Calibration test | Score quality | F0.5 with/without calibration |
| E12 | Per-country thresholds | Threshold refinement | F0.5 improvement measured |
| E13 | France stress test | Domain shift | France-like F0.5 acceptable |
| E14 | Full dev validation | Scale test | Consistent with quick validation |
| E15 | Locked validation | Final confirmation | F0.5 stable, ready for submission |
| E16 | Ablation studies | Feature importance | Confirm each component contributes |

### 12.2 Experiment Tracking

```
experiments/experiments.csv

Columns:
experiment_id, timestamp, blocking_version, feature_version, model_type,
model_version, negative_sampling, threshold, margin_threshold,
val_n_entities, overall_f05, us_f05, india_f05,
blocking_pair_recall, blocking_entity_recall,
pair_precision, pair_recall, singleton_fp_rate,
avg_candidates_per_s1, total_candidates,
train_time_sec, inference_time_sec, peak_ram_gb,
notes
```

---

## 13. Resource / Runtime Plan

### 13.1 Estimated Resource Requirements

| Component | CPU Hours | RAM (Peak) | Disk |
|-----------|-----------|------------|------|
| Data loading + normalization | 0.5h | 8 GB | 5 GB (Parquet) |
| TF-IDF matrix building | 1-2h | 10-16 GB | 3 GB (sparse matrices) |
| Blocking (all passes) | 2-4h | 12-16 GB | 2-5 GB (candidates) |
| Feature generation | 3-6h | 8-12 GB | 4-8 GB |
| Model training | 0.5-1h | 4-8 GB | 50 MB |
| Validation | 0.5h | 4 GB | — |
| Test inference | 4-8h | 12-16 GB | 2 GB |
| **Total** | **~12-22h** | **16 GB peak** | **~20 GB** |

### 13.2 Memory Management Strategy

- **Country partitioning** — process US, India, France separately
- **Chunked TF-IDF multiplication** — process S1 entities in batches of 10K-50K
- **Streaming feature generation** — compute features per batch, write to disk
- **Sparse matrix operations** — all TF-IDF stored as scipy sparse
- **Garbage collection** — explicit gc.collect() between country partitions
- **Intermediate disk caching** — save normalized data, TF-IDF matrices, candidates to Parquet

### 13.3 Parallelism

- **TF-IDF fit/transform:** sklearn handles internally with n_jobs=-1
- **Feature computation:** Parallelize across S1 entities using multiprocessing or joblib
- **LightGBM/XGBoost:** Native multi-thread (n_jobs=-1)
- **Blocking:** Can be parallelized per S1 batch

---

## 14. Submission Plan

### 14.1 Leaderboard Strategy

Only 15 total submissions, 5/day, 3 days.

| Submission # | Purpose | Prerequisite | Expected Timing |
|-------------|---------|--------------|-----------------|
| **1** | Validated baseline | E07 complete, F0.5 measured | Day 1 evening |
| **2** | Extended features + hard negatives | E08-E09, ≥3% F0.5 improvement | Day 2 morning |
| **3** | Blocking + model improvements | E04-E06, blocking ≥97% | Day 2 afternoon |
| **4** | France + domain shift | E13, France-like validated | Day 2 evening |
| **5** | Final best model | Full validation, locked set confirmed | Day 3 |
| 6-15 | Reserve for validated improvements | Only if significant improvement measured | Day 3 |

### 14.2 Local ↔ Leaderboard Correlation

After each submission:
```
Submission #:
Local F0.5:
Leaderboard F0.5:
Delta (leaderboard - local):
Correlation notes:
```

If local and leaderboard diverge by >5%, investigate before further submissions.

---

## 15. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Blocking recall still <97% after TF-IDF | Low | High | Add more passes, increase top-K, investigate failures |
| Memory overflow on full dataset | Medium | High | Country partitioning, chunked processing, Parquet intermediates |
| France domain shift too severe | Medium | High | TF-IDF features generalize well; stress testing; conservative threshold |
| TF-IDF matrix too large for RAM | Medium | Medium | Reduce max_features, use sublinear_tf, chunk processing |
| LightGBM overfits on training data | Low | Medium | Regularization, early stopping, cross-validation |
| Threshold doesn't transfer to test | Medium | Medium | Multiple validation checks, locked set confirmation |
| Runtime exceeds 24 hours for test inference | Low-Medium | High | Optimize batch sizes, reduce candidate volume, profile hotspots |
| Feature computation is the bottleneck | Medium | Medium | Cache TF-IDF vectors, batch RapidFuzz calls, parallelize |
| Singleton false positives dominate errors | Medium | Medium | Margin gating, conservative threshold, error analysis |
| Local ↔ leaderboard correlation fails | Low | Very High | Investigate format, France handling, distribution shift |

---

## 16. Ablation Plan

### 16.1 Feature Group Ablations

| Ablation | What to Remove | Expected Impact |
|----------|---------------|-----------------|
| A1: No TF-IDF features | Remove name_char_tfidf, name_word_tfidf, addr_char_tfidf | Large drop (verify TF-IDF value) |
| A2: No address features | Remove all addr_* features | Moderate drop |
| A3: No cross features | Remove interaction features | Small-moderate drop |
| A4: No rare token features | Remove rare_token_overlap | Small drop |
| A5: No structural features | Remove source/country/length features | Variable |

### 16.2 Blocking Ablations

| Ablation | What to Remove | Expected Impact on Blocking Recall |
|----------|---------------|-----------------------------------|
| B1: No TF-IDF name blocking | Remove Pass 5 | Large recall drop |
| B2: No TF-IDF address blocking | Remove Pass 6 | Moderate recall drop |
| B3: Deterministic only | Remove all TF-IDF passes | Severe recall drop |
| B4: TF-IDF only | Remove deterministic passes | Moderate recall drop, higher volume |

### 16.3 Decision Ablations

| Ablation | What to Change | Measure |
|----------|----------------|---------|
| D1: Global vs per-country threshold | Compare single vs per-country | F0.5 difference |
| D2: With vs without margin gating | Add/remove margin constraint | Singleton FP rate |
| D3: Calibrated vs uncalibrated | Compare threshold-optimized F0.5 | F0.5 difference |

---

## 17. Code Architecture

```
e:\KCT\amzon ml hackthon\iteration 2\
│
├── reports/
│   ├── 01_research_report.md          ← COMPLETED
│   └── 02_architecture_plan.md        ← THIS DOCUMENT
│
├── configs/
│   ├── base.yaml                      # Shared configuration
│   ├── development.yaml               # Quick iteration settings
│   └── production.yaml                # Full-scale settings
│
├── src/
│   └── entity_resolution/
│       ├── __init__.py
│       ├── io.py                      # Data loading, saving, format handling
│       ├── schema.py                  # Column names, data types, constants
│       ├── normalization.py           # Text normalization, transliteration
│       ├── language.py                # Legal suffixes, stopwords, abbreviations
│       ├── indexing.py                # Inverted indexes, TF-IDF matrix building
│       ├── blocking.py                # Multi-pass blocking, union, deduplication
│       ├── retrieval.py               # TF-IDF top-K retrieval
│       ├── features.py                # Pairwise feature computation
│       ├── negative_sampling.py       # Hard negative mining
│       ├── model.py                   # LightGBM/XGBoost training, prediction
│       ├── calibration.py             # Score calibration
│       ├── decision.py                # Threshold sweep, entity-level decisions
│       ├── metrics.py                 # Macro F0.5 evaluator
│       ├── validation.py              # Validation splits, evaluation harness
│       ├── inference.py               # Full test inference pipeline
│       ├── submission.py              # Output formatting, validator
│       └── utils.py                   # Logging, memory tracking, profiling
│
├── scripts/
│   ├── 01_profile_dataset.py          # Data profiling and exploration
│   ├── 02_build_normalizations.py     # Normalize all records, save Parquet
│   ├── 03_build_indexes.py            # Build inverted + TF-IDF indexes
│   ├── 04_evaluate_blocking.py        # Measure blocking recall
│   ├── 05_build_candidates.py         # Generate all candidate pairs
│   ├── 06_build_features.py           # Compute features for all candidates
│   ├── 07_train.py                    # Train model with hard negatives
│   ├── 08_validate.py                 # Score validation, measure F0.5
│   ├── 09_optimize_threshold.py       # Threshold sweep
│   ├── 10_predict.py                  # Full test inference
│   ├── 11_validate_submission.py      # Run official validator
│   └── 12_error_analysis.py           # Post-hoc error analysis
│
├── experiments/
│   ├── experiments.csv                # Experiment tracking
│   └── logs/                          # Experiment logs
│
├── cache/                             # Intermediate cached artifacts
│   ├── train/                         # Training intermediates
│   └── test/                          # Test intermediates
│
├── output/
│   ├── matching_results.tsv           # Final submission
│   └── candidate_pairs.tsv            # Final candidates
│
├── tests/
│   ├── test_metrics.py                # Test evaluator correctness
│   ├── test_normalization.py          # Test normalization edge cases
│   └── test_features.py              # Test feature computation
│
├── requirements.txt                   # Pinned dependencies
│
└── README.md                          # Reproduction instructions
```

### 17.1 Key Dependencies

```
# requirements.txt
numpy>=1.24
pandas>=2.0
polars>=0.19              # Fast data processing
scikit-learn>=1.3         # TF-IDF, metrics, calibration
scipy>=1.11               # Sparse matrices
lightgbm>=4.0             # Primary model
xgboost>=2.0              # Baseline comparison model
rapidfuzz>=3.0            # Fast string similarity
anyascii>=0.3             # Transliteration
pyarrow>=13.0             # Parquet support
psutil>=5.9               # Memory monitoring
pyyaml>=6.0               # Configuration
tqdm>=4.65                # Progress bars
joblib>=1.3               # Parallelism
```

---

## 18. Implementation Sequence (Post-Approval)

```
Stage 3.1: Dataset audit + profiling
                │
Stage 3.2: Build exact competition evaluator + test it
                │
Stage 3.3: Implement normalization pipeline
                │
Stage 3.4: Build baseline blocking (deterministic)
                │
Stage 3.5: Measure baseline blocking recall
                │
Stage 3.6: Add TF-IDF blocking passes
                │
Stage 3.7: Measure improved blocking recall (target ≥97%)
                │
Stage 4.1: Build feature pipeline (18 basic features)
                │
Stage 4.2: Train baseline LightGBM
                │
Stage 4.3: Threshold sweep → measure F0.5
                │
Stage 4.4: FIRST EXPERIMENT REPORT
                │
Stage 5.1: Extend to 45 features
                │
Stage 5.2: Implement hard negative mining
                │
Stage 5.3: Retrain → measure F0.5
                │
Stage 5.4: SECOND EXPERIMENT REPORT
                │
Stage 6.1: France stress test
                │
Stage 6.2: Ablation studies
                │
Stage 6.3: THIRD EXPERIMENT REPORT
                │
Stage 7.1: Full dev validation
                │
Stage 7.2: Locked validation
                │
Stage 7.3: Test inference
                │
Stage 7.4: Official validator
                │
Stage 7.5: SUBMISSION
```

---

*End of Architecture Plan*
