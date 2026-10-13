# Final Approved Architecture Specification
## Entity Resolution System for Amazon ML Challenge 2026

**Document:** `reports/08_final_approved_architecture.md`  
**Stage:** Stage 2 Final Architecture (Post-Audit Approved Specification)  
**Date:** September 25, 2026  
**Status:** **APPROVED FOR IMPLEMENTATION PENDING STAGE 3 GATE CLEARANCE**  
**Classification of Tunable Parameters:** All unverified hyperparameters are explicitly tagged `TUNABLE — VALIDATE LOCALLY`.

---

## 1. Executive Summary & Design Principles

This document specifies the definitive, audited architecture for solving the Amazon ML Challenge 2026 Business Entity Resolution task. Following the adversarial audit conducted in `reports/07_architecture_audit.md`, this specification strips away all unverified assumptions, establishes leak-free validation boundaries, prunes redundant features, and introduces rigorous stopping criteria.

### Core Architectural Axioms
1. **The Official Metric is the Sole Optimization Authority:** All models, thresholds, and gating logic must directly optimize **Entity-Level Macro F0.5**, not pair-level AUC, pair-level F1, or pair-count distribution targets.
2. **Hard Country Partitioning:** 100% empirically proven across all 24.2M records and all 7,638,365 true pairs. The pipeline operates strictly within country partitions (`US`, `India`, `France`), eliminating 60% of unnecessary candidate evaluations with zero recall loss.
3. **Multi-Pass Complementary Blocking:** Combines ultra-fast deterministic key passes with sparse character n-gram TF-IDF retrieval to break the previous system's blocking recall bottleneck.
4. **Precision-First Model Governance:** F0.5 places twice the weight on precision compared to recall. Candidate generation and model inference enforce strict volume capping and hard negative exposure to prevent catastrophic singleton false merges.
5. **No Magic Numbers:** Parameters such as retrieval depth $K$, TF-IDF n-gram boundaries, block caps, and negative ratios are designated `TUNABLE — VALIDATE LOCALLY` and optimized empirically.

---

## 2. System Architecture Overview

```
                                  INPUT DATA (TSV)
                       Source 1 (Reference) | Source 2 | Source 3
                                          │
                                          ▼
                              DATA VALIDATION & INGESTION
                         Schema validation | 0 null countries verified
                                          │
                                          ▼
                             MULTI-VIEW NORMALIZATION
               Raw text | ASCII transliteration | Alphanumeric tokens |
               Token n-grams | Legal suffix extraction | Number extraction
                                          │
                                          ▼
                               COUNTRY PARTITIONING
                     [US Partition] | [India Partition] | [France Partition]
                         (Hard invariant: 0 cross-country matches)
                                          │
                                          ▼
                             TARGET INDEXING (S2 + S3)
                     Inverted token indexes | Sparse TF-IDF matrices
                                          │
                                          ▼
                              MULTI-PASS BLOCKING (UNION)
                 Pass 1: Exact normalized name match
                 Pass 2: Rare token match (IDF > threshold)
                 Pass 3: Sorted significant-token bigram
                 Pass 4: Name prefix (4 chars) + house number
                 Pass 5: TF-IDF character n-gram top-K (name)     [TUNABLE]
                 Pass 6: TF-IDF character n-gram top-K (address)  [TUNABLE]
                 Pass 7: First token + postal code
                 Pass 8: Address numerics + name prefix (3 chars)
                                          │
                                          ▼
                              CANDIDATE DEDUPLICATION &
                                VOLUME STOPPING RULE
                   Mean cands/S1 <= 80 | Max hard cap <= 500
                                          │
                                          ▼
                               PAIRWISE FEATURE PIPELINE
                                (36 Audited Features)
                       28 Core Features + 8 Experimental Features
                                          │
                                          ▼
                               HARD NEGATIVE MINING
                     (Dev-Train partition only: 5:1 neg:pos ratio)
                                          │
                                          ▼
                                LIGHTGBM CLASSIFIER
                        Objective: binary logloss (unweighted)
                                          │
                                          ▼
                             SCORE CALIBRATION (OPTIONAL)
                       Isotonic / Platt on Dev-Calibration partition
                           (Keep ONLY if Delta Macro F0.5 >= +0.003)
                                          │
                                          ▼
                             THRESHOLD OPTIMIZATION
                   Argmax Macro F0.5 sweep on Dev-Validation partition
                      Variable cardinality match sets + Singleton abstention
                                          │
                                          ▼
                             ONE-TIME CONFIRMATION
                                Locked-Holdout Set
                                          │
                                          ▼
                             FINAL SUBMISSION GENERATION
                      matching_results.tsv | candidate_pairs.tsv
```

---

## 3. Leakage-Proof Validation Architecture

### 3.1 Grouped Partitioning Specification
All data splits are grouped strictly by `source1_entity_id` using a deterministic random seed (`seed=42`). No S1 entity or its ground-truth matches may appear across multiple partitions.

```
Total S1 Entities: 2,206,821
├── 10.0% Locked-Holdout Partition (220,682 entities)
│     └── Sealed on disk; evaluated once prior to submission.
└── 90.0% Development Pool (1,986,139 entities)
      ├── 70.0% Dev-Train (1,390,297 entities)
      │     └── Source pool for model training & hard negative mining.
      │         (Fast iteration: stratified subsample of 200,000 S1s).
      ├── 10.0% Dev-Calibration (198,614 entities)
      │     └── Held out exclusively for fitting score calibrators.
      └── 20.0% Dev-Validation (397,228 entities)
            └── Benchmark for threshold optimization & feature ablations.
            └── Contains a standardized Fast-Eval Slice of 25,000 entities
                for rapid 3-minute iterative feedback loops.
```

### 3.2 Anti-Leakage Guardrails
1. **Target Matrix Isolation:** `TfidfVectorizer` is fit exclusively on unlabelled target documents (S2 and S3). S1 entities are transformed out-of-fold as test queries.
2. **Negative Sampling Isolation:** Hard negatives for training are mined strictly from Dev-Train candidates. Dev-Validation and Locked-Holdout candidates are isolated.
3. **Threshold Selection Isolation:** Threshold $\tau^*$ is selected solely on Dev-Validation. It is never fit on Dev-Train or Locked-Holdout.
4. **Feature Independence:** No global label-dependent statistics (target encoding, label frequency) may be computed. All features are pure pairwise distance functions.

---

## 4. Multi-View Normalization Pipeline

For each record (across S1, S2, and S3), the normalization engine produces structured views cached to Parquet format:

| View Name | Transformation Description | Purpose |
| :--- | :--- | :--- |
| `name_raw` | Stripped whitespace, UTF-8 preserved. | Exact matching baseline. |
| `name_ascii` | `anyascii` transliteration, lowercased, accented diacritics stripped. | Standardized lexical matching. |
| `name_alnum` | Non-alphanumeric characters replaced with spaces, normalized whitespace. | Tokenization and n-gram extraction. |
| `name_tokens` | Space-delimited token list from `name_alnum`. | Inverted indexing and token set overlap. |
| `name_no_suffix` | Business name with legal entity terms stripped (`inc`, `llc`, `pvt`, `ltd`, `sarl`, `sas`). | Core company name comparison. |
| `legal_suffix` | Extracted legal suffix string (or `NONE`). | Explicit entity structure agreement. |
| `addr_alnum` | Lowercased, alphanumeric normalized address string. | Address similarity & TF-IDF indexing. |
| `addr_numerics` | Extracted sorted sequence of all digit tokens in address. | Physical number agreement (house, building, PIN). |
| `postal_code` | Extracted 5-digit (US) or 6-digit (India) postal code regex. | Geographic anchoring. |
| `house_number` | First numeric sequence in address string. | Exact physical premises anchor. |

---

## 5. Multi-Pass Blocking & Candidate Generation

### 5.1 Blocking Passes Specification

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MULTI-PASS BLOCKING SPECIFICATION                      │
├────┬─────────────────────────────┬──────────────────────────┬──────────────────────────┤
│ #  │ Blocking Mechanism          │ Blocking Key Structure   │ Status & Parameter Tuning│
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 1  │ Exact Normalized Name       │ `country\|name_alnum`    │ Approved (Zero Cost)     │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 2  │ Rare Token Index            │ `country\|rare_token`    │ `TUNABLE — VALIDATE`     │
│    │                             │ (Token corpus freq < 500)│ (Sweep frequency cutoff) │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 3  │ Sorted Significant Bigram   │ `country\|tok1_tok2`     │ Approved                 │
│    │                             │ (Alphabetically sorted)  │                          │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 4  │ Name Prefix + House Number  │ `country\|pfx4\|house_num│ Approved                 │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 5  │ TF-IDF Character Name Top-K │ Sparse cosine retrieval  │ `TUNABLE — VALIDATE`     │
│    │                             │ on `name_alnum` matrix   │ (Sweep K in [10, 150])   │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 6  │ TF-IDF Character Addr Top-K │ Sparse cosine retrieval  │ `TUNABLE — VALIDATE`     │
│    │                             │ on `addr_alnum` matrix   │ (Sweep K in [10, 100])   │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 7  │ Name Token + Postal Code    │ `country\|tok1\|postal`  │ Approved                 │
├────┼─────────────────────────────┼──────────────────────────┼──────────────────────────┤
│ 8  │ Addr Numerics + Name Prefix │ `country\|num\|pfx3`     │ Approved                 │
└────┴─────────────────────────────┴──────────────────────────┴──────────────────────────┘
```

### 5.2 TF-IDF Retrieval Hyperparameter Tuning Grid
All TF-IDF parameters must be validated locally on the Dev-Validation slice:
- **Analyzer:** Test `char_wb` vs `char`.
- **N-gram Range:** Test `(3, 5)` vs `(3, 6)` vs `(2, 4)`.
- **Max Features:** Test `None (min_df=3)` vs `500,000` to prevent rare n-gram truncation.
- **Sublinear TF:** Keep `sublinear_tf=True` (logarithmic scaling $\log(1 + \text{tf})$).
- **Retrieval Depth $K$:** Test $K \in \{10, 20, 30, 50, 75, 100, 150\}$.

### 5.3 Candidate Volume Stopping Rule
Candidate generation terminates when:
$$\text{Pair Recall} \ge 97.0\% \quad \text{AND} \quad \text{Mean Candidates/S1} \le 80 \quad \text{AND} \quad \text{P95 Candidates/S1} \le 250$$
A hard cap of **500 candidates per S1 entity** is enforced. Excess candidates are truncated by TF-IDF similarity score.

---

## 6. Audited Pairwise Feature Set (36 Features)

The 45 originally proposed features have been audited. Redundant and country-invariant features are removed, leaving **28 Core Features** and **8 Experimental Features**:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    36 AUDITED PAIRWISE FEATURES                                  │
├────┬─────────────────────────────┬──────────────┬────────────────────────────────────────────────┤
│ #  │ Feature Name                │ Group        │ Definition / Computation                       │
├────┼─────────────────────────────┼──────────────┼────────────────────────────────────────────────┤
│ 1  │ `name_exact_match`          │ Name (Core)  │ Binary: 1.0 if name_alnum1 == name_alnum2      │
│ 2  │ `name_ratio`                │ Name (Core)  │ RapidFuzz fuzz.ratio / 100                     │
│ 3  │ `name_token_sort_ratio`     │ Name (Core)  │ RapidFuzz fuzz.token_sort_ratio / 100          │
│ 4  │ `name_jaro_winkler`         │ Name (Core)  │ RapidFuzz distance.JaroWinkler.similarity      │
│ 5  │ `name_char_tfidf_cosine`    │ Name (Core)  │ Sparse dot product on char n-gram TF-IDF       │
│ 6  │ `name_word_tfidf_cosine`    │ Name (Core)  │ Sparse dot product on word TF-IDF              │
│ 7  │ `name_token_jaccard`        │ Name (Core)  │ Set intersection / union on name tokens        │
│ 8  │ `name_rare_token_overlap`   │ Name (Core)  │ Jaccard on distinctive tokens (freq < 500)     │
│ 9  │ `name_len_ratio`            │ Name (Core)  │ min(len1, len2) / max(len1, len2)              │
│ 10 │ `name_token_count_diff`     │ Name (Core)  │ abs(num_tokens1 - num_tokens2)                 │
│ 11 │ `name_no_suffix_ratio`      │ Name (Core)  │ fuzz.ratio on names stripped of legal suffixes │
│ 12 │ `name_suffix_match`         │ Name (Core)  │ Binary: suffix1 == suffix2 (both non-empty)    │
│ 13 │ `name_partial_ratio`        │ Name (Exp)   │ RapidFuzz fuzz.partial_ratio / 100             │
│ 14 │ `name_token_set_ratio`      │ Name (Exp)   │ RapidFuzz fuzz.token_set_ratio / 100           │
├────┼─────────────────────────────┼──────────────┼────────────────────────────────────────────────┤
│ 15 │ `addr_exact_match`          │ Addr (Core)  │ Binary: 1.0 if addr_alnum1 == addr_alnum2      │
│ 16 │ `addr_ratio`                │ Addr (Core)  │ RapidFuzz fuzz.ratio on address strings        │
│ 17 │ `addr_char_tfidf_cosine`    │ Addr (Core)  │ Sparse dot product on addr char n-gram TF-IDF  │
│ 18 │ `addr_token_jaccard`        │ Addr (Core)  │ Token Jaccard overlap on address words         │
│ 19 │ `addr_numeric_overlap`      │ Addr (Core)  │ Overlap coefficient on extracted digits        │
│ 20 │ `addr_house_number_eq`      │ Addr (Core)  │ Binary: house_num1 == house_num2               │
│ 21 │ `addr_postal_eq`            │ Addr (Core)  │ Binary: postal_code1 == postal_code2           │
│ 22 │ `addr_len_ratio`            │ Addr (Core)  │ min(addr_len1, addr_len2) / max(addr_len)      │
│ 23 │ `is_addr_missing`           │ Addr (Core)  │ Binary: 1.0 if candidate address is empty      │
│ 24 │ `addr_token_set_ratio`      │ Addr (Exp)   │ RapidFuzz fuzz.token_set_ratio on address      │
│ 25 │ `addr_postal_prefix`        │ Addr (Exp)   │ Binary: postal1[:3] == postal2[:3]             │
├────┼─────────────────────────────┼──────────────┼────────────────────────────────────────────────┤
│ 26 │ `is_source2`                │ Context(Core)│ Binary: 1.0 if candidate record is from S2     │
│ 27 │ `is_source3`                │ Context(Core)│ Binary: 1.0 if candidate record is from S3     │
│ 28 │ `name_len`                  │ Context(Core)│ Character length of S1 entity name             │
│ 29 │ `addr_len`                  │ Context(Core)│ Character length of S1 address string          │
│ 30 │ `name_token_count`          │ Context(Core)│ Number of whitespace tokens in S1 name         │
│ 31 │ `name_has_legal_suffix`     │ Context(Core)│ Binary: 1.0 if S1 entity has legal suffix      │
│ 32 │ `candidate_retrieval_score` │ Context(Core)│ Max retrieval score emitted during blocking    │
├────┼─────────────────────────────┼──────────────┼────────────────────────────────────────────────┤
│ 33 │ `name_x_addr`               │ Cross (Core) │ name_char_tfidf_cosine * addr_char_tfidf_cosine│
│ 34 │ `high_name_exact_postal`    │ Cross (Exp)  │ Binary: name_ratio > 0.85 AND postal_eq == 1.0 │
│ 35 │ `high_name_exact_house`     │ Cross (Exp)  │ Binary: name_ratio > 0.85 AND house_eq == 1.0  │
│ 36 │ `rare_name_rare_addr`       │ Cross (Exp)  │ Binary: rare_token_overlap > 0 AND addr_jacc>0 │
└────┴─────────────────────────────┴──────────────┴────────────────────────────────────────────────┘
```

---

## 7. Hard Negative Mining Specification

To ensure high model precision without data drift, training pairs are generated using **Strategy C: Structured Mixed Mining** with dynamic sampling:

1. **Positive Pairs:** All ground truth matches $(S_1, \text{Target})$ for Dev-Train entities.
2. **Negative Pairs (Ratio: 5 negatives per 1 positive):**
   - **70% Blocking Negatives:** Candidates retrieved by Passes 1–8 that are not ground truth matches.
   - **20% High-Lexical Negatives:** Non-match candidates with `name_char_tfidf_cosine >= 0.50` (near-miss distractors).
   - **10% Controlled Random Negatives:** Randomly sampled targets within the same country partition.
3. **Singleton Negative Injection:** For true singletons in Dev-Train, the top-5 highest-scoring blocking candidates are injected as negative examples to train model abstention.

---

## 8. Classification Engine & Training Specification

### 8.1 Primary Model: LightGBM
- **Objective:** `binary` (binary logloss)
- **Evaluation Metric:** `auc` (for early stopping on Dev-Validation)
- **Boosting Type:** `gbdt`
- **Hyperparameter Configuration:**
  ```python
  params = {
      'objective': 'binary',
      'metric': 'auc',
      'boosting_type': 'gbdt',
      'num_leaves': 127,
      'max_depth': -1,
      'learning_rate': 0.05,
      'n_estimators': 800,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
      'min_child_samples': 50,
      'reg_alpha': 0.1,
      'reg_lambda': 1.0,
      'scale_pos_weight': 1.0,  # CRITICAL: Keep 1.0 to protect precision
      'n_jobs': -1,
      'random_state': 42,
      'verbose': -1,
  }
  ```

### 8.2 Baseline Benchmark: XGBoost
- Trained on identical Dev-Train pairs with `tree_method='hist'`, `max_depth=7`, `learning_rate=0.05`.
- Selected only if validation Macro F0.5 exceeds LightGBM.

### 8.3 Calibration Gate
- Raw LightGBM scores are benchmarked against **Isotonic Regression** and **Platt Scaling** fitted on the Dev-Calibration partition.
- **Retention Rule:** Calibration is deployed in production **ONLY if it improves Dev-Validation Macro F0.5 by at least +0.003**. Otherwise, raw model scores are passed directly to thresholding.

---

## 9. Entity-Level Threshold Optimization & Decision Engine

### 9.1 Evaluation Algorithm
Threshold optimization operates directly on the competition metric:

```python
def optimize_decision_threshold(
    val_candidate_scores, val_ground_truth, tau_grid=np.arange(0.15, 0.92, 0.01)
):
  """Evaluates variable cardinality match sets on ENTITY-LEVEL Macro F0.5."""
  best_tau = 0.50
  best_macro_f05 = 0.0

  for tau in tau_grid:
    pred_dict = {}
    for s1_id, candidate_pairs in val_candidate_scores.items():
      # candidate_pairs: list of (target_id, score)
      matched_targets = {
          target_id for target_id, score in candidate_pairs if score >= tau
      }
      pred_dict[s1_id] = matched_targets  # Empty set [] if none pass tau

    # Calculate exact competition Macro F0.5
    macro_f05 = calculate_exact_competition_macro_f05(
        val_ground_truth, pred_dict
    )

    if macro_f05 > best_macro_f05:
      best_macro_f05 = macro_f05
      best_tau = tau

  return best_tau, best_macro_f05
```

### 9.2 Singleton Abstention Rule
- If all candidate pairs for an S1 entity score $< \tau^*$, emit an empty set `[]`.
- If `prediction == []` and `ground_truth == []`, the entity score is **1.0**.
- Optional **Margin Gating** (`score_top1 - score_top2 < delta`) will be tested as an ablation.

---

## 10. France Domain-Shift & Stress-Test Protocol

### 10.1 Ground Truth Reality
Because France ground-truth labels do not exist in training data, the France threshold $\tau^*_{\text{France}}$ cannot be directly validated. It is explicitly categorized as an **unsupervised stress-tested heuristic**.

### 10.2 France Strategy
1. **Language-Agnostic Baseline:** Rely primarily on sublinear character n-gram TF-IDF and word TF-IDF, which down-weight common French function words (`de`, `la`, `du`) naturally without language-specific rules.
2. **Stress-Test Subsets:** Benchmark model stability across three France-like proxy slices:
   - **F1 (Generic Business Names):** Entities composed of high-frequency words.
   - **F2 (Legal-Form Heavy):** Entities with corporate acronyms (`SARL`, `SAS`).
   - **F3 (Address Ambiguity):** Dense multi-tenant addresses with generic street terms.
3. **Threshold Setting:** Default $\tau^*_{\text{France}} = \tau^*_{\text{global}}$. Adjust only if stress subsets show severe false-positive inflation requiring conservative threshold elevation.

---

## 11. Engineering Implementation Sequence (Stage 3 & Beyond)

Upon receiving user approval, implementation proceeds in strict sequential order:

```
STAGE 3.1: EXACT COMPETITION EVALUATOR (MANDATORY FIRST TASK)
           Implement `src/entity_resolution/metrics.py`
           Build unit tests verifying exact match, partial match, empty singletons,
           and false singleton penalties against manual worked examples.
                                │
                                ▼
STAGE 3.2: LEAKAGE-FREE VALIDATION HARNESS & DATA PROFILING
           Implement `src/entity_resolution/validation.py`
           Generate and seal the 4 partitions on disk (Parquet).
           Profile data completeness, character distributions, and entity counts.
                                │
                                ▼
STAGE 3.3: MULTI-VIEW NORMALIZATION PIPELINE
           Implement `src/entity_resolution/normalization.py`
           Transliteration (`anyascii`), cleaning, suffix extraction.
           Cache normalized records to disk.
                                │
                                ▼
STAGE 3.4: MULTI-PASS BLOCKING & RECALL AUDIT
           Implement `src/entity_resolution/blocking.py` and `retrieval.py`
           Build Inverted Index & Sparse TF-IDF Index.
           Execute K sweep and n-gram ablation on Dev-Validation slice.
           Verify stopping rule: Blocking Recall >= 97.0%, Mean Cands <= 80.
                                │
                                ▼
STAGE 3.5: FEATURE EXTRACTION PIPELINE
           Implement `src/entity_resolution/features.py`
           Compute 36 audited features in streaming chunks.
                                │
                                ▼
STAGE 3.6: HARD NEGATIVE MINING & MODEL TRAINING
           Implement `src/entity_resolution/model.py` and `negative_sampling.py`
           Train LightGBM on Dev-Train with 5:1 structured negative ratio.
           Train benchmark XGBoost model on identical features.
                                │
                                ▼
STAGE 3.7: CALIBRATION & THRESHOLD OPTIMIZATION
           Implement `src/entity_resolution/decision.py`
           Sweep entity-level threshold on Dev-Validation.
           Evaluate Isotonic/Platt calibration gate.
                                │
                                ▼
STAGE 3.8: FRANCE STRESS TESTING & ABLATIONS
           Evaluate on F1, F2, F3 stress subsets.
           Conduct feature importance and group ablations.
                                │
                                ▼
STAGE 3.9: ONE-TIME LOCKED VALIDATION CONFIRMATION
           Score Locked-Holdout partition once to confirm lack of overfitting.
                                │
                                ▼
STAGE 3.10: FULL TEST INFERENCE & OFFICIAL SUBMISSION VALIDATION
           Run full test set pipeline (`US`, `India`, `France`).
           Format `matching_results.tsv` and `candidate_pairs.tsv`.
           Verify with official `validate_submission.py`.
```

---

## 12. Implementation Gate Status Table

| Component | Architecture Decision | Status | Needs Local Experiment? |
| :--- | :--- | :--- | :--- |
| **Competition Evaluator** | Exact Entity-Level Macro F0.5 | **FINAL APPROVED** | **NO** (Must be built first) |
| **Validation Scheme** | 4-Way Grouped S1 Split (63/9/18/10) | **FINAL APPROVED** | **NO** (Locked foundation) |
| **Normalization** | Multi-View Transliteration (`anyascii`) | **FINAL APPROVED** | **YES** (Ablate French terms) |
| **Country Constraint** | Hard Partitioning (`US`, `India`, `France`)| **FINAL APPROVED (100% Empirically Verified)**| **NO** (Zero Mismatches in GT) |
| **Exact Blocking** | Pass 1: Normalized Name Hash | **FINAL APPROVED** | **YES** (Measure isolated recall) |
| **Rare-Token Blocking** | Pass 2: Inverted Index ($IDF > \theta$) | **FINAL APPROVED** | **YES** (Tune frequency cutoff) |
| **TF-IDF Name Blocking** | Pass 5: Character N-Gram Top-K | **FINAL APPROVED** | **YES** (Ablate n-grams & K) |
| **TF-IDF Addr Blocking** | Pass 6: Character N-Gram Top-K | **FINAL APPROVED** | **YES** (Ablate n-grams & K) |
| **TF-IDF K Depth** | $K_{\text{name}} \in [10, 150]$, $K_{\text{addr}} \in [10, 100]$| **TUNABLE — VALIDATE LOCALLY** | **YES** (Multi-metric sweep) |
| **Candidate Cap** | Mean $\le 80$, Hard Cap $\le 500$ | **TUNABLE — VALIDATE LOCALLY** | **YES** (Validate stopping rule) |
| **Pairwise Features** | 36 Features (28 Core + 8 Experimental) | **FINAL APPROVED** | **YES** (Permutation importance) |
| **Hard Negatives** | Strategy C: Mixed Structured Mining | **FINAL APPROVED** | **YES** (Compare A, B, C, D) |
| **Primary Model** | LightGBM (`binary logloss`, unweighted) | **FINAL APPROVED** | **YES** (Tune hyperparameters) |
| **Baseline Model** | XGBoost (`hist`, `max_depth=7`) | **FINAL APPROVED** | **YES** (Direct benchmark) |
| **Score Calibration** | Isotonic / Platt Regression | **CONDITIONAL** | **YES** (Keep only if $\Delta F0.5 \ge +0.003$) |
| **Decision Threshold** | Argmax Macro F0.5 on Val S1 | **FINAL APPROVED** | **YES** (Sweep 0.15 to 0.92) |
| **Singleton Gating** | Threshold Abstention (empty set = 1.0) | **FINAL APPROVED** | **YES** (Ablate margin gating) |
| **France Stress Test** | F1 (Generic), F2 (Legal), F3 (Address) | **FINAL APPROVED** | **YES** (Evaluate proxy slices) |
| **Dense Embeddings** | Sentence-Transformers / ANN Retrieval | **SHELVED** | **Conditional** (Only if blk recall < 97%) |
| **Neural Reranker** | Cross-Encoder Transformer Reranking | **SHELVED** | **Conditional** (Only if semantic errors) |

---

*End of Final Approved Architecture (`reports/08_final_approved_architecture.md`)*
