# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Unit Tests](https://img.shields.io/badge/tests-28%2F28%20passing-brightgreen.svg)](tests/)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM-ff69b4.svg)](https://lightgbm.readthedocs.io/)
[![RapidFuzz C++](https://img.shields.io/badge/Kernels-RapidFuzz%20C%2B%2B-orange.svg)](https://github.com/maxbachmann/RapidFuzz)
[![US Macro F0.5](https://img.shields.io/badge/US%20Macro%20F0.5-0.8691-success.svg)](reports/06_final_system_summary.md)
[![Precision](https://img.shields.io/badge/Micro%20Precision->96.2%25-blueviolet.svg)](reports/06_final_system_summary.md)

An end-to-end, high-performance, leakage-free machine learning system for large-scale **Business Entity Resolution (Record Linkage)** developed for the **Amazon ML Challenge 2026**.

The system matches noisy reference queries (**Source 1**) against a heterogeneous target corpus (**Source 2** and **Source 3**) comprising **~24.2M records** across **United States (US)**, **India**, and **France**, optimizing directly for the official precision-weighted **Entity Macro $F_{0.5}$** metric with singleton false-merge penalization.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [System Architecture](#system-architecture)
- [Quantitative Performance Benchmarks](#quantitative-performance-benchmarks)
- [Repository Structure](#repository-structure)
- [Quickstart: Ready to Run](#quickstart-ready-to-run)
  - [1. Installation](#1-installation)
  - [2. Run Complete Demo Pipeline (5 seconds)](#2-run-complete-demo-pipeline-5-seconds)
  - [3. Run Complete Unit Test Suite](#3-run-complete-unit-test-suite)
- [End-to-End Pipeline Workflow](#end-to-end-pipeline-workflow)
- [Scientific & Engineering Insights](#scientific--engineering-insights)
- [Pre-Trained Models](#pre-trained-models)
- [Reports & Provenance](#reports--provenance)
- [License & Author](#license--author)

---

## Key Highlights

- **Pre-Trained Models Included**: Ready-to-run LightGBM classifiers (`models/lgbm_model_us.joblib`, `models/lgbm_model_india.joblib`, `models/lgbm_model_france.joblib`) calibrated at optimal threshold $\tau^* = 0.98$.
- **1-Command Demo**: Comes with sample datasets (`sample_data/`) and `demo_pipeline.py` demonstrating end-to-end normalization, blocking, feature extraction, scoring, evaluation, and TSV export in $< 3$ seconds.
- **100% Passing Test Suite**: 28 unit tests covering metrics, normalization, blocking, 36 pairwise features, and models.
- **Strict Leakage Prevention**: Stratified 4-way partition harness grouped by `source1_entity_id` (Dev-Train, Dev-Calibration, Dev-Validation, Locked-Holdout, Fast-Eval slice).
- **Hard Country Partition Invariant**: 100% mathematically proven 0 cross-country matches across 7.6M pairs, isolating target search spaces and achieving a $3\times$ speedup.
- **Ultra-Clean Blocking Backbone**: High-selectivity passes combining normalized alphanumeric prefixes and house numbers (`name[:4] + house_number`) capturing $>62\%$ true pairs with only 5–7 candidates per query.
- **C++ RapidFuzz Vectorization**: 36-dimensional feature extraction with zero-copy vectorized comparisons handling millions of pairs efficiently.

---

## System Architecture

```
Raw Multi-Source Data (Source 1, Source 2, Source 3)
   │
   ▼
[Stage 1] Multi-View Normalization Engine
   ├── AnyAscii Transliteration (Hindi, Marathi, Tamil, etc.)
   ├── French Diacritics Stripping (NFKD decomposition)
   ├── Legal Suffix Extraction (LLC, Inc, Pvt Ltd, SARL, SAS)
   └── Structured Address Parsing (House #, Postal Code, Numerics)
   │
   ▼
[Stage 2] Deterministic Multi-Pass Blocker & Candidate Generation
   ├── Country Partition Isolation (US, India, France)
   ├── Pass 1: Exact Normalized Name Match
   ├── Pass 4: Name Prefix (4 chars) + House Number
   ├── Pass 8: Address House Number + Name Prefix (3 chars)
   ├── Pass 7: Primary Name Token + Postal Code
   ├── Pass 3: Sorted Significant-Token Bigrams
   └── Pass 2: Corpus Rare Tokens (Frequency 5 - 500)
   │
   ▼
[Stage 3] 36 Audited Pairwise Feature Engine
   ├── RapidFuzz C++ String Metrics (ratio, token sort, jaro-winkler, partial)
   ├── Address Numeric, House Number & Postal Code Matches
   ├── Token Set Overlaps, Jaccard Similarities & Rare Token Intersections
   └── Non-Linear Cross Features (name x addr, high-name + exact postal)
   │
   ▼
[Stage 4] LightGBM Classifier & Calibrated Thresholding
   ├── Unweighted Binary Logloss (preserves calibrated true probabilities)
   ├── Structured Negative Mining (blocking negatives + lexical distractors)
   └── Exact Entity Macro F0.5 Optimization (Calibrated tau* = 0.98)
   │
   ▼
[Stage 5] Streaming Inference & Competition TSV Generation
   ├── Country-partitioned memory management (zero OOM risk)
   ├── Incremental checkpointing per batch
   └── Exact schema: source1_entity_id \t matched_entity_ids
```

---

## Quantitative Performance Benchmarks

Evaluated on the held-out Fast-Eval validation slice using the exact competition metric:

$$\text{Macro } F_{0.5} = \frac{1}{|E|} \sum_{e \in E} \frac{1.25 \cdot \text{Precision}(e) \cdot \text{Recall}(e)}{0.25 \cdot \text{Precision}(e) + \text{Recall}(e)}$$

| Cohort | Optimal Threshold $\tau^*$ | Peak Entity Macro $F_{0.5}$ | Micro Precision | Micro Recall | Singleton FP Rate | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **United States (US)** | **0.98** | **0.869128** | **96.19%** | **76.69%** | **14.42%** | **Passed** |
| **India** | **0.98** | **0.774461** | **96.23%** | **63.29%** | **11.02%** | **Passed** |
| **Combined Weighted** | **0.98** | **~0.825** | **>96.2%** | **~71.5%** | **<13.5%** | **Top Tier** |

> **Why $\tau^* = 0.98$?**  
> Under competition rules, singleton entities (queries with 0 true target matches) are penalized to **0.0** if even a single false match is predicted. Furthermore, $F_{0.5}$ weights precision twice as heavily as recall ($2\times$). Operating at $\tau^* = 0.98$ maintains $>96.2\%$ precision, shielding singletons from false merges while capturing high-confidence true matches.

---

## Repository Structure

```text
Amazon-ML-Challenge-2026/
├── demo_pipeline.py                 # 1-Command self-contained runnable demo
├── pyproject.toml                   # Package specifications & build config
├── requirements.txt                 # Exact dependencies
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
│
├── models/                          # Pre-trained production LightGBM models
│   ├── lgbm_model_us.joblib         # Calibrated US model (tau* = 0.98)
│   ├── lgbm_model_india.joblib      # Calibrated India model (tau* = 0.98)
│   └── lgbm_model_france.joblib     # Calibrated France model (tau* = 0.98)
│
├── sample_data/                     # Multi-country sample benchmark datasets
│   ├── sample_source1.tsv           # Sample query records (US, India, France)
│   ├── sample_source2.tsv           # Sample target records with noise
│   ├── sample_source3.tsv           # Sample target records with inversions
│   └── sample_ground_truth.tsv      # Benchmark ground truth with singletons
│
├── src/
│   └── entity_resolution/           # Core production Python package
│       ├── __init__.py
│       ├── normalization.py         # Multi-view text & address normalization
│       ├── blocking.py              # Inverted index multi-pass candidate blocker
│       ├── retrieval.py             # Sparse TF-IDF cosine candidate retriever
│       ├── features.py              # 36 audited pairwise feature extractors
│       ├── training_data.py         # Negative pair miner (blocking + lexical)
│       ├── model.py                 # LightGBM wrapper & threshold optimizer
│       ├── validation.py            # Leakage-free 4-way stratified partitioner
│       └── metrics.py               # Exact competition Macro F0.5 harness
│
├── scripts/                         # Pipeline orchestration scripts
│   ├── 01_build_validation_splits.py # Generate stratified leakage-free splits
│   ├── 02_profile_dataset.py        # Stream-profile schemas, lengths, & encodings
│   ├── 03_build_normalizations.py   # Multi-view text normalization to Parquet
│   ├── 04a_profile_blocking_passes.py # Per-pass ablation & recall profiling
│   ├── 04_evaluate_blocking.py      # End-to-end multi-pass candidate evaluation
│   ├── 05_train_model.py            # Model training & threshold calibration
│   └── 06_generate_test_submission.py # Streaming test set inference & TSV export
│
├── tests/                           # Pytest unit test suite (28/28 passing)
│   ├── test_metrics.py              # Mathematical validation of Macro F0.5
│   ├── test_normalization.py        # Indic, French, suffix, address tests
│   ├── test_blocking.py             # Inverted index & TF-IDF candidate tests
│   ├── test_features.py             # 36-dimensional feature vector verifications
│   ├── test_model.py                # Model training, serialization, & thresholding
│   └── test_validation.py           # Split disjointness & leakage prevention
│
├── submissions/                     # Submission outputs & samples
│   ├── dry_run_submission.tsv       # Benchmark submission format sample
│   └── demo_submission.tsv          # Generated demo submission output
│
└── reports/                         # Research notes & empirical audit logs
    ├── 01_research_report.md
    ├── 02_architecture_plan.md
    ├── 03_data_profile_report.md
    ├── 04_blocking_evaluation_report.md
    ├── 05_model_training_report.md
    ├── 06_final_system_summary.md
    ├── 07_architecture_audit.md
    ├── 08_final_approved_architecture.md
    └── 09_final_test_submission_report.md
```

---

## Quickstart: Ready to Run

### 1. Installation

Clone the repository and install the verified dependencies:

```bash
git clone https://github.com/sshanmugave/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026

# Create virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Or install as an editable package:
pip install -e .
```

### 2. Run Complete Demo Pipeline (5 seconds)

Run the self-contained demo pipeline on the bundled sample dataset using the pre-trained models:

```bash
python demo_pipeline.py
```

**Expected Output:**
```text
================================================================================
  AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION PIPELINE DEMO  
================================================================================

[Step 1/6] Loading Sample Datasets...
  Source 1 queries: 15 records
  Source 2 targets: 15 records
  Source 3 targets: 15 records
  Ground Truth:     15 entities

[Step 2/6] Running Multi-View Normalization Pipeline...
  Normalized 45 total records in 0.007s.

[Step 3/6] Indexing Targets & Generating Multi-Pass Candidate Pairs...
  Generated 24 candidate pairs across 15 queries in 0.000s.

[Step 4/6] Computing 36 Pairwise Features & Running Model Scoring...
  Loaded pre-trained US LightGBM model: models/lgbm_model_us.joblib
  Loaded pre-trained INDIA LightGBM model: models/lgbm_model_india.joblib
  Loaded pre-trained FRANCE LightGBM model: models/lgbm_model_france.joblib

[Step 5/6] Evaluating Predictions Against Competition Ground Truth...
==============================================================================
                   COMPETITION EVALUATION SUMMARY REPORT                   
==============================================================================
  Macro F0.5 (Official Leaderboard Metric) : 0.940741
------------------------------------------------------------------------------
  Total S1 Entities Evaluated              : 15
    - Matched Entities                     : 12 (80.00%)
    - Singleton Entities                   : 3 (20.00%)
    - Singleton False Merges (FP)          : 0 (0.00%)
------------------------------------------------------------------------------
  Pair-Level Micro Metrics:
    - Total True Pairs                     : 22
    - Total Predicted Pairs                : 24
    - True Positives (TP)                  : 22
    - Micro Precision                      : 0.916667
    - Micro Recall                         : 1.000000
    - Micro F0.5                           : 0.932203
==============================================================================

[Step 6/6] Exporting Final TSV Submission...
  Successfully generated: submissions/demo_submission.tsv (15 rows)

[SUCCESS] Pipeline completed successfully!
```

### 3. Run Complete Unit Test Suite

Execute the entire test suite covering all modules:

```bash
pytest tests/ -v
```

All 28 unit tests execute and pass in $\approx 2$ seconds:
```text
tests/test_blocking.py::test_generate_blocking_keys PASSED
tests/test_blocking.py::test_inverted_index_blocker_roundtrip PASSED
tests/test_blocking.py::test_sparse_tfidf_retriever PASSED
tests/test_features.py::test_feature_vector_dimension PASSED
tests/test_features.py::test_exact_match_pair PASSED
tests/test_features.py::test_disjoint_pair PASSED
tests/test_features.py::test_missing_address_handling PASSED
tests/test_metrics.py::test_official_readme_worked_example PASSED
tests/test_metrics.py::test_perfect_prediction PASSED
tests/test_metrics.py::test_zero_overlap_prediction PASSED
tests/test_metrics.py::test_correct_singleton PASSED
tests/test_metrics.py::test_false_merge_on_singleton PASSED
tests/test_metrics.py::test_missed_matched_entity PASSED
tests/test_metrics.py::test_precision_recall_asymmetry PASSED
tests/test_metrics.py::test_macro_f05_multi_entity PASSED
tests/test_metrics.py::test_evaluate_predictions_detailed_summary PASSED
tests/test_model.py::test_model_training_and_serialization PASSED
tests/test_model.py::test_optimize_macro_f05_threshold PASSED
tests/test_normalization.py::test_indic_transliteration PASSED
tests/test_normalization.py::test_french_diacritics_stripping PASSED
tests/test_normalization.py::test_legal_suffix_extraction PASSED
tests/test_normalization.py::test_address_decomposition_us PASSED
tests/test_normalization.py::test_address_decomposition_india PASSED
tests/test_normalization.py::test_address_decomposition_france PASSED
tests/test_normalization.py::test_normalize_record_end_to_end PASSED
tests/test_validation.py::test_stratified_splits_properties PASSED
tests/test_validation.py::test_save_and_load_roundtrip PASSED
tests/test_validation.py::test_integrity_failure_on_leakage PASSED
============================= 28 passed in 1.89s =============================
```

---

## End-to-End Pipeline Workflow

To reproduce or run the pipeline on the full competition dataset:

```bash
# 1. Build & seal leakage-free stratified validation partitions
python scripts/01_build_validation_splits.py --data-dir <path_to_train_dataset>

# 2. Profile raw datasets & inspect character encodings
python scripts/02_profile_dataset.py --dataset-root <path_to_dataset>

# 3. Build multi-view normalized Parquet tables
python scripts/03_build_normalizations.py --dataset-root <path_to_dataset> --split all

# 4. Audit blocking passes & candidate volume
python scripts/04_evaluate_blocking.py --country US

# 5. Train LightGBM model and calibrate optimal threshold tau*
python scripts/05_train_model.py --country US --train-sample 25000 --val-sample 3000

# 6. Run full test set streaming inference & generate submission
python scripts/06_generate_test_submission.py --output submissions/submission.tsv
```

All scripts support configurable CLI flags (`--cache-dir`, `--model-dir`, `--gt-path`) and environment variables (`AMAZON_ML_DATASET_ROOT`, `AMAZON_ML_DATA_DIR`).

---

## Scientific & Engineering Insights

### 1. The Country Invariant (100% Proven)
An exhaustive scan of all **7,638,365 training ground truth pairs** confirmed exactly **0 cross-country matches**. By enforcing country isolation before candidate retrieval:
- Cross-country false positives are mathematically reduced to **0.0%**.
- Candidate search space per query is reduced by $\approx 60\%-75\%$.

### 2. The "House Number + Prefix" Breakthrough
Inverted index passes:
- **Pass 4**: `name[:4] + house_number`
- **Pass 8**: `house_number + name[:3]`

These two rules capture **$>62\%$ of all true pairs** while yielding an average of only **5 to 7 candidate targets** per query out of millions of database records.

### 3. Asymmetric Metric Alignment
Under official evaluation rules:
- Precision is weighted $2\times$ over recall ($F_{0.5}$).
- Singleton false merges yield an instant entity score of **0.0**.

By calibrating decision thresholds to $\tau^* = 0.98$, the system prevents spurious matches, keeping singleton false positive rates below **14.4%** while preserving high macro averages.

---

## Pre-Trained Models

The repository includes pre-trained LightGBM model weights in the `models/` folder:

| Model File | Target Partition | Features | Trees | Calibrated $\tau^*$ | File Size |
| :--- | :--- | :---: | :---: | :---: | :---: |
| [`models/lgbm_model_us.joblib`](models/lgbm_model_us.joblib) | United States | 36 | 300 | 0.98 | 2.28 MB |
| [`models/lgbm_model_india.joblib`](models/lgbm_model_india.joblib) | India | 36 | 300 | 0.98 | 2.44 MB |
| [`models/lgbm_model_france.joblib`](models/lgbm_model_france.joblib) | France (Transfer) | 36 | 300 | 0.98 | 2.28 MB |

---

## Reports & Provenance

Full empirical investigation notes, ablation results, and architecture audits are archived under [`reports/`](reports/):
- [`reports/01_research_report.md`](reports/01_research_report.md) — Literature review and methodology selection
- [`reports/02_architecture_plan.md`](reports/02_architecture_plan.md) — Architectural design blueprint
- [`reports/03_data_profile_report.md`](reports/03_data_profile_report.md) — Comprehensive dataset analysis
- [`reports/04_blocking_evaluation_report.md`](reports/04_blocking_evaluation_report.md) — Multi-pass candidate generator audit
- [`reports/05_model_training_report.md`](reports/05_model_training_report.md) — Training dynamics and threshold calibration
- [`reports/06_final_system_summary.md`](reports/06_final_system_summary.md) — Production delivery overview
- [`reports/07_architecture_audit.md`](reports/07_architecture_audit.md) — Diagnostic verification audit
- [`reports/08_final_approved_architecture.md`](reports/08_final_approved_architecture.md) — Final architecture sign-off
- [`reports/09_final_test_submission_report.md`](reports/09_final_test_submission_report.md) — 1.73M entity test inference audit

---

## License & Author

- **Author**: Shanmugavel S ([GitHub](https://github.com/sshanmugave) | [leoshanmuga007@gmail.com](mailto:leoshanmuga007@gmail.com))
- **License**: Released under the [MIT License](LICENSE).
