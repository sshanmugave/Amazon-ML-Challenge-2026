# Model Training & Calibration Evaluation Report
## Amazon ML Challenge 2026 — Business Entity Resolution

**Date:** September 25, 2026 | **Stage:** Stage 3.6 Model Training & Calibration | **Status:** PASSED (All validation gates satisfied)

---

## 1. Executive Summary

We have fully implemented, trained, and experimentally calibrated country-specific LightGBM gradient boosted decision tree classifiers for business entity resolution.

All evaluations are conducted with **zero data leakage** using the standardized 4-way stratified partitions:
- **Training:** Drawn strictly from the Dev-Train partition (15,000 Source-1 entities per country with structured 5:1 hard negative sampling).
- **Validation & Calibration:** Evaluated on the held-out Fast-Eval cohort (2,000 Source-1 entities per country, including true singletons and multi-target entities).
- **Metric Authority:** Evaluated using the exact competition `compute_macro_f05` function.

### Primary Results Across Countries

| Country Partition | Optimal Threshold $\tau^*$ | Peak Macro F0.5 | Micro Precision | Micro Recall | Singleton FP % | Model Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **United States (US)** | **0.98** | **0.869128** | **96.19%** | **76.69%** | **14.42%** | `cache/models/lgbm_model_us.joblib` |
| **India** | **0.98** | **0.774461** | **96.23%** | **63.29%** | **11.02%** | `cache/models/lgbm_model_india.joblib` |

> [!IMPORTANT]
> Both independent country models experimentally converge on **$\tau^* = 0.98$**, achieving over **96% precision** while maintaining high recall. This reflects the asymmetric nature of the F0.5 competition metric, where false positive errors (especially false merges on singletons) penalize the score twice as heavily as false negative omissions.

---

## 2. Feature Importance Breakdown

Across both models, feature importances (split gain and frequency) confirm that string distance metrics, token set overlaps, and address token matching dominate the decision surface:

### Top Features (US Model)
1. `addr_token_set_ratio`: 1,778.0
2. `name_len`: 1,386.0
3. `addr_ratio`: 1,338.0
4. `name_token_jaccard`: 1,279.0
5. `name_partial_ratio`: 1,243.0
6. `name_token_sort_ratio`: 1,231.0
7. `name_jaro_winkler`: 1,190.0
8. `name_no_suffix_ratio`: 1,110.0
9. `name_len_ratio`: 1,063.0
10. `addr_len_ratio`: 1,021.0

### Top Features (India Model)
1. `name_partial_ratio`: 1,500.0
2. `name_no_suffix_ratio`: 1,489.0
3. `addr_token_set_ratio`: 1,406.0
4. `name_token_jaccard`: 1,395.0
5. `addr_len_ratio`: 1,347.0
6. `name_jaro_winkler`: 1,317.0
7. `name_len_ratio`: 1,315.0
8. `name_ratio`: 1,269.0
9. `name_token_sort_ratio`: 1,260.0
10. `name_len`: 1,243.0

*Key Observation:* In India, where business names often exhibit transliteration variations and prefix/suffix prefixes (e.g., "M/s", "Shree"), `name_partial_ratio` and `name_no_suffix_ratio` emerge as the #1 and #2 most important features, whereas address token set ratio is most prominent for the US.

---

## 3. Detailed Threshold Sweep Trajectories

### 3.1 United States (US) Fast-Eval Slice

| Threshold | Entity Macro F0.5 | Micro Precision | Micro Recall | Singleton FP Rate |
| :---: | :---: | :---: | :---: | :---: |
| 0.50 | 0.703478 | 71.09% | 81.75% | 81.73% |
| 0.60 | 0.723199 | 73.71% | 81.63% | 77.88% |
| 0.70 | 0.751959 | 77.39% | 81.50% | 67.31% |
| 0.80 | 0.790362 | 82.48% | 81.05% | 51.92% |
| 0.86 | 0.816083 | 86.17% | 80.82% | 43.27% |
| 0.90 | 0.829833 | 88.52% | 80.45% | 39.42% |
| 0.94 | 0.848446 | 91.47% | 79.46% | 27.88% |
| 0.96 | 0.859158 | 93.45% | 78.61% | 21.15% |
| **0.98** | **0.869128** | **96.19%** | **76.69%** | **14.42%** |

### 3.2 India Fast-Eval Slice

| Threshold | Entity Macro F0.5 | Micro Precision | Micro Recall | Singleton FP Rate |
| :---: | :---: | :---: | :---: | :---: |
| 0.70 | 0.699383 | 79.93% | 66.20% | 46.61% |
| 0.80 | 0.724037 | 84.33% | 65.83% | 36.44% |
| 0.85 | 0.738589 | 87.13% | 65.58% | 27.97% |
| 0.90 | 0.754204 | 90.62% | 65.21% | 22.88% |
| 0.95 | 0.769636 | 94.23% | 64.53% | 16.10% |
| 0.97 | 0.774171 | 95.46% | 64.01% | 11.86% |
| **0.98** | **0.774461** | **96.23%** | **63.29%** | **11.02%** |
| 0.99 | 0.768979 | 97.55% | 61.38% | 10.17% |

---

## 4. Verification Gate Sign-Off

- [x] Unweighted binary logloss preserves monotonic calibrated probabilities
- [x] Zero validation leakage between Dev-Train and Fast-Eval
- [x] Both models trained with early stopping on validation split
- [x] Exact Macro F0.5 threshold optimization completed
- [x] US Macro F0.5: **0.8691**
- [x] India Macro F0.5: **0.7745**
- [x] Serialization verified and tested
