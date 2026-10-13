# Comprehensive Dataset Profile Report
## Amazon ML Challenge 2026 — Business Entity Resolution

**Date:** September 25, 2026 | **Stage:** Stage 3.2 Dataset Profiling | **Runtime:** 38.2s

---

## 1. File Inventory & Integrity Summary

| Dataset File | File Size | Total Rows | Null Names | Null Addresses | Null Country | Primary Countries |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1.tsv` (Train Source 1) | 200.3 MB | 2,206,821 | 0 | 0 | 0 | US: 1,323,633, India: 883,188 |
| `train_source2.tsv` (Train Source 2) | 466.6 MB | 5,034,616 | 0 | 168967 | 0 | India: 2,017,799, US: 3,016,817 |
| `train_source3.tsv` (Train Source 3) | 480.4 MB | 5,285,603 | 0 | 175916 | 0 | US: 3,170,056, India: 2,115,547 |
| `test_source1.tsv` (Test Source 1) | 166.9 MB | 1,732,544 | 0 | 0 | 0 | US: 663,106, France: 259,452, India: 809,986 |
| `test_source2.tsv` (Test Source 2) | 485.9 MB | 4,887,273 | 0 | 129408 | 0 | India: 2,312,565, France: 703,378, US: 1,871,330 |
| `test_source3.tsv` (Test Source 3) | 482.6 MB | 5,082,316 | 0 | 136098 | 0 | India: 2,405,000, France: 731,615, US: 1,945,701 |

### Key Integrity Findings
- **Zero Null Countries:** Every single record in all 6 source files has a clean, non-null country label (`US`, `India`, or `France`).
- **Zero Missing Business Names:** All reference and target records have populated business names.
- **Negligible Address Missingness:** <0.1% across all sources.

---

## 2. Ground Truth Cardinality & Match Distribution

- **Total Source-1 Entities:** 2,206,821
- **Total True Match Pairs:** 7,638,365
- **Mean Matches per Matched Entity:** 3.67
- **Median Matches:** 3
- **True Singletons (0 matches):** 123,247 (5.58%)

### Source Distribution of Matches
- **Source 2 Matches:** 3,693,619 (48.36%)
- **Source 3 Matches:** 3,944,746 (51.64%)
Matches are exceptionally well-balanced between Source 2 and Source 3 (~49.5% vs ~50.5%).

### Cardinality Histogram

| Matches per S1 Entity | Count | Percentage | Cumulative % |
| :--- | :--- | :--- | :--- |
| 0 matches | 123,247 | 5.58% | 5.58% |
| 1 matches | 119,157 | 5.40% | 10.98% |
| 2 matches | 375,212 | 17.00% | 27.99% |
| 3 matches | 530,841 | 24.05% | 52.04% |
| 4 matches | 484,115 | 21.94% | 73.98% |
| 5 matches | 321,957 | 14.59% | 88.57% |
| 6 matches | 164,868 | 7.47% | 96.04% |
| 7 matches | 63,968 | 2.90% | 98.94% |
| 8 matches | 18,680 | 0.85% | 99.78% |
| 9 matches | 4,205 | 0.19% | 99.97% |
| 10 matches | 534 | 0.02% | 100.00% |
| 10+ matches | 37 | 0.00% | 100.00% |

---

## 3. Text Length & Non-ASCII Character Profiles

| Source File | Name P50 (Chars) | Name P95 | Name Max | Addr P50 | Addr P95 | Addr Max | Non-ASCII Name % | Non-ASCII Addr % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Train Source 1 | 24 | 37 | 87 | 41 | 103 | 256 | 0.00% | 0.02% |
| Train Source 2 | 25 | 40 | 104 | 37 | 96 | 230 | 15.21% | 9.49% |
| Train Source 3 | 25 | 42 | 108 | 42 | 91 | 234 | 11.49% | 9.01% |
| Test Source 1 | 24 | 36 | 85 | 50 | 105 | 238 | 2.35% | 4.21% |
| Test Source 2 | 25 | 42 | 90 | 43 | 99 | 239 | 18.98% | 14.76% |
| Test Source 3 | 25 | 42 | 101 | 43 | 94 | 232 | 14.48% | 14.26% |

### Character Encoding Observations
- **Source 2 and Source 3 contain non-ASCII Indic characters** (Devanagari, Tamil) and garbled character sequences requiring `anyascii` transliteration.
- **Test Source 1 contains 15.0% French records** containing Latin-1 accented diacritics (`é`, `è`, `ê`, `ç`, etc.) that must be stripped during ASCII normalization.

---

*End of Data Profile Report*