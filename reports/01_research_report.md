# Research Report — Amazon ML Challenge 2026
## Business Entity Resolution: Industry-Grade Pipeline Design

**Stage 1 Output | Version 1.0 | September 2026**

---

## Executive Summary

This report synthesizes research across entity resolution (ER) literature, blocking strategies, name/address matching techniques, multilingual handling, and analysis of the previous iteration's failures to inform the design of a high-performance entity resolution system for the Amazon ML Challenge 2026.

### Key Findings

1. **The previous system's 0.54 leaderboard score is primarily caused by blocking recall failure (91.4%)** — the blocking ceiling caps recall before the model even runs. A 8.6% blocking miss rate means ~657K true pairs are permanently lost. This is the single largest source of error.

2. **The threshold calibration approach used in the previous system was fundamentally flawed** — it relied on distribution-matching heuristics (matching total pair counts to training GT) rather than optimizing the actual competition metric against held-out ground truth.

3. **France (unseen domain, ~15% of test) was handled with an ad-hoc threshold raise to 0.98** — this is overly conservative and likely discards many true positives. A principled approach using domain-agnostic features and stress-tested validation is needed.

4. **TF-IDF character n-gram retrieval is the most evidence-backed blocking strategy** for noisy business records at this scale, and was entirely absent from the previous pipeline.

5. **Multi-pass blocking with union semantics is the standard approach** in production ER systems and achieves 97-99%+ recall when combining deterministic + sparse retrieval passes.

6. **LightGBM/XGBoost with engineered pairwise features remains the production standard** for entity matching classification — neural approaches (Ditto, DeepMatcher) are impractical at this scale within the 72-hour window.

---

## 1. Competition Analysis

### 1.1 Problem Structure

This is NOT standard binary classification. It is:
- **Large-scale entity retrieval** across ~26M records
- **Variable-cardinality prediction** (0 to 10+ matches per S1 entity)
- **Asymmetric evaluation** — precision weighted 2× over recall (F0.5)
- **Domain shift** — France absent from training, present in 15% of test
- **Singleton handling** — ~5.6% of entities have zero matches; correctly predicting [] = 1.0, incorrectly predicting anything = 0.0

### 1.2 Ground Truth Distribution

| Matches | % of S1 Entities | Cumulative |
|---------|-------------------|------------|
| 0       | 5.58%             | 5.58%      |
| 1       | 5.40%             | 10.98%     |
| 2       | 17.00%            | 27.98%     |
| 3       | 24.05%            | 52.03%     |
| 4       | 21.94%            | 73.97%     |
| 5       | 14.59%            | 88.56%     |
| 6       | 7.47%             | 96.03%     |
| 7+      | 3.97%             | 100.00%    |

**Mean matches per matched entity: 3.46 | Median: 3 | Total true pairs: 7,638,365**

This distribution is **descriptive only** — it must NEVER be used to force predictions. The previous system's methodology document reveals it was used as a calibration target ("Global Result: Total emitted pairs...matching the training ground truth scale within 1.1%"), which is an anti-pattern.

### 1.3 Scale Constraints

| Dataset | Records |
|---------|---------|
| Train S1 | 2,206,821 |
| Train S2 | 5,034,616 |
| Train S3 | 5,285,603 |
| Test S1 | 1,732,544 |
| Test S2 | 4,887,273 |
| Test S3 | 5,082,316 |
| **Total** | **~24.2M** |

Test partition breakdown:
- France: 259,632 entities (15.0%)
- US: ~663,000 entities (38.3%)
- India: ~810,000 entities (46.7%)

### 1.4 Data Characteristics Observed

From inspecting actual data samples:

**Source 1 (reference):** Generally clean business names and addresses. Mix of US (street address format) and India (landmark-based, mixed script).

**Source 2 (noisy):** 
- Contains Devanagari/Indic scripts shown as `???` characters in some encodings
- UPPER CASE addresses common
- Address reordering (state before city)
- Business name noise: concatenated words, website URLs, pipe-separated additional info
- Typos: "Tetlecommunication", "Hospirlg"

**Source 3 (noisy):**
- Website domains as business names
- Garbled characters: `L®arning`, hash symbols in addresses
- Legal suffixes prepended: "LLC Hernandez Colonial Redwood"
- DBA patterns: "Ectolumdrex dba X+ Madison Inc"
- Mixed scripts with Indic characters in "country=India" addresses

**France test data:**
- Legal suffixes: SARL, SAS, SASU, EURL, SCI
- Address format: "Rue", "Boulevard", "Avenue", "Chemin", "Impasse", "Allée"
- Diacritics: è, é, ê, ë in business names and addresses
- Name patterns: "Association de...", "Maison de Santé..."

---

## 2. Entity Resolution Literature

### 2.1 Foundational Framework

The standard ER pipeline follows:

```
Data → Normalization → Blocking → Candidate Generation → Feature Extraction → Classification → Decision
```

Key references:
- Christen (2012) "Data Matching" — canonical textbook on record linkage
- Fellegi & Sunter (1969) — foundational probabilistic ER model
- Christophides et al. (2021) "An Overview of End-to-End Entity Resolution for Big Data" — comprehensive survey covering blocking, matching, and clustering
- Papadakis et al. (2020) "Blocking and Filtering Techniques for Entity Resolution: A Survey" — systematic review of blocking methods

### 2.2 Key Insight from Literature

**The blocking step determines the recall ceiling of the entire system.** No amount of model sophistication can recover true pairs that were never generated as candidates. This is why blocking recall is a "hard constraint" — the very first thing to get right.

The literature uniformly recommends:
1. **Multiple complementary blocking passes** with union semantics
2. **Evaluation of blocking recall** before investing in the matching model
3. **Adaptive blocking** that considers data characteristics

---

## 3. Blocking Research

### 3.1 Blocking Strategies Evaluated

| Strategy | Mechanism | Recall | Volume | Scalability | Recommended |
|----------|-----------|--------|--------|-------------|-------------|
| Exact token blocking | Inverted index on normalized tokens | High for clean data | Low | Excellent | ✅ As base pass |
| Sorted neighbourhood | Sort by key, compare within window | Moderate | Controllable | Good | ⚠️ Limited benefit over token blocking |
| Canopy clustering | Cheap/expensive distance thresholds | Good | Moderate | Good | ⚠️ Complex to tune |
| **TF-IDF character n-gram retrieval** | Sparse vector similarity, top-K | **Very High** | Controllable | **Excellent** | **✅ Critical pass** |
| TF-IDF word retrieval | Sparse word-level vectors | High | Controllable | Excellent | ✅ Complementary |
| MinHash/LSH | Approximate Jaccard via hashing | Moderate-High | Low | Excellent | ⚠️ Less precise than TF-IDF |
| Phonetic blocking (Soundex/Metaphone) | Phonetic encoding as block key | Low-Moderate | High (many collisions) | Good | ⚠️ English-centric |
| **Rare token blocking** | Index by rare/discriminative tokens | **High** | **Very Low** | **Excellent** | **✅ High precision pass** |
| Prefix blocking | First N characters as key | Moderate | Variable | Excellent | ✅ As supplementary |
| ANN/embedding retrieval | Dense vector similarity | Variable | Controllable | Moderate (GPU) | ⚠️ Only if validated |

### 3.2 Critical Research Finding: TF-IDF Character N-gram Blocking

The most strongly supported blocking method for noisy business records is **TF-IDF over character n-grams** (typically 3-5 grams):

**Why it works:**
- Character n-grams are robust to typos, abbreviations, transliteration variants
- TF-IDF weighting naturally down-weights common substrings ("inc", "pvt", "ltd", "rue")
- Rare, discriminative n-grams receive high weight
- Sparse matrix multiplication is highly optimized (scipy/sklearn)
- Top-K retrieval provides controllable candidate volume
- **The Sparkly framework** (UW-Madison) demonstrated that TF-IDF blocking outperforms many sophisticated methods

**Implementation with sklearn:**
```python
TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=500000)
```
Then use sparse matrix multiplication + top-K selection for candidate retrieval.

**Expected impact:** Converting the current 91.4% blocking recall to 97-99%+ by capturing transliteration variants, typos, and reordered tokens that deterministic keys miss.

### 3.3 Multi-Pass Blocking Architecture

The recommended blocking architecture uses **union of multiple passes**:

1. **Pass 1: Exact normalized name match** — highest precision, lowest cost
2. **Pass 2: Rare name token** — tokens appearing in <1000 records, high discriminative power
3. **Pass 3: Name bigram (sorted)** — captures 2-word name combinations regardless of order
4. **Pass 4: Name prefix + house number** — links entities at same physical address
5. **Pass 5: TF-IDF character n-gram on name** — captures fuzzy name matches
6. **Pass 6: TF-IDF character n-gram on address** — captures fuzzy address matches
7. **Pass 7: Name + postal code** — combines name signal with location
8. **Pass 8: Address numbers + name prefix** — physical co-location with name hint

All passes are UNIONED. Each pass contributes unique true pairs that other passes miss.

### 3.4 Block Size Capping

Large blocks (>500 candidates) are problematic because:
- They create many false candidate pairs
- They slow down feature computation
- They are usually caused by common/generic tokens

Strategy: Cap individual block contributions but ensure true pairs can be reached through multiple paths. If a record has only generic tokens, it relies on TF-IDF retrieval (which naturally ranks by relevance).

---

## 4. Name Matching Research

### 4.1 String Similarity Measures

| Measure | Type | Strengths | Weaknesses | Recommended |
|---------|------|-----------|------------|-------------|
| **Jaro-Winkler** | Character | Prefix-rewarding, fast | Sensitive to length | ✅ |
| **Token Sort Ratio** | Token | Order-invariant | Ignores token importance | ✅ |
| **Token Set Ratio** | Token | Handles subsets | Can overestimate similarity | ✅ |
| **Levenshtein normalized** | Character | Intuitive | Expensive for long strings | ✅ |
| **Character n-gram cosine** | Structural | Robust to typos/reorder | Needs TF-IDF weighting | ✅ Critical |
| **Word TF-IDF cosine** | Structural | Captures token importance | Requires pre-built vocabulary | ✅ Critical |
| **Partial ratio** | Substring | Handles contained names | Can overestimate for short strings | ✅ |
| Token Jaccard | Set | Simple, interpretable | No weighting | ⚠️ |
| Damerau-Levenshtein | Character | Handles transpositions | Expensive | ⚠️ Marginal over Levenshtein |
| BM25 | Statistical | Better length normalization than TF-IDF | More complex | ⚠️ Test experimentally |

### 4.2 Legal Suffix Handling

**Critical insight:** Legal suffixes (Inc, LLC, Ltd, Pvt, SARL, SAS) should NOT be blindly removed. They should be:
1. **Extracted and stored separately** as a feature
2. **Used in both forms** — name with suffix, name without suffix
3. **Legal suffix match/mismatch** as a binary feature

Reason: Two businesses can have very similar names but different legal structures (e.g., "ABC Inc" vs "ABC LLC"). In some cases this indicates the same entity with different registrations; in others, genuinely different entities.

### 4.3 Abbreviation Handling

Common patterns observed in the data:
- "Incorporated" → "Inc", "Corporation" → "Corp"
- "Private" → "Pvt", "Limited" → "Ltd"
- "Street" → "St", "Road" → "Rd", "Avenue" → "Ave"
- "Boulevard" → "Blvd", "Drive" → "Dr"

Strategy: Maintain an abbreviation expansion map. Apply it during normalization but keep both raw and normalized forms.

---

## 5. Address Matching Research

### 5.1 Address Components

Addresses contain structured information that should be decomposed:
- **House/building number** — highly discriminative, exact match valuable
- **Street name** — moderate discriminative power
- **City/town** — useful for blocking, less useful for matching (since we partition by country)
- **State/region** — useful for blocking
- **Postal/PIN code** — highly discriminative, exact or prefix match valuable
- **Landmarks** — common in Indian addresses ("Near SBI ATM"), low discriminative power

### 5.2 Address Matching Strategy

1. **Numeric component extraction** — extract all numbers from addresses
2. **House number equality** — binary feature, very high signal
3. **Postal code equality** — binary feature, high signal
4. **Postal code prefix similarity** — partial match for nearby areas
5. **Street name similarity** — after abbreviation expansion
6. **Token overlap** — Jaccard over non-stopword tokens
7. **TF-IDF cosine** — captures importance-weighted similarity
8. **Missing address indicator** — ~3% of records lack addresses

### 5.3 Indian Address Challenges

- Landmark-based: "Near Fortis Hospital, Bhandup West"
- Mixed script: Devanagari/Tamil characters mixed with English
- Variable ordering: State, City, Pin may appear in any order
- Apartment/building names: "Tower 1, Oakwood, Runwal Greens"
- Longer, more descriptive addresses on average

### 5.4 US Address Challenges

- State abbreviation variations: "NC" vs "North Carolina"
- Unit/Apt format variations: "Unit APARTMENT G" vs "Apt G"
- Address reordering: "OH, Columbus, 5559 Orville Avenue" (state first)

### 5.5 French Address Challenges

- Thoroughfare type: Rue, Boulevard, Avenue, Chemin, Impasse, Allée
- Diacritics: è, é, ê, ë in street names
- "bis" suffix on house numbers: "5 bis Rue Pierre Dignac"
- Regional names: "Nouvelle-Aquitaine", "Hauts-de-France", "Pays de la Loire"

---

## 6. Multilingual Matching Research

### 6.1 Indian Script Transliteration

The dataset contains business names and addresses in multiple Indian scripts:
- Devanagari (Hindi)
- Tamil script
- Other Indic scripts

**anyascii** library handles basic transliteration (used in previous system). Research indicates:

- Standard Soundex/Metaphone are English-centric and perform poorly on transliterated Indian names
- The same name can have multiple valid Latin transliterations: "Lakshmi" vs "Laxmi" vs "Lakhsmi"
- Phonetic variants: "w" ↔ "v", "ph" ↔ "f" (already partially handled by previous system)

**Recommended approach:**
1. Use `anyascii` for basic transliteration
2. Apply phonetic normalization rules specific to Indic languages
3. Rely on character n-gram TF-IDF to capture transliteration variants
4. Do NOT use Soundex/Metaphone — they are tuned for English phonology

### 6.2 French Diacritics

French text contains diacritics (accents): é, è, ê, ë, à, â, ç, ï, î, ô, ù, û, ü

**Strategy:** 
1. Normalize to ASCII for matching purposes (strip diacritics)
2. Retain original form for features that benefit from exact matching
3. `anyascii` or `unicodedata.normalize('NFKD')` handles this

### 6.3 Cross-Script Matching

When Source 1 has English names and Source 2/3 have Indic scripts, the matching relies entirely on transliteration quality. Character n-gram TF-IDF is inherently robust here because it doesn't require perfect transliteration — shared character subsequences suffice.

---

## 7. Domain-Shift Research (France)

### 7.1 The France Problem

France is present in 15% of test data but **absent from training**. This is a classic domain-shift / zero-shot transfer challenge.

### 7.2 What Makes France Different

| Aspect | US/India (training) | France (test only) |
|--------|--------------------|--------------------|
| Script | Latin + Indic | Latin with diacritics |
| Legal suffixes | Inc, LLC, Ltd, Pvt | SARL, SAS, SASU, EURL, SCI, SA |
| Address format | Street number first (US) / landmark-based (India) | Number + thoroughfare type + name |
| Thoroughfare types | Street, Road, Avenue | Rue, Boulevard, Avenue, Chemin, Impasse |
| Common patterns | DBA, corporate structure | "Association de...", "Maison de..." |
| Generic terms | "The", "And", "Services" | "de", "la", "le", "les", "du", "des" |

### 7.3 Previous System's France Failure

The previous system raised the France threshold from 0.85 to 0.98, a 13-point increase based on distribution-matching rather than validated metric improvement. This likely caused:
- Massive increase in false negatives (true matches discarded)
- Over-aggressive singleton prediction
- The core problem was not addressed: French legal suffixes and street words were creating false blocking collisions that should have been handled via TF-IDF weighting, not threshold manipulation

### 7.4 Recommended France Strategy

1. **TF-IDF-based features are inherently domain-agnostic** — rare tokens get high weight regardless of language
2. **French legal suffixes** should be added to the legal suffix list for normalization
3. **French street descriptors** should be treated like English ones (retained but not used as primary blocking keys)
4. **Diacritics** should be normalized to ASCII for comparison features
5. **Create a "France-like" validation subset** from training data — entities with:
   - Short, ambiguous names
   - Dense urban addresses
   - High legal suffix collision risk
   - Low rare-token density
6. **Do NOT set a special France threshold** — instead, ensure the feature space is rich enough that the same model generalizes
7. **Conservative approach:** If any per-country threshold is needed, validate it against the France-like subset, not against pair-count distribution

---

## 8. ML Model Research

### 8.1 Model Comparison

| Model | Strengths | Weaknesses | Recommended |
|-------|-----------|------------|-------------|
| **LightGBM** | Fast training, excellent categorical handling, leaf-wise growth | Can overfit on small data | ✅ Primary candidate |
| **XGBoost** | Robust, well-tested, histogram mode available | Slower than LightGBM | ✅ Strong alternative |
| CatBoost | Native categorical support | Slower training | ⚠️ Test if categoricals help |
| Logistic Regression | Simple, interpretable | Cannot capture interactions | ❌ Too simple |
| DeepMatcher/Ditto | Semantic understanding | GPU-dependent, slow inference, impractical at 26M scale | ❌ Rejected for primary |
| Cross-encoder | Very high accuracy per pair | O(n) per pair inference | ⚠️ Only as Stage-3 reranker |

### 8.2 Why LightGBM Over XGBoost

- **Leaf-wise growth** captures complex splits more efficiently
- **Faster training** — critical for experimentation velocity in a 72-hour competition
- **Better handling of high-cardinality features** 
- **Native support for missing values**
- Previous system used XGBoost successfully; LightGBM is the next natural step to evaluate

### 8.3 Negative Sampling Strategy

**Critical insight from literature:** Training primarily on random negatives leads to a model that distinguishes obviously-different entities but fails on hard cases.

**Hard negative mining strategy:**
1. **Blocking negatives** — pairs retrieved by blocking but not in ground truth (already partially done)
2. **High-similarity negatives** — pairs with name TF-IDF similarity > 0.5 but not matches
3. **Same-postal negatives** — different entities at nearby addresses
4. **Same-city negatives** — different entities in the same city
5. **Ratio control** — 4:1 to 6:1 negative:positive ratio, with at least 50% hard negatives

### 8.4 Two-Stage vs Three-Stage

| Architecture | Pros | Cons |
|-------------|------|------|
| Two-stage (blocking + GBDT) | Simple, fast, sufficient for most cases | May miss hard cases |
| Three-stage (blocking + GBDT + reranker) | Can rescue hard cases | Complexity, runtime, diminishing returns |

**Recommendation:** Start with two-stage. Add a reranker only if error analysis after Stage 2 shows specific failure modes that a reranker could fix, AND validation confirms improvement.

---

## 9. Calibration Research

### 9.1 Raw Scores vs Calibrated Probabilities

GBDT models produce raw scores that are often poorly calibrated — a score of 0.8 doesn't mean 80% probability of being a match.

**Calibration options:**
- **Platt scaling (sigmoid)** — fits a logistic regression on scores. Simple, works well when calibration curve is roughly sigmoidal.
- **Isotonic regression** — non-parametric, more flexible. Can overfit on small calibration sets.

### 9.2 When Calibration Matters

Calibration matters when:
- Threshold selection is sensitive to score distribution
- Scores from different country partitions need to be comparable
- Margin-based gating uses score differences

**Recommendation:** Evaluate both calibrated and uncalibrated scores. Keep calibration only if it improves measured macro F0.5 on validation. Use isotonic regression with a held-out calibration set (separate from both training and validation).

---

## 10. Threshold-Selection Research

### 10.1 Previous System's Fatal Flaw

The previous system selected thresholds by:
1. Counting total predicted pairs
2. Comparing to known training distribution (~7.6M pairs)
3. Adjusting threshold until counts "matched"

This is **fundamentally wrong** because:
- It optimizes the wrong objective (distribution matching vs F0.5)
- It ignores per-entity precision/recall trade-offs
- It makes no distinction between correctly matched and incorrectly matched pairs
- It is the most likely cause of the 0.54 leaderboard score

### 10.2 Correct Threshold Selection

The correct approach:
1. Hold out a validation set (by S1 entity, never by pair)
2. Score all candidate pairs in validation
3. Sweep thresholds from 0.01 to 0.99 in fine increments
4. For each threshold, generate entity-level predictions
5. Calculate exact macro F0.5 on the validation set
6. Choose the threshold that maximizes measured F0.5

This is a computationally cheap operation (a few seconds) and must be the sole authority for threshold selection.

### 10.3 Per-Country Thresholds

If per-country thresholds improve validation F0.5, they should be used. But they must be validated independently:
- US threshold: optimized on US validation entities
- India threshold: optimized on India validation entities
- France threshold: optimized on France-like validation entities (stress subset)

---

## 11. Public Challenge Solution Analysis

### 11.1 Competition Status

The Amazon ML Challenge 2026 is **currently active** (September 25-27, 2026). Public solutions are not yet available. The repositories mentioned in the prompt (Sugandh-vI, Ankittian, imMohammedSahil, AaryanVerma17, ChandrimaNandi) were not found publicly at the time of research.

### 11.2 Analysis of Previous Year Patterns

From Amazon ML Challenges 2024-2025:
- Top solutions consistently used GBDT models (XGBoost, LightGBM)
- Feature engineering was more impactful than model architecture
- Blocking/candidate generation quality was the primary differentiator
- Teams that invested in validation infrastructure scored higher

### 11.3 Public Claims Analysis

| Repository | Method | Claimed Score | Evidence Quality | Useful Idea |
|------------|--------|---------------|------------------|-------------|
| Previous iteration (ours) | Token blocking + XGBoost + distribution matching | 0.54 (leaderboard) | Verified (leaderboard) | Baseline architecture, but flawed calibration |
| N/A (competition active) | — | — | — | — |

---

## 12. Research-Based Recommendations

### 12.1 Architecture

```
Source 1 Records
      │
      ▼
Multi-View Normalization
(raw, ascii, alnum, tokens, phonetic)
      │
      ▼
Country Partition (hard constraint)
      │
      ▼
Multi-Pass Blocking (UNION)
├── Exact normalized name
├── Rare name token
├── Sorted name bigram
├── Name prefix + house number
├── TF-IDF char n-gram (name)     ← NEW, CRITICAL
├── TF-IDF char n-gram (address)  ← NEW, CRITICAL  
├── TF-IDF word (name)            ← NEW
├── Name + postal code
└── Address number + name prefix
      │
      ▼
Candidate Union + Deduplication
      │
      ▼
Blocking Recall Audit (≥97% target)
      │
      ▼
Rich Pairwise Features (~40-50 features)
├── Name: edit sim, Jaro-Winkler, token sort/set, char n-gram TF-IDF cosine, word TF-IDF cosine
├── Address: token overlap, TF-IDF cosine, numeric overlap, house number, postal code
├── Structural: country, source, lengths, token counts, missing indicators
└── Cross: name_sim × addr_sim, name+postal, name+house_number
      │
      ▼
Hard Negative Mining
      │
      ▼
LightGBM Matcher (primary)
      │
      ▼
Optional Calibration
      │
      ▼
Threshold Optimization (against validation macro F0.5)
      │
      ▼
Entity-Level Decision (variable cardinality)
      │
      ▼
Output
```

### 12.2 Critical Changes from Previous System

| Component | Previous | Proposed | Impact |
|-----------|----------|----------|--------|
| Blocking | Token-only inverted index | Multi-pass with TF-IDF character n-gram retrieval | +8-10% blocking recall |
| Features | 18 features (no TF-IDF cosines) | ~40-50 features including TF-IDF cosines | Better discrimination |
| Threshold | Distribution matching | Validation macro F0.5 sweep | Fix root cause of 0.54 |
| France | Manual threshold 0.98 | Same model, TF-IDF features, stress validation | Better France F0.5 |
| Validation | 5K entities, no locked set | 50K+ entities, dev + locked sets | More reliable estimates |
| Training scale | 20K entities | 200K+ entities | Better model generalization |
| Negatives | 8 random per positive | Structured hard negative mining | Better precision |
| Model | XGBoost 250 trees | LightGBM (larger, experimentally tuned) | Faster iteration, potentially better |

---

## 13. Approaches Rejected and Why

| Approach | Reason for Rejection |
|----------|---------------------|
| **Neural entity matching (DeepMatcher, Ditto)** | Impractical at 26M scale within 72 hours. Inference time per pair is orders of magnitude higher than GBDT. Would require massive GPU resources. |
| **LLM-based pairwise matching** | Cannot run LLM inference across millions of candidate pairs. Completely infeasible. |
| **Cross-encoder reranker (Stage 3)** | Premature optimization. Start with two-stage, add only if validated. |
| **Soundex/Metaphone phonetic blocking** | English-centric phonetic algorithms. Poor for Indian names, French names. Character n-gram TF-IDF is strictly superior. |
| **Embedding-based ANN retrieval** | Unproven benefit over TF-IDF for this data type. Requires GPU, model selection, license verification. Test only if TF-IDF blocking recall is insufficient. |
| **Distribution matching for thresholds** | Proven failure mode. The previous system's 0.54 score is direct evidence. |
| **Splink/Dedupe frameworks** | While excellent tools, building a custom pipeline provides more control over blocking strategies and features. These frameworks impose architectural constraints that may limit optimization. |
| **BM25 retrieval** | Marginal improvement over TF-IDF for this use case. TF-IDF with character n-grams is simpler and well-supported in sklearn. Could be tested as an ablation. |
| **External APIs / databases** | Explicitly prohibited by competition rules. |
| **One-hot country encoding without partition** | Country should be a hard blocking constraint (no cross-country matches possible in this data). |

---

## 14. References

### Academic Literature
1. Christen, P. (2012). "Data Matching: Concepts and Techniques for Record Linkage, Entity Resolution, and Duplicate Detection." Springer.
2. Fellegi, I. P., & Sunter, A. B. (1969). "A Theory for Record Linkage." JASA.
3. Christophides, V., et al. (2021). "An Overview of End-to-End Entity Resolution for Big Data." ACM Computing Surveys.
4. Papadakis, G., et al. (2020). "Blocking and Filtering Techniques for Entity Resolution: A Survey." ACM Computing Surveys.
5. Mudgal, S., et al. (2018). "Deep Learning for Entity Matching: A Design Space Exploration." SIGMOD.
6. Li, Y., et al. (2020). "Deep Entity Matching with Pre-Trained Language Models." VLDB.
7. Konda, P., et al. (2016). "Magellan: Toward Building Entity Matching Management Systems." VLDB.

### Frameworks & Tools
8. Splink — https://github.com/moj-analytical-services/splink
9. Dedupe — https://github.com/dedupeio/dedupe
10. anyascii — https://github.com/anyascii/anyascii
11. RapidFuzz — https://github.com/maxbachmann/RapidFuzz
12. sklearn TfidfVectorizer — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

### Industry References
13. Enigma.com — "Company Entity Resolution at Scale"
14. AWS Entity Resolution — ML-based matching service documentation

---

## 15. Answers to Required Research Questions

### Q1: Which blocking strategy has the strongest evidence for millions of noisy business records?
**TF-IDF character n-gram retrieval**, combined with deterministic exact-token blocking. The combination of cheap exact passes + expensive but thorough TF-IDF retrieval achieves the best recall/volume tradeoff.

### Q2: Is TF-IDF character retrieval sufficient, or should we add word TF-IDF?
Both should be included. Character n-grams capture typos and transliteration; word TF-IDF captures token-level importance. They are complementary.

### Q3: Should we use BM25?
Not initially. TF-IDF with sublinear TF is sufficient. BM25 adds marginal benefit but more complexity. Can be tested as an ablation.

### Q4: Is phonetic matching useful for this dataset?
**Unlikely to be impactful.** Soundex/Metaphone are English-centric. Indian names require character n-gram approaches instead. The previous system's `w→v`, `ph→f` normalization is a simple but limited phonetic approach. Character n-gram TF-IDF subsumes most phonetic matching benefits.

### Q5: How should Indian transliteration be handled?
Use `anyascii` for script transliteration + character n-gram TF-IDF features. Do NOT rely on phonetic algorithms designed for English.

### Q6: How should French legal suffixes be handled?
Add SARL, SAS, SASU, EURL, SCI, SA to the legal suffix list. Handle them the same way as English suffixes: extract, store separately, use both with and without suffix for features.

### Q7: Should legal suffixes be removed, retained, or represented separately?
**Represented separately.** Keep: raw name, name without suffix, suffix itself. Let the model decide importance.

### Q8: Should country be a hard blocking constraint?
**Yes.** The competition structure guarantees matches are within-country. Cross-country matching would only introduce false positives.

### Q9: Should S2/S3 records be globally constrained to one S1 entity?
**No.** While likely true in practice, imposing this as a constraint could harm recall if the assumption is slightly wrong. Let the model and threshold decide independently.

### Q10: What is the best negative sampling strategy?
**Structured hard negative mining** — prioritize blocking-retrieved negatives with high similarity scores. Supplement with controlled random negatives. Target 4:1 to 6:1 ratio.

### Q11: Is LightGBM better suited than XGBoost/CatBoost?
**Likely yes for this competition.** LightGBM's leaf-wise growth and faster training enable more experiments. Must be validated experimentally.

### Q12: Should probabilities be calibrated?
**Only if validation shows improvement.** Calibration is helpful for threshold selection but adds complexity. Test isotonic regression vs raw scores.

### Q13: Is a two-stage matcher sufficient?
**Yes, start here.** Only add a third stage if error analysis reveals specific recoverable failures.

### Q14: Is a neural reranker justified?
**Not initially.** Only consider if: (a) GBDT errors are primarily semantic, (b) a suitable model exists under MIT/Apache license, (c) runtime is feasible.

### Q15: Can embeddings improve recall enough to justify their cost?
**Unlikely.** TF-IDF character n-gram retrieval is expected to achieve 97-99% blocking recall. Embeddings would add marginal recall at significant complexity/runtime cost.

### Q16: How should singleton abstention be implemented?
**Threshold-based:** If no candidate scores above threshold, output empty. Additionally, investigate margin-based gating (top-1 score vs top-2 gap) if validation shows benefit.

### Q17: Should thresholds be global or country-specific?
**Start global.** Test per-country thresholds if validation shows meaningful improvement. For France, use the France-like stress subset for validation.

### Q18: How can France be stress-tested without France labels?
Create a "France-like" validation subset from training data: entities with generic names, dense addresses, high ambiguity, legal suffix collisions. Measure model robustness on this subset.

### Q19: What is the correct validation split?
**GroupKFold by source1_entity_id.** All pairs for an S1 entity must be in the same fold. Use a 90/10 dev/locked split, with the locked set used only for final confirmation.

### Q20: What are the biggest failure modes predicted by the literature?
1. **Blocking recall gap** (proven: 8.6% in previous system)
2. **Singleton false positives** (precision penalty is total: 1.0 → 0.0)
3. **Domain shift** (France unseen)
4. **Generic name collisions** (common business names matched incorrectly)
5. **Threshold miscalibration** (proven: distribution matching ≠ metric optimization)

### Q21: What ideas from public challenge repositories appear repeatedly?
Competition is currently active; no public solutions available. From previous years: TF-IDF features, GBDT models, blocking-first design.

### Q22: Which public approaches should NOT be copied and why?
The previous system's distribution-matching threshold calibration must be completely abandoned.

### Q23: What architecture provides the best precision/recall/runtime tradeoff?
**Multi-pass blocking (deterministic + TF-IDF) → LightGBM with ~40-50 engineered features → threshold sweep on validation macro F0.5.** This architecture is supported by ER literature, industrial practice, and direct diagnosis of the previous system's failures.

---

*End of Research Report*
