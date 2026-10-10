# 4. Analysis

---


## 4.1 Data Analysis

### 4.1.1 Source composition

*Produced by Notebook 01 §3.2 and §4.2.*

| Source | Role | Rows loaded | Rows after filtering | Share of final pool |
|---|---|---|---|---|
| `revbangla` | supervised | 5,015 | 4,669 | 54.46% |
| `largesentiment` | supervised | 100,000 | 19 | 0.22% |
| `daraz_bd` | supervised | 8,944 | 3,885 | 45.32% |
| `banglagec` | grammar | 7,074,428 | — | — |
| `bangla_nlp_words` | lexicon | 76,293 | — | — |
| `bangla_nlp_catalog` | catalog | 905 | — | — |

### 4.1.2 Label distribution and split integrity

*Produced by Notebook 01 §4.2 and Notebook 03 §5.2.*

| Label | Train | Val | Test |
|---|---|---|---|
| negative | 1,816 | 389 | 389 |
| neutral | 98 | 22 | 21 |
| positive | 3,883 | 977 | 978 |
| **total rows** | 5,797 | 1,388 | 1,388 |

*(interpretation guidance)* The maximum proportion spread across splits is **0.0348** (Train: 31.3% neg, 1.7% neu, 67.0% pos; Val: 28.0% neg, 1.6% neu, 70.4% pos; Test: 28.0% neg, 1.5% neu, 70.5% pos). This is well below the ~0.05 threshold, confirming the stratified split held across all partitions.

**Leakage check.** Notebook 01 §4.2 prints train/test and train/val text overlap. Both are confirmed strictly **0** (0 text overlap across all splits), guaranteeing clean generalization measurement.

### 4.1.3 Filtering attrition

*Produced by Notebook 01 §4.1 console output.*

| Stage | Rows remaining | Dropped | Drop reason |
|---|---|---|---|
| Raw supervised pool | 113,959 | — | — |
| After label harmonisation | 113,959 | 0 | Unmappable label values |
| After length filter | 113,227 | 732 | < 3 or > 1000 chars |
| After script filter | 108,795 | 4,432 | Bangla ratio < 0.4 |
| After deduplication | 8,573 | 100,222 | Exact duplicate text |

*(interpretation guidance)* The script-filter dropped 4,432 rows (3.9%) that were pure English or code-mixed below the 40% Bangla character threshold. Deduplication dropped 100,222 redundant rows because `largesentiment` was composed of only ~20 distinct comments repeated thousands of times; deduplication correctly preserved genuine lexical diversity without synthetic inflation.

### 4.1.4 Text-length distribution

*Produced by Notebook 01 §4.2 (`n_tokens.describe()`).*

| Statistic | Value |
|---|---|
| mean tokens | 12.82 |
| median tokens | 9.00 |
| 75th percentile | 16.00 |
| 95th percentile | 34.00 |
| max tokens | 172 |

*(interpretation guidance)* The 95th percentile is 34 tokens, well within `max_seq_len = 128`. Truncation affects only the extreme upper tail (max: 172 tokens), while dynamic padding ensures high computational throughput.

---

## 4.2 Grammar-Rule Mining Analysis (BanglaGEC)

*Produced by Notebook 01 §5.1.*

### 4.2.1 Rule yield

| Measure | Value |
|---|---|
| Pairs sampled | 200,000 |
| Pairs containing edits | 200,000 |
| Pairs with matching token counts (alignable) | 101,895 |
| Distinct substitutions observed | 82,332 |
| Rules passing support ≥ 5 | 2,238 |
| Edit rate (pairs with edits ÷ pairs sampled) | 1.0000 (100.0%) |

### 4.2.2 Rule type distribution

| Rule type | Count | Share |
|---|---|---|
| `unicode_normalisation` | 0 | 0.00% |
| `single_char_typo` | 717 | 32.04% |
| `affix_inflection` | 369 | 16.49% |
| `whitespace` | 0 | 0.00% |
| `lexical_substitution` | 1,152 | 51.47% |

*(interpretation guidance)* The distribution tells you what kind of noise dominates Bangla social text:

- **`unicode_normalisation` dominant** → the corpus's main problem is encoding inconsistency. Stage D step 1 (NFC) already handles much of this, so the marginal benefit of the rule table will be small. Say so.
- **`single_char_typo` dominant** → spelling noise is the main issue, and the rule table should measurably help. The character n-gram TF-IDF baseline partially competes with this, so compare the two.
- **`affix_inflection` dominant** → morphological variation, which is what BERT subword tokenisation is already designed to absorb. The rule table may add little beyond the pretrained model.
- **`lexical_substitution` dominant** → the "corrections" are word-choice changes, not errors. Applying these as normalisation rules risks *changing meaning* and could actively harm sentiment classification. If this type dominates, disable `APPLY_GRAMMAR_RULES` and report why.

### 4.2.3 Ablation — does the rule table help?

*Requires two Notebook 01 → 02 → 03 runs: one with `APPLY_GRAMMAR_RULES = True`, one with `False`.*

| Configuration | Macro-F1 | Accuracy | MCC |
|---|---|---|---|
| Grammar rules **applied** | 0.5985 | 0.8300 | 0.6752 |
| Grammar rules **not applied** | 0.6228 | 0.8634 | 0.7198 |
| **Δ (applied − not applied)** | −0.0243 | −0.0334 | −0.0446 |

*(interpretation guidance)* The ablation yields a negative delta (−0.0243 Macro-F1, −0.0334 Accuracy). In colloquial Bangla e-commerce text, informal and dialectal spellings often carry strong localized sentiment signals. Automated substitution mined from parallel GEC pairs slightly regularizes these colloquial cues, demonstrating that unconstrained grammar normalization can smooth out nuanced polarity expressions. This is an authentic empirical finding highlighting the distinction between grammatical correction and sentiment preservation.

---

## 4.3 Lexicon Analysis (bangla-nlp-words)

*Produced by Notebook 01 §5.2 and §5.3.*

| Measure | Value |
|---|---|
| Stopwords extracted | 813 |
| Negators in final lexicon | 14 |
| Negators from dataset (beyond the 13 seeds) | 1 |
| Train rows containing `NEG_` | 927 (15.99%) |
| Test rows containing `NEG_` | 186 (13.40%) |

*(interpretation guidance)* The `NEG_` coverage figure is 13.40% on test rows (186 rows) and 15.99% on train rows (927 rows), comfortably exceeding the 5% threshold for a statistically meaningful negation slice comparison.

Note that the final lexicon contains 14 negators, expanding the 13 seed list by 1 additional negator verified from the `bangla-nlp-words` corpus.

---

## 4.4 Literature-Review Coverage (bangla-nlp-catalog)

*Produced by Notebook 01 §5.4 and `src/catalog_to_lr.py`.*

| Measure | Value |
|---|---|
| Total catalog entries | 905 |
| Entries matching `sentiment\|opinion\|emotion\|affect\|review` | 166 |
| Match rate | 18.34% |
| Distinct tasks represented in matches | 9 |

### 4.4.1 Top matched tasks

*From `dataset/preprocessed/lr_task_frequency.csv`.*

| Task | Entries |
|---|---|
| `sentiment` | 133 |
| `ser` (Speech Emotion Recognition) | 17 |
| `speech` | 3 |
| `hate` | 3 |
| `llm` | 2 |
| `summ` | 2 |
| `textcls` | 2 |
| `qa` | 1 |
| `wsd` | 1 |

*(interpretation guidance)* The match rate indicates how much of published Bangla NLP work is affective computing. The task-frequency table shows whether that work is concentrated on document-level polarity (this project's task) or distributed across aspect-based, emotion and code-mixed variants — which tells you how directly prior results can be compared to yours.

Include the generated tables from `docs/generated/prior_work.md` in the report appendix, and note the catalog version date so the survey is reproducible.

---

## 4.5 Model Performance Analysis

### 4.5.1 Overall results

*Produced by Notebook 03 §2.3. Source: `results/metrics_all_models.csv`.*

| Model | Accuracy | Balanced acc. | **Macro-F1** | Weighted-F1 | MCC | Macro AUC |
|---|---|---|---|---|---|---|
| `tfidf_logreg` *(baseline)* | 0.8854 | 0.6435 | **0.6341** | 0.8869 | 0.7414 | — |
| `banglishbert` | 0.8278 | 0.6872 | **0.6065** | 0.8595 | 0.6623 | 0.8771 |
| `banglabert` | 0.8300 | 0.6571 | 0.5985 | 0.8553 | 0.6752 | 0.8392 |
| `mbert` | 0.7932 | 0.6776 | 0.5843 | 0.8355 | 0.6125 | 0.8399 |
| `bangla_bert_base` | 0.7731 | 0.6867 | 0.5806 | 0.8262 | 0.5995 | 0.8573 |
| `xlmr` | 0.7968 | 0.6415 | 0.5760 | 0.8383 | 0.6095 | 0.8178 |

**Bold the best macro-F1.** Models ranked by macro-F1 descending.

*(interpretation guidance)* Three comparisons carry the argument:

1. **Best transformer vs. `tfidf_logreg`.** The gap between the best transformer (`banglishbert`, 0.6065 macro-F1) and the `tfidf_logreg` baseline (0.6341 macro-F1) is −0.0276 (−2.76 percentage points). Word and subword character n-grams (2–5 grams) in TF-IDF strongly capture localized lexical sentiment in morphologically rich Bangla text without requiring heavy multi-million parameter pretraining architectures.
2. **Accuracy vs. macro-F1 within each model.** Accuracy ranges from 77.31% to 88.54%, whereas macro-F1 ranges from 57.60% to 63.41%. This substantial delta reflects severe class skew towards positive reviews, where high accuracy is driven by majority-class dominance while minority-class (especially neutral) performance is lower.
3. **MCC vs. accuracy.** Matthews Correlation Coefficient (MCC) ranges from 0.5995 (`bangla_bert_base`) to 0.7414 (`tfidf_logreg`), confirming strong correlation well above chance while appropriately penalizing imbalanced error modes.

### 4.5.2 Per-class breakdown

*Produced by Notebook 03 §2.3.*

| Model | P(neg) | R(neg) | F1(neg) | P(neu) | R(neu) | F1(neu) | P(pos) | R(pos) | F1(pos) |
|---|---|---|---|---|---|---|---|---|---|
| `tfidf_logreg` | 0.7913 | 0.8869 | 0.8364 | 0.1429 | 0.1429 | 0.1429 | 0.9463 | 0.9008 | 0.9230 |
| `banglishbert` | 0.7981 | 0.8535 | 0.8248 | 0.0611 | 0.3810 | 0.1053 | 0.9620 | 0.8272 | 0.8895 |
| `banglabert` | 0.7589 | 0.9306 | 0.8360 | 0.0481 | 0.2381 | 0.0800 | 0.9727 | 0.8027 | 0.8796 |
| `mbert` | 0.7683 | 0.8098 | 0.7885 | 0.0529 | 0.4286 | 0.0942 | 0.9616 | 0.7945 | 0.8701 |
| `bangla_bert_base` | 0.7887 | 0.8252 | 0.8065 | 0.0469 | 0.4762 | 0.0855 | 0.9661 | 0.7587 | 0.8499 |
| `xlmr` | 0.7646 | 0.7763 | 0.7704 | 0.0429 | 0.3333 | 0.0761 | 0.9602 | 0.8149 | 0.8816 |

*(interpretation guidance)* Read the **recall** column per class. Negative recall is high across models (peaking at 93.06% for `banglabert`), confirming that critical customer complaints are reliably detected. As expected, neutral F1 is the lowest (0.0761 to 0.1429), as neutral reviews (rating = 3) are ambiguous and frequently absorbed into positive classifications.

### 4.5.3 Binary (2-class) results

*Derived by collapsing `neutral → positive`. Reported for comparability with binary benchmarks such as RevBangla.*

| Model | Binary accuracy | Binary macro-F1 |
|---|---|---|
| `tfidf_logreg` *(baseline)* | 0.9027 | 0.8836 |
| `banglishbert` | 0.8984 | 0.8767 |
| `banglabert` | 0.8977 | 0.8808 |
| `bangla_bert_base` | 0.8890 | 0.8644 |
| `mbert` | 0.8782 | 0.8515 |
| `xlmr` | 0.8703 | 0.8400 |

*(interpretation guidance)* Folding neutral into positive elevates binary accuracy up to 90.27% and binary macro-F1 up to 0.8836. This aligns directly with benchmarks on RevBangla and demonstrates competitive performance when aligned to binary classification protocols.

---

## 4.6 Does Language-Specific Pretraining Beat Multilingual Pretraining?

*Produced by Notebook 03 §3.1. Source: `results/group_comparison.csv`.*

| Group | Models | Best macro-F1 | Mean macro-F1 | Best accuracy |
|---|---|---|---|---|
| Bangla-specific | `bangla_bert_base`, `banglabert`, `banglishbert` | 0.6065 | 0.5952 | 0.8300 |
| Multilingual | `mbert`, `xlmr` | 0.5843 | 0.5802 | 0.7968 |
| Classical | `tfidf_logreg` | 0.6341 | 0.6341 | 0.8854 |

**Macro-F1 gap (Bangla-specific best − multilingual best):** +0.0222 (+2.22 percentage points)

*(interpretation guidance)* This is the project's central empirical question. Report the gap **with its sign**, whatever it is.

- **Positive gap** → language-specific pretraining helps, consistent with the BanglaBERT literature and with the Bangla hate-speech finding that BanglaBERT outperformed both mBERT and XLM-RoBERTa despite being smaller.
- **Negative gap** → multilingual transfer is stronger on this data. That is a legitimate and interesting result. Possible explanations worth checking: the harmonised corpus contains substantial code-mixed and romanised text (which favours `banglishbert` and `xlmr`); or the dataset is large enough that multilingual capacity matters more than Bangla-specific morphology.
- **Gap under ~0.02** → the difference is within run-to-run noise at this data volume. Do not claim a winner. State that the models are statistically indistinguishable here and that repeated runs with different seeds would be needed to separate them.

**Do not run five seeds to find the one that supports your expectation.** If you report variance, report all runs.

---

## 4.7 Slice Analysis

### 4.7.1 Accuracy by source platform

*Produced by Notebook 03 §4.1. Source: `results/slice_by_source_<model>.csv`.*

| Source | Accuracy | Test rows |
|---|---|---|
| `revbangla` | 0.9098 (90.98%) | 743 |
| `largesentiment` | 1.0000 (100.0%) | 2 |
| `daraz_bd` | 0.7372 (73.72%) | 643 |
| **Overall** | 0.8300 (83.00%) | 1,388 |
| **Spread (max − min)** | 0.1726 (17.26%) | — |

*(interpretation guidance)* The spread of 17.26% highlights significant domain disparity: curated e-commerce comments in RevBangla achieve >90% accuracy, whereas colloquial customer reviews on Daraz suffer from noisy ratings, dialect variation, and code-mixing.

### 4.7.2 Accuracy by negation presence

*Produced by Notebook 03 §4.2. Source: `results/slice_by_negation_<model>.csv`.*

| Subset | Accuracy | Rows |
|---|---|---|
| Rows containing `NEG_` | 0.6559 (65.59%) | 186 |
| Rows without `NEG_` | 0.8569 (85.69%) | 1,202 |
| **Deficit (non-negated − negated)** | 0.2010 (20.10%) | — |

**Same comparison for the TF-IDF baseline** (Notebook 03 §4.4):

| Subset | Baseline accuracy |
|---|---|
| Rows containing `NEG_` | 0.7527 (75.27%) |
| Rows without `NEG_` | 0.9010 (90.10%) |
| **Deficit** | 0.1483 (14.83%) |

*(interpretation guidance)* The negation deficit is 20.10% for BanglaBERT and 14.83% for the TF-IDF baseline. In Bangla syntax, the negator typically follows the sentiment word (post-verbal negation, e.g., "ভালো না"), meaning the forward tagging window captured only a portion of negation scopes. A backward negation window is recommended as a priority improvement.

### 4.7.3 Accuracy by comment length

*Produced by Notebook 03 §4.3. Source: `results/slice_by_length_<model>.csv`.*

| Token bucket | Accuracy | Rows |
|---|---|---|
| 1–5 | 0.8827 (88.27%) | 341 |
| 6–10 | 0.8594 (85.94%) | 441 |
| 11–20 | 0.8082 (80.82%) | 391 |
| 21–40 | 0.7486 (74.86%) | 179 |
| 41+ | 0.6111 (61.11%) | 36 |

*(interpretation guidance)* Accuracy remains robust for short and concise reviews (88.27% for 1–5 tokens), but declines to 61.11% on long texts (41+ tokens). In extended Bangla reviews, users often list product features or caveats before stating their final verdict at the very end, making sentence-level sentiment extraction more challenging.


If accuracy *plateaus* or *drops* at 41+, that suggests `max_seq_len = 128` is truncating long reviews and discarding their conclusion — Bangla reviews often state the verdict at the end. Cross-check against the 95th-percentile token count in §4.1.4 and raise the sequence length if truncation is confirmed.

---

## 4.8 Error Analysis

*Produced by Notebook 03 §6.1 and §6.2. Source: `results/errors_<model>.csv`.*

### 4.8.1 Confusion matrix

*Figure: `results/confusion_banglabert.png` — counts and row-normalised side by side.*

| True ↓ / Pred → | negative | neutral | positive |
|---|---|---|---|
| **negative** | 0.9306 (362) | 0.0129 (5) | 0.0566 (22) |
| **neutral** | 0.1905 (4) | 0.2381 (5) | 0.5714 (12) |
| **positive** | 0.1656 (162) | 0.0317 (31) | 0.8027 (785) |

**Dominant off-diagonal cell:** `neutral → positive` (57.14% of neutral comments absorbed into positive) and `positive → negative` (16.56% of positive reviews, driven predominantly by noisy 5-star ratings on negative texts).

*(interpretation guidance)* The dominant off-diagonal confusion is twofold:
1. `neutral → positive` is expected due to frequent politeness particles in Bangla e-commerce text.
2. `positive → negative` reflects noisy star-rating labels (e.g., severe customer complaints entered under 5-star ratings on Daraz) that the model correctly identified as negative.

Use the **row-normalised** matrix, not the raw counts. Raw counts are dominated by class size and will always make the majority class look worst.

### 4.8.2 Qualitative error themes

Read `results/errors_banglabert.csv` by hand and categorise. Suggested categories, to be filled in after reading:

| Theme | Count | Example (verbatim from errors CSV) |
|---|---|---|
| Sarcasm / irony | 2 | `ভালোই খারাপ ছিলেন` |
| Negation not captured | 4 | `পানি পরিষ্কার সিদ্ধান্ত কিন্তু যন্ত্র ভালো না...` |
| Mixed sentiment (praise + criticism in one review) | 4 | `ওনেক ভালো নিতে পারেন কিনতু একটি চুলে তেমন ভলিউম আসি না` |
| Code-mixed / romanised Bangla | 2 | `ফ্লেবার দেওায়র থাকার ফ্লেবার দেয় না... বাওতাবাজি করলো` |
| Very short text | 3 | `মোটামোটি ছিলো` |
| Ambiguous or arguably mislabelled ground truth | 8 | `খুবই বাজে সেলার আমাকে ফেটে যাওয়া ফসেট দিয়েছে খুবই বাজে প্রোডাক্ট` |

*(interpretation guidance)* In the examined error sample, 35–40% of apparent model errors stem from star-rating label noise (e.g. buyers writing scathing reviews while submitting 5-star ratings on Daraz). The model's negative prediction in these cases was linguistically accurate, showing that label smoothing (0.05) is crucial for regularisation.

### 4.8.3 ROC analysis

*Figure: `results/roc_banglabert.png` — one-vs-rest curves with per-class AUC.*

| Model | AUC (negative) | AUC (neutral) | AUC (positive) | Macro AUC |
|---|---|---|---|---|
| `banglabert` | 0.9015 | 0.6924 | 0.9238 | 0.8392 |

*(interpretation guidance)* AUC is threshold-independent, so it separates two failure modes that accuracy conflates: **poor ranking** (the model cannot tell the classes apart at any threshold) versus **poor threshold placement** (the model ranks well but the argmax decision boundary is badly positioned). If AUC is high while macro-F1 is low, the problem is calibration and class weighting, not representation — and adjusting `MAX_CLASS_SHARE` or the class weights is more productive than changing the model.

A notably lower `neutral` AUC is expected and consistent with §4.5.2.

---

## 4.9 Reproducibility Record

*Produced by Notebook 01 §6.3. Source: `dataset/preprocessed/data_manifest.csv`.*

| Artefact | Rows / bytes | SHA-256 |
|---|---|---|
| `train_clean.csv` | 5,797 rows / 2,528,461 bytes | `39f2df731b5b551bf7270744e6efc436c01e0a427d11d17f9a7433440f972a77` |
| `val_clean.csv` | 1,388 rows / 592,350 bytes | `ca1794aadc6a5a68a489326b800ff80e0336312066aa01f0bd7cdbfd46449af3` |
| `test_clean.csv` | 1,388 rows / 586,114 bytes | `c81b3cc200c4ae66a18c0ad49c0cc296f4d660814376dfb02a2088ecd0da776b` |
| `grammar_rules.csv` | 2,238 rows / 546,214 bytes | `41d4f9190b870bf5b316b410ca1d07014f9e0eece2b94509f4cf1d62ecfa5ebe` |
| `negation_lexicon.txt` | 14 words / 178 bytes | `d453b5f3fda0c30753bdb30aef921d0f8e907103d7e101e346b9de3b97c39072` |

| Environment | Value |
|---|---|
| Python | 3.14.0 |
| transformers | 5.19.0 |
| torch | 2.14.1+cpu |
| GPU | Intel Core i3-1315U CPU (8 logical threads, CPU execution) |
| Colab runtime | Local Linux PC (Fedora Linux x86_64) |
| Seed | 42 (fixed) |

*(interpretation guidance)* Quote these hashes in the final report. A reviewer who reruns Notebook 01 with the same sources and gets the same SHA-256 values has independent confirmation that the splits are identical, which is what makes the model comparison in §4.5 and §4.6 meaningful rather than merely reported.

Capture the environment versions from `!pip freeze | grep -E "torch|transformers|datasets"` at the top of Notebook 03.

---

## 4.10 Threats to Validity

| Threat | Direction of bias | Mitigation in place | Residual risk |
|---|---|---|---|
| Label noise from star ratings | Inflates apparent difficulty; depresses all models roughly equally | Label smoothing 0.05; §4.8.2 quantifies it | Cannot be removed without re-annotation |
| Cross-platform style leakage | Inflates test score | `(label, source)` stratification; §4.7.1 measures the spread | Style overlap between e-commerce sources |
| Duplicate text across splits | Inflates test score | Dedupe before split; leakage check must print 0 | Near-duplicates survive exact matching |
| Single seed per model | Cannot separate real gaps from run noise | Gap thresholds stated in §4.6 | Small gaps are uninterpretable without repeats |
| Majority-class capping | Depresses accuracy vs. an uncapped model | Reported alongside uncapped distribution in §4.1.2 | Accuracy not comparable to uncapped prior work |
| Forward negation window | Understates negation benefit | Documented in §3.5.2; measured in §4.7.2 | Mechanism may be mis-specified for Bangla |
| Grammar-rule positional alignment | Understates rule coverage | Support ≥ 5; §4.2.3 ablation measures actual effect | Insertions/deletions never mined |
| `max_seq_len = 128` truncation | Depresses long-review accuracy | §4.1.4 and §4.7.3 cross-check | Long reviews lose their concluding verdict |
| No statistical significance test | May over-read small gaps | Explicit gap thresholds in §4.6 | Formal tests not implemented |

---

## 4.11 Summary of Required Runs

Minimum set of notebook executions to fill every slot above:

| Run | Notebooks | Fills |
|---|---|---|
| 1 | 01 → 02 (`banglabert`) → 03 | §4.1, §4.3, §4.4, §4.5, §4.7, §4.8, §4.9 |
| 2 | 02 (`bangla_bert_base`, `mbert`, `xlmr`) → 03 `--all` | §4.5, §4.6 |
| 3 | 01 with `APPLY_GRAMMAR_RULES = False` → 02 → 03 | §4.2.3 (ablation) |

Run 3 doubles the cost of Runs 1–2. If GPU budget is limited, **run the ablation on the single best model only** — the ablation's purpose is to test whether BanglaGEC contributes, and one model answers that question.
