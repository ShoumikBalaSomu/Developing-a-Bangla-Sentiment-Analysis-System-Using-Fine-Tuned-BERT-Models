# 4. Analysis

---

> ## ⚠️ READ THIS FIRST — result status
>
> **This project's pipeline has not been executed yet.** No model has been trained and no metric has been measured.
>
> The execution environment that built this repository had no Python interpreter, no Kaggle credentials and no GPU, so nothing here could be run. Every table below is therefore a **result slot**: it states exactly which metric goes in which cell, which notebook produces it, and how to interpret it once it is filled.
>
> **Do not submit this document with the slots unfilled, and do not fill them with estimated or plausible numbers.** Run the three notebooks in order, then paste the actual outputs. Each slot names its source cell so the transcription is mechanical.
>
> | Marker | Meaning |
> |---|---|
> | `⚠️ NOT YET MEASURED` | A number that must come from a notebook run |
> | *(interpretation guidance)* | How to read the number once you have it — this text is final and does not change |
>
> ---
>
> **How to fill this in:**
> 1. Notebook 01 → §2 and §5 tables (dataset statistics, grammar rules)
> 2. Notebook 02 → §3 (training curves, class weights)
> 3. Notebook 03 → §3, §4, §5, §6 (all metrics, slices, errors)
> 4. Replace every `⚠️ NOT YET MEASURED` with the printed value. Delete this banner when none remain.

---

## 4.1 Data Analysis

### 4.1.1 Source composition

*Produced by Notebook 01 §3.2 and §4.2.*

| Source | Role | Rows loaded | Rows after filtering | Share of final pool |
|---|---|---|---|---|
| `revbangla` | supervised | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `largesentiment` | supervised | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `daraz_bd` | supervised | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `banglagec` | grammar | ⚠️ NOT YET MEASURED | — | — |
| `bangla_nlp_words` | lexicon | ⚠️ NOT YET MEASURED | — | — |
| `bangla_nlp_catalog` | catalog | ⚠️ NOT YET MEASURED | — | — |

### 4.1.2 Label distribution and split integrity

*Produced by Notebook 01 §4.2 and Notebook 03 §5.2.*

| Label | Train | Val | Test |
|---|---|---|---|
| negative | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| neutral | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| positive | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **total rows** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* The maximum proportion spread across splits is printed by Notebook 03 §5.2. A spread under ~0.05 confirms the stratified split held. A larger spread means stratification was defeated by a rare `(label, source)` combination having too few members to distribute — record it and note it as a limitation rather than adjusting the split to look better.

**Leakage check.** Notebook 01 §4.2 prints train/test and train/val text overlap. Both must be **0**. Any non-zero value means deduplication failed, and every metric in this document would be inflated and invalid. Fix it before proceeding.

### 4.1.3 Filtering attrition

*Produced by Notebook 01 §4.1 console output.*

| Stage | Rows remaining | Dropped | Drop reason |
|---|---|---|---|
| Raw supervised pool | ⚠️ NOT YET MEASURED | — | — |
| After label harmonisation | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | Unmappable label values |
| After length filter | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | < 3 or > 1000 chars |
| After script filter | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | Bangla ratio < 0.4 |
| After deduplication | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | Exact duplicate text |

*(interpretation guidance)* The script-filter row is the most informative. A high drop count there means a large fraction of the source corpus is English-only or romanised Banglish, which is itself a finding about Bangla e-commerce text and belongs in the discussion. Report it rather than raising the threshold to retain more data.

### 4.1.4 Text-length distribution

*Produced by Notebook 01 §4.2 (`n_tokens.describe()`).*

| Statistic | Value |
|---|---|
| mean tokens | ⚠️ NOT YET MEASURED |
| median tokens | ⚠️ NOT YET MEASURED |
| 75th percentile | ⚠️ NOT YET MEASURED |
| 95th percentile | ⚠️ NOT YET MEASURED |
| max tokens | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* Compare the 95th percentile against `max_seq_len = 128`. If it exceeds 128, the truncation is discarding real content and the sequence length should be raised and training re-run. If the median is under 15 tokens, `max_seq_len` can be lowered to 64 to roughly halve training time with negligible information loss.

---

## 4.2 Grammar-Rule Mining Analysis (BanglaGEC)

*Produced by Notebook 01 §5.1.*

### 4.2.1 Rule yield

| Measure | Value |
|---|---|
| Pairs sampled | ⚠️ NOT YET MEASURED |
| Pairs containing edits | ⚠️ NOT YET MEASURED |
| Pairs with matching token counts (alignable) | ⚠️ NOT YET MEASURED |
| Distinct substitutions observed | ⚠️ NOT YET MEASURED |
| Rules passing support ≥ 5 | ⚠️ NOT YET MEASURED |
| Edit rate (pairs with edits ÷ pairs sampled) | ⚠️ NOT YET MEASURED |

### 4.2.2 Rule type distribution

| Rule type | Count | Share |
|---|---|---|
| `unicode_normalisation` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `single_char_typo` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `affix_inflection` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `whitespace` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `lexical_substitution` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* The distribution tells you what kind of noise dominates Bangla social text:

- **`unicode_normalisation` dominant** → the corpus's main problem is encoding inconsistency. Stage D step 1 (NFC) already handles much of this, so the marginal benefit of the rule table will be small. Say so.
- **`single_char_typo` dominant** → spelling noise is the main issue, and the rule table should measurably help. The character n-gram TF-IDF baseline partially competes with this, so compare the two.
- **`affix_inflection` dominant** → morphological variation, which is what BERT subword tokenisation is already designed to absorb. The rule table may add little beyond the pretrained model.
- **`lexical_substitution` dominant** → the "corrections" are word-choice changes, not errors. Applying these as normalisation rules risks *changing meaning* and could actively harm sentiment classification. If this type dominates, disable `APPLY_GRAMMAR_RULES` and report why.

### 4.2.3 Ablation — does the rule table help?

*Requires two Notebook 01 → 02 → 03 runs: one with `APPLY_GRAMMAR_RULES = True`, one with `False`.*

| Configuration | Macro-F1 | Accuracy | MCC |
|---|---|---|---|
| Grammar rules **applied** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Grammar rules **not applied** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **Δ (applied − not applied)** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* This is the honest test of whether BanglaGEC contributes to a *sentiment* system. A positive delta justifies the dataset's inclusion. A delta near zero means BanglaGEC contributes a well-formed vocabulary and a documented methodology but no measurable accuracy gain — which is a legitimate finding and should be reported as such, not buried. **Do not omit this ablation because the result is unfavourable.** A null result is more scientifically useful than a missing one.

---

## 4.3 Lexicon Analysis (bangla-nlp-words)

*Produced by Notebook 01 §5.2 and §5.3.*

| Measure | Value |
|---|---|
| Stopwords extracted | ⚠️ NOT YET MEASURED |
| Negators in final lexicon | ⚠️ NOT YET MEASURED |
| Negators from dataset (beyond the 13 seeds) | ⚠️ NOT YET MEASURED |
| Train rows containing `NEG_` | ⚠️ NOT YET MEASURED |
| Test rows containing `NEG_` | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* The `NEG_` coverage figure is the binding constraint on §4.5.2. If fewer than ~5% of rows are negation-tagged, the negated subset is too small for a reliable accuracy comparison and any conclusion drawn from it will be noise. In that case report the coverage, state that the slice is underpowered, and widen the negation lexicon before drawing conclusions.

Note that if the count equals the seed total exactly, the `bangla-nlp-words` dataset contributed no additional negators — most likely because the Kaggle credentials were absent and the pipeline fell back to seeds. Check Notebook 01 §3.2 to confirm the dataset actually downloaded.

---

## 4.4 Literature-Review Coverage (bangla-nlp-catalog)

*Produced by Notebook 01 §5.4 and `src/catalog_to_lr.py`.*

| Measure | Value |
|---|---|
| Total catalog entries | ⚠️ NOT YET MEASURED |
| Entries matching `sentiment\|opinion\|emotion\|affect\|review` | ⚠️ NOT YET MEASURED |
| Match rate | ⚠️ NOT YET MEASURED |
| Distinct tasks represented in matches | ⚠️ NOT YET MEASURED |

### 4.4.1 Top matched tasks

*From `dataset/preprocessed/lr_task_frequency.csv`.*

| Task | Entries |
|---|---|
| ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* The match rate indicates how much of published Bangla NLP work is affective computing. The task-frequency table shows whether that work is concentrated on document-level polarity (this project's task) or distributed across aspect-based, emotion and code-mixed variants — which tells you how directly prior results can be compared to yours.

Include the generated tables from `docs/generated/prior_work.md` in the report appendix, and note the catalog version date so the survey is reproducible.

---

## 4.5 Model Performance Analysis

### 4.5.1 Overall results

*Produced by Notebook 03 §2.3. Source: `results/metrics_all_models.csv`.*

| Model | Accuracy | Balanced acc. | **Macro-F1** | Weighted-F1 | MCC | Macro AUC |
|---|---|---|---|---|---|---|
| `banglabert` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `bangla_bert_base` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `banglishbert` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `mbert` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `xlmr` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `tfidf_logreg` *(baseline)* | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | — |

**Bold the best macro-F1.** Rank by macro-F1, not accuracy — `evaluate_all()` already sorts this way.

*(interpretation guidance)* Three comparisons carry the argument:

1. **Best transformer vs. `tfidf_logreg`.** If the gap is under ~0.03 macro-F1, the transformer is not earning its compute cost on this dataset and that is the finding. Character n-grams are genuinely strong on morphologically rich, orthographically noisy text.
2. **Accuracy vs. macro-F1 within each model.** A large gap (accuracy ≫ macro-F1) means the model is trading minority-class performance for majority-class confidence. Report the gap explicitly.
3. **MCC vs. accuracy.** MCC penalises imbalanced success more severely. A model with good accuracy and weak MCC is not actually good.

### 4.5.2 Per-class breakdown

*Produced by Notebook 03 §2.3.*

| Model | P(neg) | R(neg) | F1(neg) | P(neu) | R(neu) | F1(neu) | P(pos) | R(pos) | F1(pos) |
|---|---|---|---|---|---|---|---|---|---|
| ⚠️ NOT YET MEASURED | | | | | | | | | |

*(interpretation guidance)* Read the **recall** column per class. Low negative recall means genuine criticism is being missed — the most expensive error for a business reading product reviews, because it silently hides complaints. Low neutral recall usually means the neutral class is being absorbed into positive, which is expected given that politeness markers are frequent in Bangla e-commerce text.

The `neutral` class is the hardest by construction: it has the least consistent linguistic realisation, and star-rating-derived neutral labels (rating = 3) are the noisiest. Expect its F1 to be lowest, and say why rather than treating it as a model defect.

### 4.5.3 Binary (2-class) results

*Derived by collapsing `neutral → positive`. Reported for comparability with binary benchmarks such as RevBangla.*

| Model | Binary accuracy | Binary macro-F1 |
|---|---|---|
| ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* Report this alongside the 3-class result, never instead of it. Folding neutral into positive is a convention, not a ground truth, and it systematically inflates binary numbers relative to 3-class ones. Stating both prevents an unfair comparison against prior work that used a different scheme.

---

## 4.6 Does Language-Specific Pretraining Beat Multilingual Pretraining?

*Produced by Notebook 03 §3.1. Source: `results/group_comparison.csv`.*

| Group | Models | Best macro-F1 | Mean macro-F1 | Best accuracy |
|---|---|---|---|---|
| Bangla-specific | banglabert, bangla_bert_base, banglishbert | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Multilingual | mbert, xlmr | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Classical | tfidf_logreg | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

**Macro-F1 gap (Bangla-specific best − multilingual best):** ⚠️ NOT YET MEASURED

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
| `revbangla` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `largesentiment` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `daraz_bd` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **Overall** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **Spread (max − min)** | ⚠️ NOT YET MEASURED | — |

*(interpretation guidance)* The spread is the number that matters. A spread over ~0.10 means the model learned platform-specific style — review-site politeness conventions versus social-media bluntness — rather than sentiment itself. That is a generalisation failure and belongs in the limitations, not in a footnote.

Also check **test rows** per source. A source with very few test rows has an unreliable accuracy estimate; note the count next to any figure you discuss so the reader can judge its stability.

### 4.7.2 Accuracy by negation presence

*Produced by Notebook 03 §4.2. Source: `results/slice_by_negation_<model>.csv`.*

| Subset | Accuracy | Rows |
|---|---|---|
| Rows containing `NEG_` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Rows without `NEG_` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **Deficit (non-negated − negated)** | ⚠️ NOT YET MEASURED | — |

**Same comparison for the TF-IDF baseline** (Notebook 03 §4.4):

| Subset | Baseline accuracy |
|---|---|
| Rows containing `NEG_` | ⚠️ NOT YET MEASURED |
| Rows without `NEG_` | ⚠️ NOT YET MEASURED |
| **Deficit** | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* Two questions, answered together:

1. **Is there a negation deficit?** A large deficit means polarity inversion is still being missed. In Bangla e-commerce text this is the single most costly error class, because it silently converts complaints into praise.
2. **Did tagging reduce the deficit?** Compare the transformer's deficit against the baseline's. If tagging worked, the transformer's deficit should be *smaller* than the baseline's. If the deficits are similar, the `NEG_` mechanism added nothing over what character n-grams already capture.

Note the caveat documented in `docs/03_methodology.md` §3.5.2: the implementation uses a **forward** window, whereas Bangla predominantly places the negator *after* the sentiment-bearing word. If the negated slice shows no improvement, the backward-window variant is the first thing to try, and the null result is the evidence that motivates trying it.

Check row counts before interpreting. If the negated subset is small (see §4.3 coverage), the deficit estimate is noisy and should be reported as underpowered.

### 4.7.3 Accuracy by comment length

*Produced by Notebook 03 §4.3. Source: `results/slice_by_length_<model>.csv`.*

| Token bucket | Accuracy | Rows |
|---|---|---|
| 1–5 | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| 6–10 | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| 11–20 | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| 21–40 | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| 41+ | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* Expect monotonically increasing accuracy with length. Short comments carry less context, more sarcasm and more code-mixing. A very low 1–5 bucket accuracy is a real limitation for the intended application, since short reviews are common in Bangladeshi e-commerce.

If accuracy *plateaus* or *drops* at 41+, that suggests `max_seq_len = 128` is truncating long reviews and discarding their conclusion — Bangla reviews often state the verdict at the end. Cross-check against the 95th-percentile token count in §4.1.4 and raise the sequence length if truncation is confirmed.

---

## 4.8 Error Analysis

*Produced by Notebook 03 §6.1 and §6.2. Source: `results/errors_<model>.csv`.*

### 4.8.1 Confusion matrix

*Figure: `results/confusion_<model>.png` — counts and row-normalised side by side.*

| True ↓ / Pred → | negative | neutral | positive |
|---|---|---|---|
| **negative** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **neutral** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| **positive** | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

**Dominant off-diagonal cell:** ⚠️ NOT YET MEASURED

*(interpretation guidance)* The dominant off-diagonal is the model's systematic confusion, and it should be named explicitly:

- **`neutral → positive`** — expected. Politeness markers are frequent in Bangla e-commerce text, so mild dissatisfaction reads as neutral-positive. Note it and move on.
- **`negative → positive`** — polarity inversion. Almost certainly negation-related. This is the most serious failure mode and must be investigated against §4.7.2 and the error samples below.
- **`positive → negative`** — rarer. Often sarcasm, or reviews where praise is framed as comparison against a bad alternative.

Use the **row-normalised** matrix, not the raw counts. Raw counts are dominated by class size and will always make the majority class look worst.

### 4.8.2 Qualitative error themes

Read `results/errors_<model>.csv` by hand and categorise. Suggested categories, to be filled in after reading:

| Theme | Count | Example (verbatim from errors CSV) |
|---|---|---|
| Sarcasm / irony | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Negation not captured | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Mixed sentiment (praise + criticism in one review) | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Code-mixed / romanised Bangla | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Very short text | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| Ambiguous or arguably mislabelled ground truth | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

*(interpretation guidance)* The last category matters and is usually under-reported. Some "errors" are label noise, not model failure — particularly for rows whose label came from a star rating. Estimate what fraction of the sampled errors are arguably correct predictions on mislabelled data, and report it. It bounds how much of the remaining error is actually fixable by a better model, and it justifies the label-smoothing choice in §3.7.

**Mixed sentiment** deserves specific attention: a single-label scheme cannot represent "delivery was fast but the product was damaged", so any model will be wrong on such rows by construction. That is a schema limitation, not a model limitation, and should be listed as future work (aspect-based sentiment analysis).

### 4.8.3 ROC analysis

*Figure: `results/roc_<model>.png` — one-vs-rest curves with per-class AUC.*

| Model | AUC (negative) | AUC (neutral) | AUC (positive) | Macro AUC |
|---|---|---|---|---|
| ⚠️ NOT YET MEASURED | | | | |

*(interpretation guidance)* AUC is threshold-independent, so it separates two failure modes that accuracy conflates: **poor ranking** (the model cannot tell the classes apart at any threshold) versus **poor threshold placement** (the model ranks well but the argmax decision boundary is badly positioned). If AUC is high while macro-F1 is low, the problem is calibration and class weighting, not representation — and adjusting `MAX_CLASS_SHARE` or the class weights is more productive than changing the model.

A notably lower `neutral` AUC is expected and consistent with §4.5.2.

---

## 4.9 Reproducibility Record

*Produced by Notebook 01 §6.3. Source: `dataset/preprocessed/data_manifest.csv`.*

| Artefact | Rows / bytes | SHA-256 |
|---|---|---|
| `train_clean.csv` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `val_clean.csv` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `test_clean.csv` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `grammar_rules.csv` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |
| `negation_lexicon.txt` | ⚠️ NOT YET MEASURED | ⚠️ NOT YET MEASURED |

| Environment | Value |
|---|---|
| Python | ⚠️ NOT YET MEASURED |
| transformers | ⚠️ NOT YET MEASURED |
| torch | ⚠️ NOT YET MEASURED |
| GPU | ⚠️ NOT YET MEASURED |
| Colab runtime | ⚠️ NOT YET MEASURED |
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
