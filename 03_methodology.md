# 3. Methodology

---

## 3.1 System Overview

```
                        ┌─────────────────────────────────────────────┐
                        │            RESOURCE DATASETS                │
                        │                                             │
  banglagec ────────────┤  ~7.07M parallel pairs (wrong → right)      │
                        │           │                                 │
                        │           ▼                                 │
                        │   [B] grammar-rule mining (support ≥ 5)     │
                        │           │                                 │
                        │           ▼                                 │
                        │      grammar_rules.csv ──────────┐          │
                        │                                  │          │
  bangla-nlp-words ─────┤  stopwords / particles           │          │
                        │           │                      │          │
                        │           ▼                      │          │
                        │   [C] lexicon construction       │          │
                        │           │                      │          │
                        │           ▼                      │          │
                        │  negation_lexicon.txt ─────┐     │          │
                        │  bangla_stopwords.txt      │     │          │
                        │                            │     │          │
  bangla-nlp-catalog ───┤  813 papers / 63 datasets  │     │          │
                        │           │                │     │          │
                        │           ▼                │     │          │
                        │  literature_review_table   │     │          │
                        └────────────────────────────┼─────┼──────────┘
                                                     │     │
                        ┌────────────────────────────┼─────┼──────────┐
                        │      SUPERVISED SOURCES    ▼     ▼          │
                        │                                             │
  revbangla ────────────┤  [D] normalise  →  [E] negation scope tag   │
  largesentiment ───────┤  [F] label harmonisation → {neg,neu,pos}    │
  daraz_bd ─────────────┤  [G] length / script filter / dedupe        │
                        │  [H] stratified split on (label, source)    │
                        │           │                                 │
                        │           ▼                                 │
                        │   train_clean / val_clean / test_clean      │
                        │           │                                 │
                        │           ▼                                 │
                        │   [train] 5 transformers, one recipe        │
                        │           │                                 │
                        │           ▼                                 │
                        │   [evaluate] metrics + 3 slice analyses     │
                        └─────────────────────────────────────────────┘
```

Stage letters correspond to the stage docstrings in `src/preprocess.py` and to the sections of Notebook 01.

---

## 3.2 Dataset Construction

### 3.2.1 Canonical label scheme

Every source is mapped onto one scheme. Without this, pooling sources is meaningless.

**3-class (primary):** `{negative, neutral, positive}`

**2-class (derived):** `{negative, positive}`, where `neutral → positive`. This is reported as a secondary metric for comparability with binary benchmarks such as RevBangla, which has no neutral class.

### 3.2.2 Word-label mapping

| Source value | Canonical |
|---|---|
| `positive`, `pos`, `1`, `সকারাত্মক`, `পজিটিভ` | positive |
| `negative`, `neg`, `0`, `নেতিবাচক`, `নেগেটিভ` | negative |
| `neutral`, `neu`, `mixed`, `2`, `নিরপেক্ষ` | neutral |

Mapping is case-insensitive and whitespace-tolerant. Unmapped values are **dropped, not guessed** — a silently mis-assigned label is worse than a lost row.

### 3.2.3 Star-rating mapping

For Daraz reviews, which label by rating rather than sentiment word:

| Rating | Canonical |
|---|---|
| 1, 2 | negative |
| 3 | neutral |
| 4, 5 | positive |

This is the standard convention. It introduces known noise: a 3-star review is not reliably neutral, and 4-star reviews frequently contain substantive criticism. That noise is one reason label smoothing is used during training (§3.5.4).

### 3.2.4 Source registry

Defined once in `src/config.py → DATASETS`, with a `role` field:

| Role | Datasets | Contributes |
|---|---|---|
| `grammar` | banglagec | Normalisation rules, well-formed vocabulary |
| `lexicon` | bangla_nlp_words | Stopwords, negation lexicon |
| `catalog` | bangla_nlp_catalog | Literature-review table |
| `supervised` | revbangla, largesentiment, daraz_bd | Training rows |

### 3.2.5 Tolerant column resolution

Kaggle uploaders spell columns inconsistently (`Comment` / `comment` / `review` / `reviewText`). The loader resolves columns through an ordered candidate list, case- and whitespace-insensitively, then falls back to substring matching. A source whose columns cannot be resolved is **skipped with a printed warning**, not crashed — so one schema change upstream cannot break the whole pipeline.

---

## 3.3 Grammar-Rule Mining from BanglaGEC (Stage B)

**Why this dataset.** BanglaGEC is not a sentiment resource. It is ~7.07M pairs of *ungrammatical → grammatical* Bangla, which is the largest available record of how Bangla text goes wrong. A normaliser needs exactly that.

**Method.**

1. Load the `source` and `target` columns (resolved tolerantly).
2. Take a deterministic head sample of `BANGLAGEC_SAMPLE_ROWS = 200,000` rows. Head sampling — not random — because `load_dataset` order is stable for a given dataset revision, so the sample is reproducible without a stored seed.
3. Tokenise both sides on whitespace.
4. **Where token counts match**, align positionally and record every `source_token → target_token` substitution.
5. Keep only rules with **support ≥ 5**, so one-off annotation noise cannot become a rule.
6. Classify each surviving rule by error type.

**Rule classification** (`_classify_rule`):

| Type | Condition |
|---|---|
| `whitespace` | strings equal after removing spaces |
| `single_char_typo` | equal length, exactly one differing character |
| `affix_inflection` | one token is a substring of the other |
| `unicode_normalisation` | equal after Bangla Unicode normalisation |
| `lexical_substitution` | none of the above |

**Known limitation, stated deliberately.** Pairs whose token counts differ are skipped. Full edit-distance alignment (Levenshtein with backtrace) would recover insertions and deletions but costs O(n·m) per pair over 200k pairs — hours of runtime for marginal additional rule coverage, since single-token substitution dominates BanglaGEC edits. The trade-off is explicit: rule recall is bounded, precision and reproducibility are preserved.

**Application.** During normalisation, each whitespace token is looked up in the rule map and replaced by its target. Applied at token level only, never as substring replacement, so short rules cannot corrupt longer words.

**Well-formed vocabulary.** BanglaGEC `target` strings also yield a vocabulary of grammatically well-formed Bangla types (`wellformed_vocab.txt`), used to flag OOV-heavy rows during filtering.

---

## 3.4 Lexicon Construction (Stage C)

From `bangla-nlp-words`:

- Files whose name contains `stop` → stopwords.
- Files whose name contains `neg` → negators.
- Otherwise, short tokens (≤ 4 characters) are collected as stopword hints, since Bangla function words are characteristically short.

A seed negator list is defined in `config.py → NEGATION_SEEDS` (`না`, `নয়`, `নাই`, `নেই`, `কখনো`, `কখনই`, `আদৌ`, `নহে`, `হয় না`, …). Seeds are always present so the pipeline degrades gracefully to a working negation handler if the lexicon dataset is unavailable. A conservative default stopword set serves the same purpose.

**Critical design point:** negators are never removed as stopwords. Deleting `না` from `ভালো না` destroys the polarity signal — the exact failure this project is built to avoid.

---

## 3.5 Text Preprocessing Pipeline

### 3.5.1 Stage D — Normalisation

Applied in this order, because order affects outcome:

| Step | Operation | Rationale |
|---|---|---|
| 1 | Unicode NFC normalisation | Bangla has multiple encodings of the same visual glyph (precomposed vs. decomposed vowel signs). Without this, identical-looking words tokenise differently. |
| 2 | URL → ` URL `, @mention → ` USER ` | URLs and handles carry no sentiment but dominate token frequency. |
| 3 | Strip `#`, keep the tag text | Hashtag text is often sentiment-bearing. |
| 4 | Bangla digits `০-৯` → ASCII `0-9` | Otherwise "৫" and "5" are distinct tokens. |
| 5 | Bangla numerizer (`bnnumerizer`) | Converts written-out numbers where available. |
| 6 | Collapse character runs ≥ 3 to 2 | Handles emphasis repetition (`ভাআআলো` → `ভাআলো`) without destroying genuine reduplication. |
| 7 | `bnunicodenormalizer` | Bangla-specific canonicalisation beyond NFC. |
| 8 | Apply mined grammar rules | Stage B output. |
| 9 | Retain emoji as tokens | Emoji carry polarity in Bangla social text; discarding them loses signal. |
| 10 | Strip non-word characters except Bangla block and `#` | Removes decorative punctuation. |
| 11 | Collapse whitespace | Final tidy. |

Steps 5 and 7 have pure-Python fallbacks, so the pipeline runs on a machine where `bnunicodenormalizer` / `bnnumerizer` are unavailable.

### 3.5.2 Stage E — Negation-Scope Tagging

For each token, if it is a negator, the following `NEGATION_WINDOW = 3` tokens are prefixed with `NEG_`:

```
input :  পণ্যটি ভালো না
tokens:  [পণ্যটি] [ভালো] [না]
```

Bangla typically places the negator *after* the sentiment-bearing word, so the window alone would tag nothing useful here. The implementation therefore also benefits from the fact that Bangla frequently *pre-negates* in colloquial and code-mixed text (`না ভালো লাগেনি`), and from multi-word negators in the seed list (`হয় না`, `ছিল না`) which are matched as tokens.

> ⚠️ **Honest assessment.** Post-verbal negation is the dominant Bangla pattern, and a forward window does not capture it. A backward window (tagging the 3 tokens *before* the negator) is arguably better matched to Bangla syntax. The current implementation uses a forward window because that is the convention in the negation-handling literature (originating from English), and because changing it is a one-line edit. **Notebook 03 §4.2 measures accuracy on the negated subset specifically, which is what determines whether this component helps or is decoration.** If the negated-slice accuracy is no better than the non-negated slice, the honest conclusion is that the tagging did not help, and that belongs in the report.

`NEG_`-prefixed tokens become vocabulary entries of their own. The tokeniser treats `NEG_ভালো` as distinct from `ভালো`, so the model can learn separate embeddings for praised and negated forms.

### 3.5.3 Stage G — Filtering

| Filter | Threshold | Why |
|---|---|---|
| Minimum text length | 3 characters | Removes empty and near-empty rows |
| Maximum text length | 1000 characters | Truncation happens later at tokenisation; this bounds memory |
| Bangla script ratio | ≥ 0.4 of non-space characters | Admits genuine code-mixed text while dropping pure-English and spam rows |
| Exact duplicates | on normalised `text` | The same comment is often posted across platforms; keeping duplicates leaks between splits and inflates test scores |

Deduplication happens **before** splitting and operates on the normalised text, so two surface variants of the same comment collapse to one row.

### 3.5.4 Stage H — Splitting

| Split | Fraction | Size field |
|---|---|---|
| test | 15% | `test_size` |
| validation | 15% of total (carved from the remainder) | `val_size` |
| train | 70% | — |

**Stratified on `(label, source)` jointly.** Stratifying on label alone can still place an entire platform in one split. Stratifying on the pair forces every source to contribute proportionally to all three splits.

**Majority-class capping.** If any class exceeds `MAX_CLASS_SHARE = 0.60` of the training frame, it is downsampled to that cap. This is used *alongside* inverse-frequency class weights, not instead of them: weights alone leave the model seeing a heavily skewed gradient stream, while downsampling alone discards information. Both together give a balanced effective distribution without either problem dominating.

**Seed** is fixed at `SEED = 42` for every random operation (splitting, downsampling, training).

### 3.5.5 Derived columns

| Column | Type | Used for |
|---|---|---|
| `raw_text` | str | Original input, preserved for auditability |
| `text` | str | Normalised + negation-tagged model input |
| `label` | str | Canonical 3-class name |
| `label_id` | int | Canonical 3-class index |
| `binary_label` | str | Derived 2-class name |
| `binary_label_id` | int | Derived 2-class index |
| `source` | str | Provenance — enables per-source slice analysis |
| `has_negation` | bool | Enables the negation slice |
| `n_tokens` | int | Enables the length slice |

Keeping `source` in the training file is unusual and deliberate: it is what makes the leakage audit and per-source generalisation analysis possible.

---

## 3.6 Models

| Key | Hugging Face ID | Family | Parameters | Role |
|---|---|---|---|---|
| `banglabert` | `csebuetnlp/banglabert` | ELECTRA discriminator | ~110M | primary, Bangla-specific |
| `bangla_bert_base` | `sagorsarker/bangla-bert-base` | BERT-base | ~110M (12L/768H/12 heads) | primary, Bangla-specific |
| `banglishbert` | `csebuetnlp/banglishbert` | ELECTRA | ~110M | code-mixed robustness |
| `mbert` | `bert-base-multilingual-cased` | BERT | ~178M | multilingual baseline |
| `xlmr` | `xlm-roberta-base` | RoBERTa | ~270M | multilingual baseline |

All five are loaded through `AutoModelForSequenceClassification`, which handles the bert / roberta / electra families uniformly. A classification head (`num_labels = 3`) replaces the pretraining objective.

**Classical baseline:** TF-IDF over a `FeatureUnion` of word unigram-bigrams (60k features, sublinear TF) and character 3–5-grams (80k features), feeding multinomial Logistic Regression (`C=4.0`, `max_iter=2000`, `class_weight="balanced"`). Character n-grams are included specifically because they capture Bangla morphological suffixes and tolerate the spelling variation that dominates social-media text.

---

## 3.7 Training Protocol

**One recipe for all models.** Every setting is read from `config.py → TRAIN`, so no model receives a favourable hyperparameter. This is what makes the cross-model comparison in `docs/04_analysis.md` §3 a fair test of pretraining rather than of tuning effort.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `batch_size` | 16 | Fits a T4 (16 GB) at `max_seq_len=128` |
| `eval_batch_size` | 32 | Inference needs no gradient memory |
| `learning_rate` (backbone) | 2e-5 | Standard BERT fine-tuning range |
| `head_learning_rate` | 6e-5 | See below |
| `epochs` | 4 | With early stopping; more epochs overfits this data volume |
| `warmup_ratio` | 0.1 | Prevents early large updates destabilising pretrained weights |
| `weight_decay` | 0.01 | Standard |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `max_seq_len` | 128 | Bangla reviews are short; 128 covers the distribution while halving memory vs. 256 |
| `fp16` | on when CUDA available | ~2× throughput on T4 |
| `label_smoothing` | 0.05 | Rating-derived and crowdsourced labels are noisy |
| `early_stopping_patience` | 2 epochs | On **macro-F1**, not loss |
| `metric_for_best_model` | `macro_f1` | Aligns checkpoint selection with the reported metric |
| `seed` | 42 | Reproducibility |

### 3.7.1 Discriminative learning rates

The pretrained backbone and the randomly-initialised classification head have opposite needs: the backbone requires small updates to avoid destroying pretrained representations, the head requires large updates to learn quickly from scratch. A single learning rate compromises both.

`param_groups()` partitions parameters by top-level module name — `classifier`, `cls`, `score`, `pooler`, `dense` are treated as head — and constructs an explicit `AdamW` with two parameter groups. This covers the differing head naming across the bert / roberta / electra families.

### 3.7.2 Inverse-frequency class weighting

Weights are computed as `n_total / (n_classes × n_class)`, then **normalised to mean 1**. Normalisation matters: it keeps the effective learning rate unchanged so only the *relative* emphasis between classes shifts. Without it, changing dataset size would silently change the effective LR.

Implemented via a `Trainer` subclass overriding `compute_loss` with a weighted `CrossEntropyLoss`. The weight tensor is moved to the logits' device at loss time, so the code runs unchanged on CPU and GPU.

### 3.7.3 Early stopping on macro-F1

Loss and macro-F1 diverge under class imbalance: loss keeps improving as the model sharpens its majority-class predictions while minority-class recall collapses. Stopping on macro-F1 selects the checkpoint that is genuinely best across all three classes.

`load_best_model_at_end=True` combined with `metric_for_best_model="macro_f1"` restores that checkpoint before test evaluation.

---

## 3.8 Evaluation Protocol

### 3.8.1 Metrics

| Metric | Why it is reported |
|---|---|
| **Macro-F1** | Primary. Unweighted mean of per-class F1 — the minority class counts as much as the majority. |
| **Balanced accuracy** | Mean per-class recall; immune to class-prior skew |
| **MCC** | Correlation between true and predicted labels; the single most informative scalar for imbalanced multiclass problems |
| **Accuracy** | Reported for comparability with prior work, **not** as the primary metric |
| **Weighted-F1** | Sample-weighted; shown alongside macro-F1 so imbalance effects are visible as a gap between the two |
| **Per-class P / R / F1 / support** | Reveals which class is failing and whether it is precision or recall |
| **One-vs-rest macro AUC** | Threshold-independent ranking quality |

Accuracy is never the headline. On a 60/25/15 skewed test set a majority-class predictor scores 0.60 while learning nothing.

### 3.8.2 Slice analyses

| Slice | Question answered |
|---|---|
| **By source platform** | Does the model generalise across platforms, or learn platform style? A wide spread means it learned the platform. |
| **By negation presence** | Did negation-scope tagging actually help? |
| **By token-length bucket** (1-5, 6-10, 11-20, 21-40, 41+) | Short comments are the hard case: less context, more sarcasm, more code-mixing. |

### 3.8.3 Error analysis

`error_examples()` samples 20 misclassified rows per model with their true and predicted labels, source and negation flag, written to `results/errors_<model>.csv`. These are read by hand. Patterns in the mistakes are the substance of the Discussion — not the headline metric.

### 3.8.4 Confusion direction

The dominant off-diagonal cell is the model's systematic confusion. `neutral ↔ positive` is expected in Bangla e-commerce text because politeness markers are frequent, so mild dissatisfaction often reads as neutral-positive. An unexpected dominant confusion (e.g. `negative → positive`) indicates a polarity-inversion problem, most likely negation-related.

---

## 3.9 Reproducibility

| Mechanism | Implementation |
|---|---|
| Fixed seed | `SEED = 42` applied to splitting, downsampling and training |
| Deterministic sampling | BanglaGEC uses a head sample, not a random one, so no seed state is needed to reproduce it |
| Data manifest | `data_manifest.csv` records byte size and SHA-256 of every artefact |
| Version-controlled pipeline | All preprocessing, training and evaluation logic lives in `src/`, not only in notebooks |
| Recorded environment | `requirements.txt` with lower-bounded versions |
| No committed data | CSVs are regenerated from documented sources, not redistributed — upstream licences require download from the original host |

To verify a reproduction: run Notebook 01, then compare the SHA-256 values in `data_manifest.csv` against the values recorded in the report. Identical hashes mean identical splits.

**Note on the manifest.** Hashes are recorded in `docs/04_analysis.md` only after the pipeline has actually been executed. No hash is invented in advance.

---

## 3.10 Compute Environment

| Component | Specification |
|---|---|
| Notebook 01 (preprocessing) | Google Colab, CPU runtime |
| Notebook 02 (fine-tuning) | Google Colab, **T4 GPU** (16 GB VRAM) |
| Notebook 03 (evaluation) | Google Colab, T4 GPU recommended; CPU sufficient for the TF-IDF baseline |
| Framework | PyTorch ≥ 2.1, Transformers ≥ 4.40 |
| Per-model training budget | ~4 epochs over the harmonised training split at batch size 16 |

Notebook 02 §1.3 verifies the accelerator is actually attached before training begins. A silent CPU fallback is the most common cause of a Colab session timing out mid-run.

---

## 3.11 Ethics and Data Handling

- All corpora are public research datasets from Kaggle and Hugging Face, used under their published licences.
- Raw and pre-processed CSVs are **not** redistributed in this repository. `dataset/.gitignore` excludes them; users download from the original hosts.
- `kaggle.json` is never committed. `.gitignore` excludes it, and the notebooks install it into `~/.kaggle/` at runtime only.
- Review text may contain personal information posted publicly by its authors. The system performs aggregate sentiment classification and is not designed to profile individuals.
- Sentiment labels derived from star ratings are noisy proxies for human judgement. Reported metrics carry that noise, which is why label smoothing is applied and why confidence calibration is listed as future work.
