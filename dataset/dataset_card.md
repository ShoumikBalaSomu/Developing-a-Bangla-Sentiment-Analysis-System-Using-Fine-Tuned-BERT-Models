# Dataset Card — Bangla Sentiment (Harmonised)

**Project:** Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models
**Course:** CSE432 — Natural Language Processing
**Authors:** Shoumik Bala Somu (232-15-657), Raisha Khan Raka (232-15-644)

---

## 1. Summary

A harmonised Bangla sentiment corpus assembled from three labelled sources and supported by three linguistic resource datasets. All rows are mapped onto one canonical 3-class scheme, normalised through a pipeline whose grammar rules are mined from data rather than hand-written, and split with joint `(label, source)` stratification.

| Property | Value |
|---|---|
| Language | Bangla (Bengali), with code-mixed Bangla-English content retained |
| Task | Document-level sentiment classification |
| Classes | 3 primary `{negative, neutral, positive}`; 2 derived `{negative, positive}` |
| Domains | E-commerce product reviews, social-media comments |
| Size | ⚠️ Not yet measured — run Notebook 01 and record from `data_manifest.csv` |
| Splits | train 70% / val 15% / test 15% |
| Seed | 42 (all random operations) |
| Committed to Git | **No** — see §7 |

---

## 2. Composition

### 2.1 Resource datasets (no sentiment labels)

The three datasets specified in the project brief. None carries sentiment labels, so none can supervise a classifier. Each is used for a concrete engineering purpose.

| Dataset | Host | Content | Role |
|---|---|---|---|
| **BanglaGEC** | [Kaggle `nahidhub/banglagec`](https://www.kaggle.com/datasets/nahidhub/banglagec) · [HF `nahid-hub/BanglaGEC`](https://huggingface.co/datasets/nahid-hub/BanglaGEC) | Parallel GEC corpus, ~7,074,425 sentence pairs. Splits: train 5,730,275 / validation 636,702 / test 707,448 rows. | **Grammar-rule mining.** Frequent `source → target` substitutions extracted from a 200,000-row deterministic sample at support ≥ 5, classified into five error types. The `target` column also yields a well-formed Bangla vocabulary for OOV analysis. |
| **Bangla NLP words** | [Kaggle `saifsunny/bangla-nlp-words`](https://www.kaggle.com/datasets/saifsunny/bangla-nlp-words) | Compilation of essential Bangla grammatical elements — stopwords, particles, punctuation. | **Lexicon construction.** Supplies the negation lexicon that drives negation-scope tagging, plus the stopword list. |
| **Bangla NLP Catalog** | [Kaggle `kishormorol/bangla-nlp-catalog`](https://www.kaggle.com/datasets/kishormorol/bangla-nlp-catalog) · [HF mirror](https://huggingface.co/datasets/kishormorol/bangla-nlp-catalog) | Machine-readable catalog: 813 papers, 63 datasets, 20 models, 9 tools across 26 tasks, each tagged by task. | **Literature-review generation.** Filtered by `src/catalog_to_lr.py` with the regex `sentiment\|opinion\|emotion\|affect\|review` to produce the auditable prior-work tables in `docs/generated/prior_work.md`. |

### 2.2 Labelled sentiment sources (supervision)

Added because the project title requires sentiment classification and the three datasets above contain no sentiment labels.

| Dataset | Host | Content | Label field | Style |
|---|---|---|---|---|
| **RevBangla** | [Kaggle `saifsunny/revbangla`](https://www.kaggle.com/datasets/saifsunny/revbangla) | Bangla e-commerce product comments. Also on Mendeley Data (`doi:10.17632/bnbbcdsf4m`). | `Label_Sentiment` | word |
| **Large Sentiment Analysis Bangla** | [Kaggle `tahmidmir/largesentimentdata`](https://www.kaggle.com/datasets/tahmidmir/largesentimentdata) | Comments from Twitter, YouTube and Google. Columns: Platform, Comment, Sentiment, Label. | `Sentiment` | word |
| **Bangladesh Daraz Reviews** | [Kaggle `robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis`](https://www.kaggle.com/datasets/robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis) | Daraz Bangladesh product reviews, curated for Bangla NLP and ML research. | `rating` | star rating |

---

## 3. Label Harmonisation

### 3.1 Canonical scheme

**Primary (3-class):** `negative` (0) · `neutral` (1) · `positive` (2)

**Derived (2-class):** `negative` (0) · `positive` (1), obtained by mapping `neutral → positive`.

The 2-class scheme is reported only as a secondary metric, for comparability with binary benchmarks such as RevBangla. It is never reported instead of the 3-class result: folding neutral into positive is a convention that systematically inflates binary numbers relative to 3-class ones.

### 3.2 Word-label mapping

Case-insensitive, whitespace-tolerant.

| Source value | Canonical |
|---|---|
| `positive`, `pos`, `1`, `সকারাত্মক`, `পজিটিভ` | positive |
| `negative`, `neg`, `0`, `নেতিবাচক`, `নেগেটিভ` | negative |
| `neutral`, `neu`, `mixed`, `2`, `নিরপেক্ষ` | neutral |

Values that do not match are **dropped, not guessed**. A silently mis-assigned label is worse than a lost row.

### 3.3 Star-rating mapping

| Rating | Canonical |
|---|---|
| 1, 2 | negative |
| 3 | neutral |
| 4, 5 | positive |

**Known noise.** A 3-star review is not reliably neutral, and 4-star reviews frequently contain substantive criticism. This mapping introduces label noise that cannot be removed without re-annotation. It is one reason label smoothing (0.05) is applied during training, and one reason `docs/04_analysis.md` §4.8.2 asks for an estimate of what share of apparent model errors are actually label noise.

---

## 4. Preprocessing Applied

Full detail in [`../docs/03_methodology.md`](../docs/03_methodology.md) §3.5.

| Stage | Operation |
|---|---|
| Normalisation | Unicode NFC → URL/mention masking → `#` stripped, tag text kept → Bangla digits `০-৯` to ASCII → numerizer → character-run collapse (≥3 → 2) → `bnunicodenormalizer` → mined grammar rules → emoji retained as polarity tokens → non-word characters stripped except the Bangla block → whitespace collapse |
| Negation tagging | Tokens within a 3-position window after a negator receive a `NEG_` prefix |
| Filtering | length ≥ 3 and ≤ 1000 characters; Bangla script ratio ≥ 0.4 of non-space characters |
| Deduplication | exact duplicates removed on the normalised `text`, **before** splitting |
| Splitting | 15% test, then 15%-of-total validation from the remainder, stratified on `(label, source)` |
| Balancing | any training class above 60% of the frame is downsampled to that cap |

**Negators are never removed as stopwords.** Deleting `না` from `ভালো না` destroys the polarity signal — the precise failure this project is built to avoid.

**Documented limitation of the negation window.** Bangla predominantly places the negator *after* the sentiment-bearing word (`ভালো না` = *good not*). The implementation uses a forward window, which matches the English-origin convention rather than Bangla syntax. A backward window is a one-line change and is the first thing to try if the negated-slice evaluation shows no benefit. See `../docs/03_methodology.md` §3.5.2 and `../docs/05_conclusion.md` L1.

---

## 5. Schema

### 5.1 `train_clean.csv` / `val_clean.csv` / `test_clean.csv`

| Column | Type | Description | Used as feature |
|---|---|---|---|
| `raw_text` | str | Original text before normalisation | no — audit only |
| `text` | str | Normalised, negation-tagged model input | **yes** |
| `label` | str | Canonical label name | no |
| `label_id` | int | Canonical label index (0/1/2) | **yes** (target) |
| `binary_label` | str | Derived 2-class name | no |
| `binary_label_id` | int | Derived 2-class index (0/1) | **yes** (secondary target) |
| `source` | str | Originating dataset | no — slice analysis only |
| `has_negation` | bool | Whether `text` contains `NEG_` | no — slice analysis only |
| `n_tokens` | int | Whitespace token count | no — slice analysis only |

`source`, `has_negation` and `n_tokens` are retained specifically so that split leakage, negation benefit and length effects are all measurable. They are never fed to a model.

### 5.2 Other artefacts

See [`README.md`](README.md) §Output schemas for `grammar_rules.csv`, `negation_lexicon.txt`, `bangla_stopwords.txt`, `wellformed_vocab.txt`, `literature_review_table.csv` and `data_manifest.csv`.

---

## 6. Known Biases and Limitations

| Issue | Effect | Mitigation present | Residual risk |
|---|---|---|---|
| E-commerce and social-media domains only | Performance on news, political or literary Bangla is unmeasured | Stated in `docs/05_conclusion.md` L8 | Should not be assumed to transfer |
| Star-rating label noise | Depresses all models roughly equally | Label smoothing; noise share estimated in error analysis | Cannot be removed without re-annotation |
| Positive-skewed sources | Majority class dominates | 60% cap + inverse-frequency class weights | Accuracy not comparable to uncapped prior work |
| Regional dialects absent | Chittagong, Sylhet, Noakhali, Mymensingh Bangla not represented | Listed as future work (ANUBHUTI corpus) | System will underperform on dialect text |
| Code-mixed Banglish retained but not separately evaluated | Romanised Bangla behaves differently from Bangla script | `banglishbert` included in the model comparison | No dedicated Banglish slice |
| Exact-match deduplication only | Near-duplicates can survive | Leakage check asserts zero exact train/test overlap | Near-duplicate leakage unmeasured |
| Forward negation window | May not capture dominant Bangla negation pattern | Documented; measured on a dedicated slice | Mechanism may be mis-specified |
| Grammar-rule alignment skips length mismatches | Insertions and deletions never mined | Support ≥ 5 threshold; ablation measures actual effect | Rule recall bounded by design |

---

## 7. Distribution and Licensing

**No dataset files are redistributed in this repository.** `raw/` and `preprocessed/` are git-ignored.

| Reason | Detail |
|---|---|
| Size | BanglaGEC alone is several GB; GitHub is not a data store |
| Licence | Each dataset carries its own Kaggle / Hugging Face terms. Redistribution here would violate them. |
| Reproducibility | A committed CSV can drift silently from the code that produced it. Regenerating from documented sources and comparing SHA-256 hashes proves data and code agree. |

Users download from the original hosts listed in §2. `data_manifest.csv` records the SHA-256 of every generated artefact so a reproduction can be verified byte-for-byte.

Each source dataset remains under the licence published by its host. Check the licence tab on each Kaggle / Hugging Face page before any use beyond this coursework.

**Credentials.** `kaggle.json` is never committed — `.gitignore` excludes it, and the notebooks install it into `~/.kaggle/` at runtime only.

---

## 8. Ethical Considerations

- All corpora are public research datasets, used for academic coursework under their published terms.
- Review text may contain personal information posted publicly by its authors. The system performs aggregate sentiment classification and is **not** designed to profile, rank or target individuals.
- Sentiment labels derived from star ratings are noisy proxies for human judgement. Reported metrics carry that noise.
- Bangla is a low-resource language. Results obtained on this harmonised corpus should not be generalised to Bangla as a whole without domain-matched evaluation.
- Downstream deployment of a sentiment classifier on user-generated content carries moderation risk: false negatives on genuinely critical reviews silently hide customer complaints, and false positives on criticism can suppress legitimate feedback. Any deployment should report per-class recall, not accuracy alone.

---

## 9. Maintenance

| Action | Command |
|---|---|
| Re-download all sources | `python src/download_datasets.py --max-rows 200000` |
| Re-run preprocessing | `python src/preprocess.py` |
| Mine grammar rules only | Notebook 01 §5.1 |
| Regenerate the literature-review table | `python src/catalog_to_lr.py --top 40` |
| Verify reproducibility | compare `data_manifest.csv` SHA-256 values against those recorded in `docs/04_analysis.md` §4.9 |

If an upstream Kaggle dataset changes schema, `src/preprocess.py` resolves columns through ordered candidate lists with substring fallback, and skips an unresolvable source with a printed warning rather than aborting. Add the new column spelling to `TEXT_COL_CANDIDATES` or `LABEL_COL_CANDIDATES` in `src/config.py` to support it explicitly.
