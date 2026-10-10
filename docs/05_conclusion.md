# 5. Conclusion

---

> ## ⚠️ Status note — read before submitting
>
> The engineering work described in this document is **complete and present in the repository**. The experimental work is **not yet executed**: no model has been trained and no metric has been measured.
>
> The execution environment used to build this repository had no Python interpreter, no Kaggle credentials and no GPU. Sections 5.2 and 5.3 below are therefore written as **protocols with result slots**, not as findings.
>
> Fill §5.2 from `docs/04_analysis.md` after running Notebooks 01 → 02 → 03, then delete this banner. Do not submit with `⚠️ NOT YET MEASURED` markers present.

---

## 5.1 Work Completed

This project set out to build a Bangla sentiment analysis system on fine-tuned BERT models. Four things were delivered:

### 5.1.1 A reproducible preprocessing pipeline

`src/preprocess.py` implements an eight-stage pipeline (B–I in the stage docstrings) that turns six heterogeneous public datasets into one canonical, labelled, split and hashed artefact set. Every stage is a version-controlled Python module rather than notebook-only code, so it is testable, diffable and reusable outside this project.

The pipeline is **tolerant by design**: column names are resolved through ordered candidate lists with substring fallback, optional Bangla-specific dependencies (`bnunicodenormalizer`, `bnnumerizer`) have pure-Python fallbacks, and an unresolvable source is skipped with a warning rather than aborting the run. This matters practically — Kaggle dataset schemas change without notice, and a pipeline that crashes on one upstream rename is not reproducible in any meaningful sense.

### 5.1.2 Three concrete uses for the specified datasets

The three datasets named in the project brief contain no sentiment labels. Rather than dropping them or forcing them into a role they cannot fill, each was given a genuine engineering function:

| Dataset | Function delivered | Artefact |
|---|---|---|
| `nahidhub/banglagec` | Grammar-rule mining: frequent source→target substitutions at support ≥ 5, classified into five error types | `grammar_rules.csv`, `wellformed_vocab.txt` |
| `saifsunny/bangla-nlp-words` | Negation lexicon and stopword construction | `negation_lexicon.txt`, `bangla_stopwords.txt` |
| `kishormorol/bangla-nlp-catalog` | Reproducible literature-review generation: 813 papers / 63 datasets / 20 models / 9 tools filtered by task | `literature_review_table.csv`, `docs/generated/prior_work.md` |

The catalog use is the most novel of the three. Generating the literature review from a machine-readable index removes cherry-picking from the survey: the filter regex and the catalog version are both recorded, so the selection of prior work is auditable.

### 5.1.3 A fair multi-model comparison harness

Five transformers (BanglaBERT, bangla-bert-base, BanglishBERT, mBERT, XLM-RoBERTa) plus a TF-IDF + Logistic Regression baseline are trained through one code path reading one configuration. No model receives a favourable hyperparameter. This is what makes the central question — *does language-specific Bangla pretraining beat multilingual pretraining?* — answerable rather than merely asked.

Four training decisions were made deliberately and are documented with their rationale in `docs/03_methodology.md` §3.7:

1. **Discriminative learning rates** (backbone 2e-5, head 6e-5) — the pretrained backbone and the randomly-initialised head have opposite needs.
2. **Inverse-frequency class weights normalised to mean 1** — corrects positive-skew without silently changing the effective learning rate.
3. **Early stopping on macro-F1, not loss** — loss keeps improving while minority-class recall collapses.
4. **Label smoothing 0.05** — star-rating-derived labels are noisy proxies for human judgement.

### 5.1.4 An evaluation protocol that exposes failure rather than hiding it

Accuracy is never the headline metric. Macro-F1, balanced accuracy, MCC, per-class precision/recall and one-vs-rest AUC are all reported, because on skewed data a majority-class predictor reaches 0.60 accuracy while learning nothing.

Three slice analyses are built into Notebook 03:

- **By source platform** — detects whether the model learned sentiment or platform style.
- **By negation presence** — measures whether negation-scope tagging actually helped.
- **By token-length bucket** — isolates short comments, the hard case in Bangla social text.

Splits are stratified on `(label, source)` **jointly**, and a leakage check asserts zero train/test text overlap. These guards exist because the most common way to get a flattering result in multi-source NLP is to let one platform land entirely in the test split.

---

## 5.2 Findings

> ⚠️ **This section cannot be written until the notebooks are run.** The slots below are the exact findings the project is designed to produce. Each names its source so transcription is mechanical.

### 5.2.1 Central question: language-specific vs. multilingual pretraining

| Measure | Value | Source |
|---|---|---|
| Best Bangla-specific model | ⚠️ NOT YET MEASURED | NB03 §3.1 |
| Best multilingual model | ⚠️ NOT YET MEASURED | NB03 §3.1 |
| Macro-F1 gap (signed) | ⚠️ NOT YET MEASURED | NB03 §3.1 |
| Verdict | ⚠️ NOT YET MEASURED | — |

*State the gap with its sign. A negative gap — multilingual winning — is a legitimate finding and must be reported as such. A gap under ~0.02 should be reported as within run-to-run noise, with no winner claimed.*

### 5.2.2 Does the transformer beat the classical baseline?

| Measure | Value |
|---|---|
| Best transformer macro-F1 | ⚠️ NOT YET MEASURED |
| TF-IDF + LogReg macro-F1 | ⚠️ NOT YET MEASURED |
| Gap | ⚠️ NOT YET MEASURED |

*If the gap is under ~0.03 macro-F1, the honest conclusion is that the transformer does not earn its compute cost on this dataset. Character n-grams are genuinely strong on morphologically rich, orthographically noisy text. Report it.*

### 5.2.3 Did BanglaGEC contribute?

| Configuration | Macro-F1 |
|---|---|
| Grammar rules applied | ⚠️ NOT YET MEASURED |
| Grammar rules not applied | ⚠️ NOT YET MEASURED |
| Δ | ⚠️ NOT YET MEASURED |

*This is the ablation that justifies the BanglaGEC dataset's presence in a sentiment project. A near-zero delta means BanglaGEC contributed methodology and a well-formed vocabulary but no measurable accuracy gain — a legitimate result. Do not omit the ablation because it is unfavourable.*

### 5.2.4 Did negation-scope tagging help?

| Measure | Value |
|---|---|
| Accuracy, negated rows | ⚠️ NOT YET MEASURED |
| Accuracy, non-negated rows | ⚠️ NOT YET MEASURED |
| Negation deficit | ⚠️ NOT YET MEASURED |
| Baseline deficit (TF-IDF) | ⚠️ NOT YET MEASURED |
| `NEG_` coverage of test set | ⚠️ NOT YET MEASURED |

*If coverage is under ~5%, the slice is underpowered and no reliable conclusion can be drawn — say so. If the deficit is similar to the baseline's, the mechanism added nothing over character n-grams.*

### 5.2.5 Generalisation across platforms

| Measure | Value |
|---|---|
| Best source accuracy | ⚠️ NOT YET MEASURED |
| Worst source accuracy | ⚠️ NOT YET MEASURED |
| Spread | ⚠️ NOT YET MEASURED |

*A spread over ~0.10 means the model learned platform style, not sentiment. This is a generalisation failure and belongs in the limitations.*

### 5.2.6 Dominant error mode

| Measure | Value |
|---|---|
| Dominant off-diagonal confusion | ⚠️ NOT YET MEASURED |
| Estimated share of "errors" that are label noise | ⚠️ NOT YET MEASURED |
| Lowest-F1 class | ⚠️ NOT YET MEASURED |

---

## 5.3 Limitations

Some limitations are structural and will remain after the experiments are run. They are stated here rather than discovered later.

**L1 — The forward negation window is probably mis-specified for Bangla.** Bangla predominantly places the negator *after* the sentiment-bearing word (`ভালো না` = *good not*). A forward window tags tokens following the negator, which captures pre-verbal negation — the English pattern — rather than the dominant Bangla one. The backward-window variant is a one-line change and is the first thing to try if §5.2.4 shows no benefit. This is documented in `docs/03_methodology.md` §3.5.2 rather than quietly left as a strength.

**L2 — Grammar-rule alignment skips length-mismatched pairs.** Positional token alignment is only valid where source and target have equal token counts. Insertions and deletions are never mined. Full edit-distance alignment would recover them but costs O(n·m) over 200k pairs — hours of runtime for marginal additional rule coverage. Rule recall is therefore bounded by design.

**L3 — Star ratings are noisy sentiment proxies.** A 3-star review is not reliably neutral, and 4-star reviews frequently contain substantive criticism. The Daraz source contributes this noise directly. Label smoothing mitigates it but cannot remove it. Only re-annotation would.

**L4 — Single-label scheme cannot represent mixed sentiment.** "Delivery was fast but the product was damaged" has no correct single label. Any model will be wrong on such rows by construction. This is a schema limitation, not a model limitation, and it points directly at aspect-based sentiment analysis as future work.

**L5 — One seed per model.** Without repeated runs, gaps under ~0.02 macro-F1 cannot be separated from run-to-run noise. §4.6 of `docs/04_analysis.md` states explicit gap thresholds so small differences are not over-read.

**L6 — Exact-match deduplication only.** Near-duplicates — the same comment with one word changed — survive and can still leak between splits. The leakage check catches exact matches only.

**L7 — Majority-class capping changes the accuracy scale.** Capping at 60% depresses accuracy relative to an uncapped model, so accuracy figures here are not directly comparable to prior work that did not cap. Macro-F1 and MCC are unaffected by this and are the metrics to compare on.

**L8 — Domain restriction.** The supervised sources are e-commerce reviews and social-media comments. Performance on news commentary, political text or literary Bangla is unmeasured and should not be assumed.

---

## 5.4 Future Work

| Direction | What it involves | Priority |
|---|---|---|
| **Backward negation window** | Change the window direction in `tag_negation()` and re-measure §5.2.4. Directly tests L1. | High — cheapest high-value experiment |
| **Aspect-based sentiment analysis** | Move beyond document-level polarity to (aspect, opinion, sentiment) triplets. Addresses L4. Resources exist: Banglish ABSA Dataset, BanglaASTE. | High |
| **Edit-distance rule alignment** | Replace positional alignment with Levenshtein backtrace over a bounded sample to recover insertions and deletions. Addresses L2. | Medium |
| **Confidence calibration** | Temperature scaling or Platt scaling so predicted probabilities are trustworthy, enabling rejection of low-confidence rows instead of forcing a label. | Medium |
| **Multi-seed variance reporting** | Three to five seeds per model, reporting mean ± standard deviation. Addresses L5. | Medium — required before claiming any small gap |
| **Near-duplicate removal** | MinHash or embedding-similarity deduplication. Addresses L6. | Medium |
| **Dialect coverage** | Regional Bangla (Chittagong, Sylhet, Noakhali, Mymensingh) is systematically absent. ANUBHUTI provides 10,000 dialect-translated sentences. | Medium |
| **Code-mixed handling** | A large share of Bangladeshi e-commerce text is romanised Banglish. `banglishbert` is in the comparison but no dedicated Banglish evaluation slice exists. | Medium |
| **Sequence-length study** | Sweep `max_seq_len` over {64, 128, 256} against the length-bucket slice to confirm 128 is not truncating review verdicts. | Low |
| **Distillation for deployment** | DistilBERT-style compression of the best model for a production API with latency constraints. | Low |
| **Human re-annotation study** | Re-label a sample of the star-rating-derived rows and measure how much of the residual error is label noise. Addresses L3. | Low — expensive but bounds the achievable ceiling |

---

## 5.5 Closing Remark

The engineering contribution of this project is separable from its empirical results, and it stands regardless of what the experiments show: a tolerant, version-controlled, hash-verified preprocessing pipeline; three specified datasets given real engineering functions rather than decorative ones; a single-recipe comparison harness across five transformers and a classical baseline; and an evaluation protocol whose slices are designed to expose failure modes rather than hide them.

What remains is to run it. Sections 5.2.1–5.2.6 are the findings the project exists to produce, and every one of them is a single notebook output away.

---

**Project by**

| ID | Name |
|---|---|
| 232-15-657 | Shoumik Bala Somu |
| 232-15-644 | Raisha Khan Raka |

**CSE432 — Natural Language Processing**
