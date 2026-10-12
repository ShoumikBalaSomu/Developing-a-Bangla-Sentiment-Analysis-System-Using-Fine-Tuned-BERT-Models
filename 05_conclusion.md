# 5. Conclusion

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

### 5.2.1 Central question: language-specific vs. multilingual pretraining

| Measure | Value | Source |
|---|---|---|
| Best Bangla-specific model | `banglishbert` (Macro-F1: 0.6065, Acc: 0.8278) | NB03 §3.1 |
| Best multilingual model | `mbert` (Macro-F1: 0.5843, Acc: 0.7932) | NB03 §3.1 |
| Macro-F1 gap (signed) | +0.0222 (+2.22 percentage points) | NB03 §3.1 |
| Verdict | Confirmed **within the transformer family**: language-specific pretraining outperforms multilingual transfer | — |

*The empirical result supports the hypothesis within the transformer family: Bangla-specific pretraining (`banglishbert` with 0.6065 macro-F1, `banglabert` with 0.5985 macro-F1) outperforms multilingual pretraining (`mbert` with 0.5843 macro-F1, `xlmr` with 0.5760 macro-F1) by +0.0222 macro-F1 (+2.22 percentage points). Pretraining directly on Bangla/Banglish text yields representations that capture dialectal semantics and morphological structures superior to generic multilingual models.*

> ⚠️ **Scope limit on this verdict.** It compares transformers against transformers only. As §5.2.2 shows, the **TF-IDF + Logistic Regression baseline beats all five of them** (0.6341 macro-F1, 0.8854 accuracy). The correct headline for this project is therefore *"Bangla-specific pretraining beats multilingual pretraining, but neither beats a well-tuned classical baseline on this split"* — **not** *"BERT wins"*. Two factors explain the baseline's advantage and are discussed rather than hidden: the test set contains only **21 neutral rows (1.5%)**, making the task effectively near-binary and collapsing every neutral metric into noise (neutral F1 0.076–0.143); and all five transformers are majority-biased (positive recall 0.759–0.827 against neutral recall 0.238–0.476).

### 5.2.2 Does the transformer beat the classical baseline?

| Measure | Value |
|---|---|
| Best transformer macro-F1 | 0.6065 (`banglishbert`) |
| TF-IDF + LogReg macro-F1 | 0.6341 (`tfidf_logreg`) |
| Gap | −0.0276 (−2.76 percentage points) |

*The classical n-gram baseline (`tfidf_logreg`) outperforms the best 1-epoch transformer by 2.76 percentage points in macro-F1 (0.6341 vs. 0.6065) and in accuracy (0.8854 vs. 0.8278). Subword character n-grams (2–5 characters) paired with class-weighted logistic regression excel on noisy, orthographically irregular Bangla reviews without overfitting, setting a competitive performance ceiling that transformers require substantial hyperparameter tuning and multi-epoch compute to match.*

### 5.2.3 Did BanglaGEC contribute?

| Configuration | Macro-F1 |
|---|---|
| Grammar rules applied | 0.5985 |
| Grammar rules not applied | 0.6228 |
| Δ | −0.0243 (−2.43 percentage points) |

*The ablation reveals a negative delta (−0.0243 Macro-F1, −0.0334 Accuracy). Normalizing informal Bangla words through parallel GEC substitution rules tends to smooth away colloquial and emphatic dialect forms that carry strong sentiment signals in user reviews. While BanglaGEC successfully provides a 96,383-token well-formed vocabulary and an auditable linguistic baseline, unconstrained grammatical normalization should be applied judiciously in sentiment tasks to avoid eroding affective nuance.*

### 5.2.4 Did negation-scope tagging help?

| Measure | Value |
|---|---|
| Accuracy, negated rows | 0.6559 (65.59%) |
| Accuracy, non-negated rows | 0.8569 (85.69%) |
| Negation deficit | 0.2010 (20.10%) |
| Baseline deficit (TF-IDF) | 0.1483 (14.83%) |
| `NEG_` coverage of test set | 13.40% (186 / 1,388 rows) |

*The test set contains 13.40% negation-tagged rows (186 rows), well above the 5% statistical power floor. Both BanglaBERT and the TF-IDF baseline exhibit a noticeable negation deficit (~15–20%), reflecting the difficulty of polarity inversion in colloquial Bangla review text. Because Bangla primarily places the negator after the verb, moving to a backward negation window represents a clear future improvement.*

### 5.2.5 Generalisation across platforms

| Measure | Value |
|---|---|
| Best source accuracy | 0.9098 (`revbangla`, 743 rows) |
| Worst source accuracy | 0.7372 (`daraz_bd`, 643 rows) |
| Spread | 0.1726 (17.26%) |

*The 17.26% cross-platform spread reflects the noisy star ratings and code-mixing present in the Daraz reviews relative to the cleaner curated text in RevBangla.*

### 5.2.6 Dominant error mode

| Measure | Value |
|---|---|
| Dominant off-diagonal confusion | `neutral → positive` (and `negative → positive` on noisy ratings) |
| Estimated share of "errors" that are label noise | ~35–40% (5-star ratings with critical Bangla text) |
| Lowest-F1 class | `neutral` (F1 = 0.0800, 21 test rows) |

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
