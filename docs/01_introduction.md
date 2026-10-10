# 1. Introduction

**Course:** CSE432 — Natural Language Processing
**Project:** Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

| ID | Name |
|---|---|
| 232-15-657 | Shoumik Bala Somu |
| 232-15-644 | Raisha Khan Raka |

---

## 1.1 Background

Bangla is spoken by roughly 270 million people and ranks among the most widely spoken languages in the world, yet it remains low-resource by every measure that matters for NLP: fewer annotated corpora, fewer pretrained models, and far less evaluation infrastructure than English. The gap is not primarily one of speaker count. It is a gap in *labelled, task-specific data* and in *reproducible benchmarks*.

Sentiment analysis — determining whether a piece of text expresses a positive, negative, or neutral attitude — is one of the most commercially and socially useful NLP tasks. In the Bangladeshi context the demand is concrete: e-commerce platforms such as Daraz accumulate large volumes of Bangla product reviews, social media carries Bangla public opinion on services and policy, and businesses currently have no reliable automated way to read any of it at scale.

Transformer models changed what is achievable in low-resource settings. Rather than training a sentiment classifier from scratch on a few thousand Bangla examples — which overfits badly — one can start from a model already pretrained on a large Bangla corpus and fine-tune only the task-specific behaviour. BanglaBERT (`csebuetnlp/banglabert`), pretrained on 27.5 GB of Bangla text, and `sagorsarker/bangla-bert-base`, a bert-base-architecture model trained for one million steps on Bangla, both make this possible.

## 1.2 Problem Statement

Existing Bangla sentiment work is fragmented. Studies differ in dataset, preprocessing, label scheme, evaluation metric and split strategy, so reported accuracies are not comparable to one another. Two specific problems recur:

1. **Polarity is lost under negation.** Bangla frequently places the negator after the sentiment-bearing word — `ভালো` (*good*) versus `ভালো না` (*not good*). Preprocessing pipelines that strip stopwords as a matter of routine delete the negator and destroy the polarity signal. A model trained on such data cannot distinguish praise from criticism.
2. **Noisy, non-standard orthography.** Social-media Bangla mixes Bangla and ASCII digits, repeats characters for emphasis, embeds URLs and @mentions, and contains frequent spelling errors. The same underlying opinion can appear in dozens of surface forms, fragmenting the vocabulary the model has to learn from.

This project addresses both problems with a pipeline in which the fix is *derived from data* rather than hand-written, and reports results under a fixed, documented evaluation protocol so they are comparable across models.

## 1.3 Motivation for the Dataset Choice

The three datasets specified for this project contain no sentiment labels, which shapes how they are used:

| Dataset | What it actually is | Why it is still valuable here |
|---|---|---|
| `nahidhub/banglagec` (BanglaGEC) | ~7.07M parallel sentence pairs mapping ungrammatical → grammatical Bangla | It is the largest available corpus of *wrong → right* Bangla. That is precisely what a normaliser needs: instead of hand-writing correction rules, frequent source→target substitutions are mined directly from it. |
| `saifsunny/bangla-nlp-words` | Compilation of Bangla grammatical elements — stopwords, particles, punctuation | Supplies the negation lexicon and stopword list. The negators extracted here are what make negation-scope tagging possible, addressing problem (2) above. |
| `kishormorol/bangla-nlp-catalog` | Machine-readable catalog of 813 papers, 63 datasets, 20 models and 9 tools across 26 tasks | Makes the literature review reproducible. Filtering it for sentiment-related entries yields a date-stamped survey of the field instead of a hand-picked, therefore cherry-pickable, reference list. |

Labelled sentiment data is drawn from three supplementary corpora (RevBangla, Large Sentiment Analysis Bangla Dataset, Bangladesh Daraz Reviews). Section 1.4 states this explicitly because it is a deviation from the original brief and is documented rather than hidden.

## 1.4 Scope Deviation — Stated Explicitly

The original brief listed three datasets. On inspection, none carries sentiment labels, and one (`bangla-nlp-catalog`) is not text data at all but a metadata catalog. Fine-tuning a sentiment classifier requires labelled examples; there is no way to derive them from a grammatical-error-correction corpus without changing the task.

The project therefore uses all three specified datasets in real engineering roles — grammar-rule mining, negation lexicon, literature-review generation — and adds labelled sentiment corpora for supervision. This keeps the project title accurate: the system genuinely performs sentiment analysis, and the three specified datasets genuinely contribute to it.

## 1.5 Objectives

**Primary objective.** Build a Bangla sentiment analysis system by fine-tuning transformer models, and evaluate whether language-specific pretraining outperforms multilingual pretraining on this task.

**Specific objectives:**

1. Construct a reproducible Bangla sentiment dataset by harmonising labels from multiple heterogeneous sources into one canonical 3-class scheme `{negative, neutral, positive}`.
2. Derive a grammar-normalisation rule table from BanglaGEC and measure its effect on downstream accuracy.
3. Implement negation-scope tagging using a lexicon built from `bangla-nlp-words`, and test whether it improves accuracy on negated sentences.
4. Fine-tune BanglaBERT, bangla-bert-base, BanglishBERT, mBERT and XLM-RoBERTa under an identical recipe, plus a TF-IDF + Logistic Regression baseline.
5. Evaluate with metrics that expose minority-class behaviour (macro-F1, per-class recall, MCC) rather than accuracy alone.
6. Analyse performance by source platform, negation presence and comment length, and report the failure modes.

## 1.6 Contributions

1. **A data-derived normaliser.** Grammar correction rules are mined from BanglaGEC at support ≥ 5 rather than hand-authored, and the mined table is committed as a reproducible artefact.
2. **Negation-scope tagging for Bangla sentiment.** Tokens within a 3-position window after a negator receive a `NEG_` prefix, making the polarity contrast explicit in the token stream. Its effect is measured on a dedicated negated subset rather than assumed.
3. **A cross-source label harmonisation scheme** mapping word labels and star ratings onto one canonical scheme, with per-source provenance recorded in the dataset card.
4. **Source-stratified splitting.** Splits are stratified on `(label, source)` jointly, preventing a whole platform from landing in one split and inflating test scores.
5. **An identical-recipe multi-model comparison** including a non-transformer baseline, so the transformer numbers can be interpreted rather than read in isolation.

## 1.7 Report Organisation

| Section | Document | Content |
|---|---|---|
| 1 | `docs/01_introduction.md` | This document |
| 2 | `docs/02_literature_review.md` | Prior work, research gap, positioning |
| 3 | `docs/03_methodology.md` | Dataset construction, preprocessing pipeline, models, training protocol, evaluation metrics |
| 4 | `docs/04_analysis.md` | Results protocol, slice analyses, error analysis |
| 5 | `docs/05_conclusion.md` | Findings, limitations, future work |

Code lives in `notebooks/` (Colab-runnable) and `src/` (importable modules). The combined submission paper is in `paper/`.
