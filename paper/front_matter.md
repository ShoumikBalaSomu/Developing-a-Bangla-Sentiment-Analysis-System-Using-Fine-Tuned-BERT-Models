# Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

**Course:** CSE432 — Natural Language Processing (NLP)

| ID | Name | Contribution |
|---|---|---|
| 232-15-657 | Shoumik Bala Somu | Data acquisition, preprocessing pipeline, grammar-rule mining from BanglaGEC, BERT fine-tuning |
| 232-15-644 | Raisha Khan Raka | Negation-scope handling, label harmonisation, evaluation protocol, analysis and report writing |

**Repository:** https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models

---

## Abstract

Bangla is among the most widely spoken languages in the world yet remains low-resource for NLP, with fewer annotated corpora and fewer pretrained models than high-resource languages. This project develops a Bangla sentiment analysis system by fine-tuning transformer language models on a harmonised corpus assembled from multiple public sources, and evaluates whether language-specific Bangla pretraining outperforms multilingual pretraining on the task.

Two problems specific to Bangla social-media and e-commerce text are addressed. First, polarity is routinely lost under negation: Bangla frequently places the negator after the sentiment-bearing word, and preprocessing pipelines that strip stopwords as a matter of routine delete the negator and destroy the polarity signal. Second, orthographic noise — mixed Bangla and ASCII digits, character repetition for emphasis, embedded URLs and frequent spelling errors — fragments the vocabulary a model must learn from.

The system contributes four things. (1) A reproducible preprocessing pipeline in which grammar-normalisation rules are *mined from data* rather than hand-authored: frequent source-to-target substitutions are extracted from the BanglaGEC parallel corpus (~7.07M sentence pairs) at support ≥ 5 and classified into five error types. (2) Negation-scope tagging driven by a lexicon built from the Bangla NLP words resource, whose effect is measured on a dedicated negated subset rather than assumed. (3) A cross-source label harmonisation scheme mapping word labels and star ratings onto one canonical three-class scheme, with splits stratified jointly on label *and* source to prevent platform leakage. (4) A single-recipe comparison of five transformers — BanglaBERT, bangla-bert-base, BanglishBERT, mBERT and XLM-RoBERTa — against a TF-IDF character-and-word n-gram baseline, evaluated with macro-F1, MCC and per-class recall rather than accuracy alone.

The literature review is itself generated from a machine-readable catalog of Bangla NLP resources (813 papers, 63 datasets, 20 models, 9 tools across 26 tasks), filtered by task tag, which makes the selection of prior work auditable rather than cherry-picked.

> **Status of results.** The engineering contribution is complete and present in the repository. The experimental results reported in Section 4 are **not yet measured**: the pipeline is delivered as an executable protocol with explicitly marked result slots, each naming the notebook cell that produces it. No metric in this paper is estimated or invented. Section 4.11 lists the minimum set of runs required to fill every slot.

**Keywords:** Bangla sentiment analysis, BERT fine-tuning, BanglaBERT, grammatical error correction, negation scope, low-resource NLP, label harmonisation, transformer evaluation

---

## Table of Contents

1. Introduction
2. Literature Review
3. Methodology
4. Analysis
5. Conclusion
6. References

| Section | Repository file |
|---|---|
| 1. Introduction | `docs/01_introduction.md` |
| 2. Literature Review | `docs/02_literature_review.md` |
| 3. Methodology | `docs/03_methodology.md` |
| 4. Analysis | `docs/04_analysis.md` |
| 5. Conclusion | `docs/05_conclusion.md` |
| 6. References | `docs/references.bib` |
| Dataset documentation | `dataset/README.md`, `dataset/dataset_card.md` |
| Code | `notebooks/` (Colab), `src/` (modules) |

---

## How to Run the Code

Three notebooks, each with a one-click Colab button in the repository README. Run them **in order** — Notebook 01 produces the data that Notebooks 02 and 03 consume.

| # | Notebook | Runtime | Purpose |
|---|---|---|---|
| 01 | `notebooks/01_data_preprocessing.ipynb` | CPU | Downloads all six sources, mines grammar rules from BanglaGEC, builds the negation lexicon, harmonises labels, writes `dataset/preprocessed/*.csv` |
| 02 | `notebooks/02_bert_finetuning.ipynb` | **T4 GPU** | Fine-tunes the transformer registry under one recipe, with class weighting and early stopping on macro-F1 |
| 03 | `notebooks/03_evaluation_and_analysis.ipynb` | T4 GPU recommended | Metrics, confusion matrices, ROC curves, per-source and per-negation slices, error analysis |

Notebook 01 section 2 walks through three ways to supply Kaggle credentials (file picker, form fields, or Google Drive). BanglaGEC and the NLP catalog download without credentials from public Hugging Face mirrors; the lexicon and the three labelled sentiment corpora require a token.

---
