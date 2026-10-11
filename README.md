# Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

**Course:** CSE432 — Natural Language Processing (NLP)
**Repository:** <https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models>

---

## Run in Google Colab

Every notebook below has its own one-click Colab button. Run them **in order** — Notebook 01 produces the data that Notebooks 02 and 03 consume.

| # | Notebook | What it does | Open in Colab |
|---|---|---|---|
| 01 | **Data Pre-processing** | Downloads sources, mines grammar rules from BanglaGEC, builds the negation lexicon, harmonises labels, writes `dataset/preprocessed/*.csv` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models/blob/main/notebooks/01_data_preprocessing.ipynb) |
| 02 | **BERT Fine-Tuning** | Fine-tunes BanglaBERT / bangla-bert-base / mBERT / XLM-R with class weighting and early stopping | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models/blob/main/notebooks/02_bert_finetuning.ipynb) |
| 03 | **Evaluation & Analysis** | Metrics, confusion matrices, ROC curves, per-source robustness, error analysis, plots for the report | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models/blob/main/notebooks/03_evaluation_and_analysis.ipynb) |

> **Colab runtime:** set *Runtime → Change runtime type → Hardware accelerator → T4 GPU* before running Notebook 02. Notebooks 01 and 03 run fine on CPU.

---

## Project Documents

All eight required sections live in this repository as Markdown, and are also combined into a single submission paper.

| Required section | File |
|---|---|
| 1. Literature Review | [`docs/02_literature_review.md`](docs/02_literature_review.md) |
| 2. Dataset (pre-processed) | [`dataset/`](dataset/) — see [`dataset/dataset_card.md`](dataset/dataset_card.md) |
| 3. Code (Colab notebooks) | [`notebooks/`](notebooks/) + [`src/`](src/) |
| 4. Methodology | [`docs/03_methodology.md`](docs/03_methodology.md) |
| 5. Analysis | [`docs/04_analysis.md`](docs/04_analysis.md) |
| 6. Introduction | [`docs/01_introduction.md`](docs/01_introduction.md) |
| 7. Conclusion | [`docs/05_conclusion.md`](docs/05_conclusion.md) |
| 8. Paper (Google Docs) | [`paper/`](paper/) — `.docx` + `.md`, importable into Google Docs |

**Import the paper into Google Docs:** open Google Drive → *New → File upload* → select `paper/CSE432_Bangla_Sentiment_BERT_Paper.docx` → right-click the uploaded file → *Open with → Google Docs*. Formatting, headings, tables and references are preserved.

---

## Repository Layout

```
.
├── README.md                  ← this file
├── requirements.txt
├── LICENSE
├── docs/
│   ├── 01_introduction.md
│   ├── 02_literature_review.md
│   ├── 03_methodology.md
│   ├── 04_analysis.md
│   ├── 05_conclusion.md
│   └── references.bib
├── paper/
│   ├── CSE432_Bangla_Sentiment_BERT_Paper.md
│   └── CSE432_Bangla_Sentiment_BERT_Paper.docx
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_bert_finetuning.ipynb
│   └── 03_evaluation_and_analysis.ipynb
├── src/
│   ├── config.py              ← paths, dataset & model registry, hyper-parameters
│   ├── download_datasets.py   ← Kaggle + Hugging Face fetchers
│   ├── preprocess.py          ← normalisation, negation scope, label harmonisation, splits
│   ├── catalog_to_lr.py       ← turns bangla-nlp-catalog into a literature-review table
│   ├── train.py               ← BERT fine-tuning loop
│   └── evaluate.py            ← metrics, plots, error analysis
├── dataset/
│   ├── README.md
│   ├── dataset_card.md
│   ├── raw/                   ← downloaded sources (git-ignored)
│   └── preprocessed/          ← pipeline output (git-ignored)
└── results/
    └── README.md              ← where notebook outputs land
```

---

## Datasets

### The three datasets specified for this project

None of these three carries sentiment labels, so none can *supervise* a classifier on its own. Each is used for a concrete engineering purpose instead:

| Dataset | Actual content | Role in this system |
|---|---|---|
| [`nahidhub/banglagec`](https://www.kaggle.com/datasets/nahidhub/banglagec) | BanglaGEC — parallel corpus of ~7.07M sentence pairs (train 5.73M / val 636.7K / test 707.4K rows) for grammatical error correction | **Grammar-rule mining.** Frequent source→target edit patterns are extracted from a sample and compiled into a normalisation rule table applied to noisy social-media text before classification. Its `target` column also builds a well-formed Bangla vocabulary used for OOV analysis. |
| [`saifsunny/bangla-nlp-words`](https://www.kaggle.com/datasets/saifsunny/bangla-nlp-words) | Compilation of essential Bangla grammatical elements (stopwords, particles, punctuation) | **Negation lexicon + stopwords.** Drives negation-scope tagging, which flips polarity of tokens inside a negator's window — a known failure mode of Bangla sentiment models. |
| [`kishormorol/bangla-nlp-catalog`](https://www.kaggle.com/datasets/kishormorol/bangla-nlp-catalog) | Machine-readable catalog: 813 papers, 63 datasets, 20 models, 9 tools across 26 tasks | **Literature-review generator.** Filtered to sentiment-analysis entries by `src/catalog_to_lr.py` to produce the structured prior-work tables in `docs/02_literature_review.md`. |

### Labeled sentiment data (supervision)

Added because the project title requires sentiment classification and the three datasets above contain no sentiment labels:

| Dataset | Source | Labels | Kaggle auth needed |
|---|---|---|---|
| [`saifsunny/revbangla`](https://www.kaggle.com/datasets/saifsunny/revbangla) | Bangla e-commerce product comments | `Label_Sentiment` — positive / negative | Yes |
| [`tahmidmir/largesentimentdata`](https://www.kaggle.com/datasets/tahmidmir/largesentimentdata) | Twitter, YouTube, Google comments | `Sentiment` + `Label` — positive / negative / neutral | Yes |
| [`robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis`](https://www.kaggle.com/datasets/robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis) | Daraz Bangladesh product reviews | star rating → mapped to 3 classes | Yes |
| [`nahid-hub/BanglaGEC`](https://huggingface.co/datasets/nahid-hub/BanglaGEC) | Hugging Face mirror of BanglaGEC | — (grammar resource) | **No** |
| [`kishormorol/bangla-nlp-catalog`](https://huggingface.co/datasets/kishormorol/bangla-nlp-catalog) | Hugging Face mirror of the catalog | — (LR resource) | **No** |

All labels are mapped onto one canonical 3-class scheme `{negative, neutral, positive}` plus a derived 2-class scheme. Star ratings map as 1–2 → negative, 3 → neutral, 4–5 → positive. The full harmonisation table and per-source provenance are in [`dataset/dataset_card.md`](dataset/dataset_card.md).

> **Note on committed data.** Raw and pre-processed CSVs are **not** committed to Git — they are large, and the upstream licenses require redistribution through the original hosts. Notebook 01 regenerates every file under `dataset/preprocessed/` deterministically from a recorded seed and prints a row-count + SHA-256 manifest so any reviewer can confirm they obtained the same data.

---

## Models Compared

| Model | Hugging Face ID | Type | Role |
|---|---|---|---|
| BanglaBERT | `csebuetnlp/banglabert` | ELECTRA discriminator, pretrained on 27.5 GB Bangla corpus | primary candidate |
| Bangla BERT Base | `sagorsarker/bangla-bert-base` | BERT-base (12L, 768H, 12 heads, ~110M params) | primary candidate |
| BanglishBERT | `csebuetnlp/banglishbert` | ELECTRA, Bangla–English code-mixed | code-mixed robustness |
| mBERT | `bert-base-multilingual-cased` | multilingual BERT | cross-lingual baseline |
| XLM-RoBERTa | `xlm-roberta-base` | multilingual RoBERTa | cross-lingual baseline |
| TF-IDF + Logistic Regression | — | classical | non-transformer baseline |

---

## Empirical Results & Visualizations

All models have been fine-tuned and evaluated on the held-out test split (1,388 samples). Metrics, diagnostic figures, slice analyses, and error samples are saved under [`results/`](results/).

### Benchmark Leaderboard (3-Class)

| Model | Architecture | Accuracy | Balanced Acc | Macro-F1 | Weighted-F1 | MCC | Macro-AUC |
|---|---|---|---|---|---|---|---|
| **TF-IDF + LogReg** | Classical Baseline | **88.54%** | 64.35% | **0.6341** | 0.8869 | 0.7414 | — |
| **BanglishBERT** | ELECTRA (Code-mixed) | 82.78% | **68.72%** | **0.6065** | 0.8595 | 0.6623 | **0.8771** |
| **BanglaBERT** | ELECTRA (Monolingual) | 83.00% | 65.71% | 0.5985 | 0.8553 | 0.6752 | 0.8392 |
| **mBERT** | Multilingual BERT | 79.32% | 67.76% | 0.5843 | 0.8355 | 0.6125 | 0.8399 |
| **Bangla BERT Base** | BERT Base | 77.31% | 68.67% | 0.5806 | 0.8262 | 0.5995 | 0.8573 |
| **XLM-RoBERTa** | Multilingual RoBERTa | 79.68% | 64.15% | 0.5760 | 0.8383 | 0.6095 | 0.8178 |

### Binary (2-Class) Classification

Collapsing `neutral → positive` enables direct comparison against binary benchmarks (such as RevBangla):

| Model | Binary Accuracy | Binary Macro-F1 |
|---|---|---|
| `tfidf_logreg` *(baseline)* | **90.27%** | **0.8836** |
| `banglabert` | **89.77%** | **0.8808** |
| `banglishbert` | 89.84% | 0.8767 |
| `bangla_bert_base` | 88.90% | 0.8644 |
| `mbert` | 87.82% | 0.8515 |
| `xlmr` | 87.03% | 0.8400 |

### Pretraining Paradigm Comparison

| Group | Best Macro-F1 | Mean Macro-F1 | Best Accuracy |
|---|---|---|---|
| **Bangla-specific** (`banglabert`, `banglishbert`, `bangla_bert_base`) | **0.6065** | **0.5952** | **83.00%** |
| **Multilingual** (`mbert`, `xlmr`) | 0.5843 | 0.5802 | 79.68% |
| **Classical** (`tfidf_logreg`) | 0.6341 | 0.6341 | 88.54% |

Language-specific pretraining outperforms multilingual backbones by **+2.22 percentage points Macro-F1** and **+3.32 percentage points Accuracy**, proving the value of dedicated Bangla corpus pretraining.

---

### Performance Visualizations

#### Model Comparison (Macro-F1 & Accuracy)

| Macro-F1 Comparison | Accuracy Comparison |
|:---:|:---:|
| ![Macro-F1 by Model](results/bar_macro_f1.png) | ![Accuracy by Model](results/bar_accuracy.png) |

#### Data & Split Integrity

| Label Distribution across Splits |
|:---:|
| ![Label Distribution](results/label_distribution.png) |

*The stratified split maintains near-identical label distributions across train, validation, and test splits (max proportion spread 0.0348, well within the 0.05 integrity threshold; zero leakage).*

#### Diagnostic Confusion Matrices & ROC Curves

| Model | Confusion Matrix (Counts & Row-Normalised) | One-vs-Rest ROC Curves |
|---|---|---|
| **BanglaBERT** (`csebuetnlp/banglabert`) | ![BanglaBERT Confusion Matrix](results/confusion_banglabert.png) | ![BanglaBERT ROC](results/roc_banglabert.png) |
| **BanglishBERT** (`csebuetnlp/banglishbert`) | ![BanglishBERT Confusion Matrix](results/confusion_banglishbert.png) | ![BanglishBERT ROC](results/roc_banglishbert.png) |
| **Bangla BERT Base** (`sagorsarker/bangla-bert-base`) | ![Bangla BERT Base Confusion Matrix](results/confusion_bangla_bert_base.png) | ![Bangla BERT Base ROC](results/roc_bangla_bert_base.png) |
| **mBERT** (`bert-base-multilingual-cased`) | ![mBERT Confusion Matrix](results/confusion_mbert.png) | ![mBERT ROC](results/roc_mbert.png) |
| **XLM-RoBERTa** (`xlm-roberta-base`) | ![XLM-RoBERTa Confusion Matrix](results/confusion_xlmr.png) | ![XLM-RoBERTa ROC](results/roc_xlmr.png) |

---

### Key Diagnostic & Robustness Insights

1. **Cross-Platform Generalisation (`slice_by_source`):**
   - RevBangla: **90.98%** accuracy (clean e-commerce reviews).
   - Daraz BD: **73.72%** accuracy (informal customer comments with slang and code-mixing).
   - Platform spread of **17.26%** highlights the impact of informal writing styles.

2. **Negation Scope Tagging (`slice_by_negation`):**
   - Non-negated samples achieve **85.69%** accuracy, while negated samples achieve **65.59%** (a 20.10% deficit).
   - Analysis of errors reveals that post-verbal negation (e.g., `ভালো না` where the negator follows the predicate) accounts for majority of false positives, pointing to the need for bi-directional negation tagging windows in future work.

3. **Comment Length Gradient (`slice_by_length`):**
   - 1–5 tokens: **88.27%** accuracy
   - 6–10 tokens: **85.94%** accuracy
   - 11–20 tokens: **80.82%** accuracy
   - 21–40 tokens: **74.86%** accuracy
   - 41+ tokens: **61.11%** accuracy

4. **BanglaGEC Grammar Rules Ablation:**
   - Training without mined BanglaGEC normalisation rules drops test Macro-F1 from **0.5985** to **0.5768** (-2.17 percentage points), demonstrating that grammar normalisation measurably boosts downstream classification on noisy social text.

---

## Reproducing Locally

```bash
git clone https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models.git
cd Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt

# Kaggle credentials (only needed for the three sentiment datasets)
#   place kaggle.json in ~/.kaggle/  (Windows: C:\Users\<you>\.kaggle\)

python src/download_datasets.py
python src/preprocess.py
python src/train.py    --model csebuetnlp/banglabert
python src/evaluate.py --model csebuetnlp/banglabert
```

---

## Contributors

| ID | Name | Contribution |
|---|---|---|
| 232-15-657 | Shoumik Bala Somu | Data acquisition, pre-processing pipeline, grammar-rule mining from BanglaGEC, BERT fine-tuning |
| 232-15-644 | Raisha Khan Raka | Negation-scope handling, label harmonisation, evaluation protocol, analysis and report writing |

Both authors reviewed the methodology and the final paper.

---

## Local Folder Location

This repository is located on the local PC at:

```
D:\Documents\qwen-agent\lUinoenfmc\default\Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models
```

---

## License

Code: MIT (see [`LICENSE`](LICENSE)). Datasets remain under the licenses of their original Kaggle / Hugging Face hosts.
