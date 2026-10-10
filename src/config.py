"""
config.py
=========
Single source of truth for paths, dataset registry, label harmonisation and
hyper-parameters. Imported by every other module and by the Colab notebooks,
so a change here propagates everywhere.

CSE432 - Natural Language Processing
Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

Authors:
    232-15-657  Shoumik Bala Somu
    232-15-644  Raisha Khan Raka
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (work identically on Colab and on a local machine)
# --------------------------------------------------------------------------
IN_COLAB = "google.colab" in os.environ.get("COLAB_BACKEND_URL", "") or Path(
    "/content"
).exists()

if IN_COLAB:
    ROOT = Path("/content/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models")
else:
    ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
PREPROC_DIR = DATASET_DIR / "preprocessed"
RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = ROOT / "checkpoints"

for _d in (RAW_DIR, PREPROC_DIR, RESULTS_DIR, CHECKPOINT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42
# BanglaGEC has ~7.07M pairs. We mine grammar rules from a deterministic sample
# so Notebook 01 finishes in minutes instead of hours. Raise this if you want
# broader rule coverage; 200k rows already saturates the frequent-pattern set.
BANGLAGEC_SAMPLE_ROWS = 200_000

# --------------------------------------------------------------------------
# Canonical label scheme
# --------------------------------------------------------------------------
LABELS_3 = ["negative", "neutral", "positive"]
LABEL2ID = {lab: i for i, lab in enumerate(LABELS_3)}
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS_3)

# Binary scheme derived from the 3-class one (neutral folds into positive,
# which is the convention used by RevBangla-derived benchmarks).
LABELS_2 = ["negative", "positive"]
BINARY_MERGE = {"negative": "negative", "neutral": "positive", "positive": "positive"}

# Star rating -> 3-class sentiment.
RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

# --------------------------------------------------------------------------
# Dataset registry
#
# role:
#   "supervised"  -> carries sentiment labels, contributes training rows
#   "grammar"     -> BanglaGEC; used for normalisation rule mining
#   "lexicon"     -> bangla-nlp-words; stopwords + negation lexicon
#   "catalog"     -> bangla-nlp-catalog; literature-review generation
# --------------------------------------------------------------------------
DATASETS = {
    # --- The three datasets specified for this project ---------------------
    "banglagec": {
        "role": "grammar",
        "kaggle": "nahidhub/banglagec",
        "hf": "nahid-hub/BanglaGEC",
        "needs_kaggle_auth": False,          # HF mirror is public, prefer it
        "columns": {"src": "source", "tgt": "target"},
        "notes": "Parallel GEC corpus (~7.07M pairs). Mined for edit rules, not for labels.",
    },
    "bangla_nlp_words": {
        "role": "lexicon",
        "kaggle": "saifsunny/bangla-nlp-words",
        "hf": None,
        "needs_kaggle_auth": True,
        "notes": "Grammatical elements: stopwords, particles, punctuation. Supplies the negation lexicon.",
    },
    "bangla_nlp_catalog": {
        "role": "catalog",
        "kaggle": "kishormorol/bangla-nlp-catalog",
        "hf": "kishormorol/bangla-nlp-catalog",
        "needs_kaggle_auth": False,
        "notes": "813 papers / 63 datasets / 20 models / 9 tools across 26 tasks.",
    },
    # --- Labeled sentiment sources (added for supervision) -----------------
    "revbangla": {
        "role": "supervised",
        "kaggle": "saifsunny/revbangla",
        "hf": None,
        "needs_kaggle_auth": True,
        "text_col": "Comment",
        "label_col": "Label_Sentiment",
        "label_style": "words",
        "notes": "Bangla e-commerce product comments, positive/negative.",
    },
    "largesentiment": {
        "role": "supervised",
        "kaggle": "tahmidmir/largesentimentdata",
        "hf": None,
        "needs_kaggle_auth": True,
        "text_col": "Comment",
        "label_col": "Sentiment",
        "label_style": "words",
        "notes": "Twitter/YouTube/Google comments, positive/negative/neutral.",
    },
    "daraz_bd": {
        "role": "supervised",
        "kaggle": "robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis",
        "hf": None,
        "needs_kaggle_auth": True,
        "text_col": "review",
        "label_col": "rating",
        "label_style": "rating",
        "notes": "Daraz Bangladesh reviews; star rating mapped to 3 classes.",
    },
}

# Column-name candidates: Kaggle uploaders spell things inconsistently, so the
# loader tries each candidate in order instead of hard-failing.
TEXT_COL_CANDIDATES = ["Comment", "comment", "review", "Review", "reviewText",
                       "review_text", "Review_text", "text", "Text", "content", "sentence", "body"]
LABEL_COL_CANDIDATES = ["Label_Sentiment", "Sentiment", "sentiment", "Label",
                        "label", "rating", "Rating", "stars", "class"]

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
MODELS = {
    "banglabert": {
        "hf_id": "csebuetnlp/banglabert",
        "family": "electra",
        "desc": "ELECTRA discriminator pretrained on 27.5 GB Bangla corpus (Bangla2B+).",
        "primary": True,
    },
    "bangla_bert_base": {
        "hf_id": "sagorsarker/bangla-bert-base",
        "family": "bert",
        "desc": "bert-base architecture: 12 layers, 768 hidden, 12 heads, ~110M params.",
        "primary": True,
    },
    "banglishbert": {
        "hf_id": "csebuetnlp/banglishbert",
        "family": "electra",
        "desc": "ELECTRA for Bangla-English code-mixed text.",
        "primary": False,
    },
    "mbert": {
        "hf_id": "bert-base-multilingual-cased",
        "family": "bert",
        "desc": "Multilingual BERT cross-lingual baseline.",
        "primary": False,
    },
    "xlmr": {
        "hf_id": "xlm-roberta-base",
        "family": "roberta",
        "desc": "XLM-RoBERTa cross-lingual baseline.",
        "primary": False,
    },
}

# --------------------------------------------------------------------------
# Pre-processing
# --------------------------------------------------------------------------
PREPROCESS = {
    "max_seq_len": 128,
    "min_text_len": 3,          # drop rows shorter than this many characters
    "max_text_len": 1000,       # truncate anything longer before tokenisation
    "dedupe": True,
    "drop_na_text": True,
    "negation_window": 3,       # tokens after a negator that get NEG_ prefixed
    "keep_emoji_token": True,   # emoji carry polarity in Bangla social text
    "url_placeholder": " URL ",
    "mention_placeholder": " USER ",
    "test_size": 0.15,
    "val_size": 0.15,           # fraction carved out of the remaining train pool
}

# Bangla negators. Seed list; extended at runtime from the bangla-nlp-words
# lexicon when that dataset is available.
NEGATION_SEEDS = [
    "না", "নয়", "নাই", "নেই", "নি", "কখনো", "কখনই", "আদৌ",
    "নাকি", "নহে", "নাইবা", "ছিল না", "হয় না", "হয়েছে না",
]

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TRAIN = {
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "eval_batch_size": 16,
    "learning_rate": 2e-5,
    "head_learning_rate": 6e-5,      # discriminative LR for the classifier head
    "epochs": 4,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "fp16": True,
    "early_stopping_patience": 2,    # on macro-F1, not loss
    "label_smoothing": 0.05,
    "class_weight": True,            # inverse-frequency weights
    "freeze_encoder_epochs": 0,      # >0 freezes the backbone to speed up runs
    "seed": SEED,
    "save_total_limit": 2,
    "report_to": "none",
}

# Downsample the majority class when its share exceeds this ratio, so class
# weighting alone does not have to carry all the imbalance burden.
MAX_CLASS_SHARE = 0.60

# --------------------------------------------------------------------------
# Output artefact names (notebooks read these)
# --------------------------------------------------------------------------
ARTIFACTS = {
    "clean_train": PREPROC_DIR / "train_clean.csv",
    "clean_val": PREPROC_DIR / "val_clean.csv",
    "clean_test": PREPROC_DIR / "test_clean.csv",
    "grammar_rules": PREPROC_DIR / "grammar_rules.csv",
    "negation_lexicon": PREPROC_DIR / "negation_lexicon.txt",
    "stopwords": PREPROC_DIR / "bangla_stopwords.txt",
    "vocab_wellformed": PREPROC_DIR / "wellformed_vocab.txt",
    "lr_table": PREPROC_DIR / "literature_review_table.csv",
    "data_manifest": PREPROC_DIR / "data_manifest.csv",
    "metrics": RESULTS_DIR / "metrics_all_models.csv",
}


def resolve_dataset_dir(kaggle_slug: str) -> Path:
    """Kaggle CLI mounts datasets at RAW_DIR/<slug-with-dashes>. Handle both
    the slug form ('a/b') and the mounted form ('a-b')."""
    dashed = kaggle_slug.replace("/", "-")
    for cand in (RAW_DIR / dashed, RAW_DIR / kaggle_slug.split("/")[-1]):
        if cand.exists():
            return cand
    return RAW_DIR / dashed


def banner(title: str) -> str:
    """Cosmetic section separator for notebook output."""
    return f"\n{'=' * 70}\n  {title}\n{'=' * 70}\n"
