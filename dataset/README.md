# Dataset Directory

---

## ⚠️ Why this directory is (almost) empty in Git

Both `raw/` and `preprocessed/` are **git-ignored**. Only this README, `dataset_card.md` and two `.gitkeep` files are version controlled.

Three reasons:

1. **Size.** BanglaGEC alone is ~7.07M sentence pairs — several GB. The supervised corpora add more. GitHub is not a data store and will reject or cripple a repo this size.
2. **Licence.** Each dataset is published under its own Kaggle / Hugging Face terms. Redistributing copies inside this repository would violate them. Users must download from the original host.
3. **Reproducibility is better than a snapshot.** A committed CSV can silently drift from the code that produced it. Regenerating from documented sources with a fixed seed, then comparing SHA-256 hashes, proves the data and the code agree.

---

## Regenerating everything

**On Colab** — run [`notebooks/01_data_preprocessing.ipynb`](https://colab.research.google.com/github/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models/blob/main/notebooks/01_data_preprocessing.ipynb). Section 2 walks through Kaggle credentials; section 3 downloads; section 4 runs the pipeline.

**Locally:**

```bash
pip install -r requirements.txt

# Kaggle credentials, needed for bangla-nlp-words and the three sentiment corpora.
#   kaggle.com -> avatar -> Settings -> API -> Create New Token
#   Windows: place kaggle.json at  C:\Users\<you>\.kaggle\kaggle.json
#   macOS/Linux:                     ~/.kaggle/kaggle.json

python src/download_datasets.py --max-rows 200000
python src/preprocess.py
python src/catalog_to_lr.py --top 40
```

Without Kaggle credentials, add `--skip-kaggle`. BanglaGEC and bangla-nlp-catalog still download from their public Hugging Face mirrors, so the grammar-mining and literature-review stages remain demonstrable — but there will be **no labelled sentiment data**, and `preprocess.py` will say so explicitly instead of failing quietly.

---

## Layout

```
dataset/
├── README.md                  ← this file
├── dataset_card.md            ← full schema, provenance, licences, ethics
├── raw/                       ← downloaded sources          (git-ignored)
│   ├── banglagec/
│   ├── bangla_nlp_words/
│   ├── bangla_nlp_catalog/
│   ├── revbangla/
│   ├── largesentiment/
│   └── daraz_bd/
└── preprocessed/              ← pipeline output             (git-ignored)
    ├── train_clean.csv
    ├── val_clean.csv
    ├── test_clean.csv
    ├── grammar_rules.csv
    ├── negation_lexicon.txt
    ├── bangla_stopwords.txt
    ├── wellformed_vocab.txt
    ├── literature_review_table.csv
    ├── lr_task_frequency.csv
    └── data_manifest.csv
```

---

## Output schemas

### `train_clean.csv` / `val_clean.csv` / `test_clean.csv`

| Column | Type | Description |
|---|---|---|
| `raw_text` | str | Original text as downloaded, before any normalisation |
| `text` | str | Normalised and negation-tagged model input |
| `label` | str | Canonical label: `negative` / `neutral` / `positive` |
| `label_id` | int | 0 / 1 / 2 — index into the canonical scheme |
| `binary_label` | str | Derived 2-class label: `negative` / `positive` |
| `binary_label_id` | int | 0 / 1 — present in the split files |
| `source` | str | Which dataset the row came from — enables per-source slice analysis |
| `has_negation` | bool | Whether `text` contains a `NEG_` tag |
| `n_tokens` | int | Whitespace token count — enables the length slice |

`raw_text` is retained so any normalisation decision can be audited against its input. `source` is retained so split leakage and platform shift are both measurable. Neither is used as a model feature.

### `grammar_rules.csv`

| Column | Type | Description |
|---|---|---|
| `source_token` | str | Ungrammatical token from BanglaGEC `source` |
| `target_token` | str | Corrected token from BanglaGEC `target` |
| `support` | int | Occurrences in the sample; all rows have support ≥ 5 |
| `rule_type` | str | `unicode_normalisation` / `single_char_typo` / `affix_inflection` / `whitespace` / `lexical_substitution` |
| `example` | str | One full source sentence containing the substitution, truncated to 120 chars |

Sorted by descending support.

### `data_manifest.csv`

| Column | Type | Description |
|---|---|---|
| `artifact` | str | Registry key, or `<split>[rows]` for row counts |
| `file` | str | Filename |
| `bytes` | int | File size in bytes, or row count for the `[rows]` entries |
| `sha256` | str | SHA-256 of the file, `-` for row-count entries |

This is the reproducibility record quoted in `docs/04_analysis.md` §4.9.

---

## Splits

| Split | Share | Stratified on |
|---|---|---|
| test | 15% | `(label, source)` jointly |
| validation | 15% of total, carved from the remainder | `(label, source)` jointly |
| train | 70% | — |

Stratifying on source as well as label prevents an entire platform from landing in one split. Without it, the test score measures domain shift rather than sentiment skill.

After splitting, any training class exceeding 60% of the frame is downsampled to that cap. Seed is fixed at 42 for every random operation.

---

See [`dataset_card.md`](dataset_card.md) for full provenance, label mappings, licences and the ethics statement.
