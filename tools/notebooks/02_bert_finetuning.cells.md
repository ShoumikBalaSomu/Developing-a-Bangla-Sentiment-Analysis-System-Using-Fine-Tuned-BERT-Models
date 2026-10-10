###CELL md###
# Notebook 02 — BERT Fine-Tuning

**CSE432 — Natural Language Processing**
**Project:** Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

| ID | Name |
|---|---|
| 232-15-657 | Shoumik Bala Somu |
| 232-15-644 | Raisha Khan Raka |

---

## Runtime

**A GPU is required.** `Runtime → Change runtime type → Hardware accelerator → T4 GPU`.

Section 1.3 verifies the accelerator is actually attached before any training starts, because a silent CPU fallback turns a 20-minute run into a 6-hour one.

## Prerequisite

`dataset/preprocessed/{train,val,test}_clean.csv` must exist. Run **Notebook 01** first, or restore the artefacts from Drive in section 1.4.

## Models

| Key | Hugging Face ID | Why it is in the comparison |
|---|---|---|
| `banglabert` | `csebuetnlp/banglabert` | ELECTRA discriminator pretrained on 27.5 GB Bangla — language-specific |
| `bangla_bert_base` | `sagorsarker/bangla-bert-base` | BERT-base (12L/768H/12 heads, ~110M) — language-specific |
| `banglishbert` | `csebuetnlp/banglishbert` | Handles Bangla–English code-mixing |
| `mbert` | `bert-base-multilingual-cased` | Cross-lingual baseline |
| `xlmr` | `xlm-roberta-base` | Cross-lingual baseline |

The point of the comparison is not "which number is biggest". It is whether **language-specific pretraining beats multilingual pretraining** on Bangla sentiment, and whether that gap is worth the extra engineering.
###CELL code###
#@title 1.1 Clone the repository
import os, sys
from pathlib import Path

REPO_URL = "https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models.git"
REPO_NAME = REPO_URL.rstrip("/").split("/")[-1].replace(".git", "")

IN_COLAB = Path("/content").exists()
ROOT = Path("/content") / REPO_NAME if IN_COLAB else Path.cwd().parent

if IN_COLAB and not ROOT.exists():
    os.system(f"git clone --depth 1 {REPO_URL} {ROOT}")

if not (ROOT / "src" / "config.py").exists():
    raise SystemExit(f"repo not found at {ROOT}")

os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
print("repo root:", ROOT)
###CELL code###
#@title 1.2 Install dependencies
if IN_COLAB:
    %pip install -q -r requirements.txt
    %pip install -q bnunicodenormalizer bnnumerizer
###CELL code###
#@title 1.3 Verify the GPU is actually attached
import torch

print("torch        :", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device       :", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram         : {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")
else:
    print("\nNO GPU. Set Runtime -> Change runtime type -> T4 GPU.")
    print("Training on CPU is possible but roughly 20x slower; use SUBSET below to smoke-test.")
###CELL code###
#@title 1.4 (Optional) Restore pre-processed data from Google Drive
LOAD_FROM_DRIVE = False  #@param {type:"boolean"}
drive_dir = "/content/drive/MyDrive/CSE432_bangla_sentiment"  #@param {type:"string"}

if LOAD_FROM_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    from config import PREPROC_DIR
    PREPROC_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in Path(drive_dir).glob("*"):
        shutil.copy(f, PREPROC_DIR / f.name)
        n += 1
    print(f"restored {n} files into {PREPROC_DIR}")
###CELL code###
#@title 1.5 Confirm the training data is present
import pandas as pd
from config import ARTIFACTS

missing = [ARTIFACTS[k] for k in ("clean_train", "clean_val", "clean_test")
           if not Path(ARTIFACTS[k]).exists()]
if missing:
    raise SystemExit("Missing splits:\n  " + "\n  ".join(str(m) for m in missing)
                     + "\nRun Notebook 01 (or section 1.4 above) first.")

train = pd.read_csv(ARTIFACTS["clean_train"])
val = pd.read_csv(ARTIFACTS["clean_val"])
test = pd.read_csv(ARTIFACTS["clean_test"])
print(f"train={len(train):,}  val={len(val):,}  test={len(test):,}")
print("\ntrain label distribution:")
print(train["label"].value_counts().to_string())
print("\nrows per source:")
print(train["source"].value_counts().to_string())
###CELL md###
---
## 2. Training configuration

Defaults come from `src/config.py → TRAIN`. The four settings that matter most for this dataset:

1. **Discriminative learning rates.** The pretrained backbone gets `2e-5`; the randomly-initialised classifier head gets `6e-5`. A single LR either under-trains the head or destabilises the backbone.
2. **Inverse-frequency class weights.** Bangla review corpora are strongly positive-skewed. Unweighted cross-entropy collapses to predicting the majority class and still reports a respectable accuracy.
3. **Early stopping on macro-F1, not loss.** Loss keeps falling while the minority class degrades — exactly the failure mode that hides behind a good balanced-accuracy headline.
4. **Label smoothing 0.05.** Crowdsourced and lexicon-derived sentiment labels are noisy; smoothing prevents over-confident fits to them.
###CELL code###
#@title 2.1 Hyper-parameters (edit here, or leave the config defaults)
from config import TRAIN, PREPROCESS, MODELS

EPOCHS = 4          #@param {type:"integer"}
BATCH_SIZE = 16     #@param {type:"integer"}
LEARNING_RATE = 2e-5  #@param {type:"number"}
MAX_SEQ_LEN = 128   #@param {type:"integer"}
USE_CLASS_WEIGHTS = True  #@param {type:"boolean"}
SUBSET = 0          #@param {type:"integer"}
#@markdown `SUBSET > 0` trains on only that many rows. Use 2000 to smoke-test the whole
#@markdown pipeline end-to-end in a couple of minutes before committing GPU hours.

TRAIN["epochs"] = EPOCHS
TRAIN["batch_size"] = BATCH_SIZE
TRAIN["learning_rate"] = LEARNING_RATE
TRAIN["class_weight"] = USE_CLASS_WEIGHTS
PREPROCESS["max_seq_len"] = MAX_SEQ_LEN

print("epochs      :", TRAIN["epochs"])
print("batch size  :", TRAIN["batch_size"])
print("lr (backbone):", TRAIN["learning_rate"])
print("lr (head)   :", TRAIN["head_learning_rate"])
print("max_seq_len :", PREPROCESS["max_seq_len"])
print("class weights:", TRAIN["class_weight"])
print("label smooth :", TRAIN["label_smoothing"])
print("early stop   :", TRAIN["early_stopping_patience"], "epochs on macro_f1")
print("subset       :", SUBSET or "full data")
###CELL md###
---
## 3. Train

Run 3.1 for a single model, or 3.2 to sweep the whole registry.

**Colab session limits:** a free T4 session disconnects after a period of inactivity and the runtime's disk is wiped. Checkpoints are saved to `checkpoints/<model>/best` after each run, and 3.3 copies them to Drive so a disconnect does not lose the work.
###CELL code###
#@title 3.1 Train ONE model
MODEL_KEY = "banglabert"  #@param ["banglabert","bangla_bert_base","banglishbert","mbert","xlmr"]

import importlib, train as train_mod
importlib.reload(train_mod)
from train import train_one, append_metrics

flat = train_one(MODEL_KEY, epochs=EPOCHS, subset=SUBSET or None,
                 use_weights=USE_CLASS_WEIGHTS)
append_metrics(flat)
###CELL code###
#@title 3.2 Or sweep the whole registry
RUN_ALL = False  #@param {type:"boolean"}
ONLY_PRIMARY = True  #@param {type:"boolean"}
#@markdown `ONLY_PRIMARY` restricts the sweep to the two language-specific models
#@markdown (BanglaBERT + bangla-bert-base), which is the comparison the report
#@markdown actually argues. Set it False to include the multilingual baselines.

if RUN_ALL:
    keys = [k for k, v in MODELS.items() if v["primary"]] if ONLY_PRIMARY else list(MODELS)
    results = []
    for k in keys:
        print(f"\n\n{'#' * 70}\n#  {k}\n{'#' * 70}")
        try:
            flat = train_one(k, epochs=EPOCHS, subset=SUBSET or None,
                             use_weights=USE_CLASS_WEIGHTS)
            append_metrics(flat)
            results.append(flat)
        except Exception as exc:
            print(f"[!] {k} failed: {type(exc).__name__}: {str(exc)[:300]}")
    summary = pd.DataFrame(results)
    if not summary.empty:
        cols = [c for c in ("model_key", "test_accuracy", "test_macro_f1",
                            "test_weighted_f1") if c in summary.columns]
        print("\n=== sweep summary ===")
        print(summary[cols].to_string(index=False))
else:
    print("skipped (set RUN_ALL = True)")
###CELL code###
#@title 3.3 Save checkpoints + metrics to Drive
SAVE_CHECKPOINTS_TO_DRIVE = True  #@param {type:"boolean"}

from config import CHECKPOINT_DIR, RESULTS_DIR
import shutil

if SAVE_CHECKPOINTS_TO_DRIVE and IN_COLAB:
    from google.colab import drive
    try:
        drive.mount("/content/drive")
    except Exception as exc:
        print("mount failed:", exc)
    dest = Path("/content/drive/MyDrive/CSE432_bangla_sentiment/checkpoints")
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in CHECKPOINT_DIR.rglob("*"):
        if src.is_file() and src.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}:
            rel = src.relative_to(CHECKPOINT_DIR)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
            copied += 1
    print(f"copied {copied} checkpoint files to {dest}")
    if Path(ARTIFACTS["metrics"]).exists():
        shutil.copy(ARTIFACTS["metrics"], dest.parent / "metrics_all_models.csv")
        print("copied metrics_all_models.csv")
else:
    print("skipped")
###CELL code###
#@title 3.4 Show accumulated metrics
if Path(ARTIFACTS["metrics"]).exists():
    m = pd.read_csv(ARTIFACTS["metrics"])
    cols = [c for c in ("model_key", "hf_id", "epochs", "n_train", "test_accuracy",
                        "test_macro_f1", "test_weighted_f1") if c in m.columns]
    with pd.option_context("display.max_columns", 20, "display.width", 160):
        print(m[cols].to_string(index=False))
else:
    print("no metrics yet")
###CELL md###
---
## 4. Quick inference demo

Sanity-check the best checkpoint on a few Bangla sentences. Watch the **negation** pair specifically: `ভালো` (good) versus `ভালো না` (not good) must land in different classes. If they do not, the negation-scope tagging from Notebook 01 is not pulling its weight and that belongs in the error analysis.
###CELL code###
#@title 4.1 Predict on custom sentences
DEMO_MODEL = "banglabert"  #@param {type:"string"}

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import ID2LABEL

ckpt = CHECKPOINT_DIR / DEMO_MODEL.replace("/", "__") / "best"
if not ckpt.exists():
    print(f"no checkpoint at {ckpt} - train it in section 3 first")
else:
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    model.eval()

    samples = [
        "পণ্যটি অসাধারণ, খুবই ভালো লেগেছে",
        "পণ্যটি একেবারেই ভালো না",
        "দাম অনুযায়ী মান খুব একটা ভালো নয়",
        "ডেলিভারি দ্রুত হয়েছে কিন্তু প্যাকেজিং নষ্ট ছিল",
        "সেবা মোটামুটি, বিশেষ কিছু বলার নেই",
    ]
    enc = tokenizer(samples, truncation=True, max_length=MAX_SEQ_LEN,
                    padding=True, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).numpy()

    print(f"{'text':<48} {'predicted':<10} confidence")
    print("-" * 78)
    for text, row in zip(samples, probs):
        idx = int(row.argmax())
        shown = text if len(text) <= 46 else text[:43] + "..."
        print(f"{shown:<48} {ID2LABEL[idx]:<10} {row[idx]:.3f}")
###CELL md###
---
## Done — next step

Open **`notebooks/03_evaluation_and_analysis.ipynb`**. It loads the checkpoints saved here and produces every figure and table used in `docs/04_analysis.md`.
