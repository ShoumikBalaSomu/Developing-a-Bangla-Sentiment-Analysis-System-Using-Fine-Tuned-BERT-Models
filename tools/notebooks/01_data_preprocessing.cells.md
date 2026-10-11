###CELL md###
# Notebook 01 — Data Pre-processing

**CSE432 — Natural Language Processing**
**Project:** Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models

| ID | Name |
|---|---|
| 232-15-657 | Shoumik Bala Somu |
| 232-15-644 | Raisha Khan Raka |

---

## What this notebook produces

Everything under `dataset/preprocessed/` that Notebooks 02 and 03 consume:

| File | Contents |
|---|---|
| `train_clean.csv` / `val_clean.csv` / `test_clean.csv` | Canonical 3-class sentiment splits |
| `grammar_rules.csv` | Substitution rules mined from **BanglaGEC** |
| `negation_lexicon.txt` | Negators from **bangla-nlp-words** |
| `bangla_stopwords.txt` | Stopwords from **bangla-nlp-words** |
| `wellformed_vocab.txt` | Well-formed Bangla vocabulary from BanglaGEC targets |
| `literature_review_table.csv` | Sentiment entries filtered from **bangla-nlp-catalog** |
| `data_manifest.csv` | Row counts + SHA-256 so anyone can verify they got the same data |

## Runtime

**CPU is enough** for this notebook: `Runtime → Change runtime type → CPU`.

## Kaggle credentials

BanglaGEC and bangla-nlp-catalog download without credentials from public Hugging Face mirrors. **bangla-nlp-words** and the labelled sentiment corpora need a Kaggle API token. Section 2 handles that — skip it and the notebook still completes, using the seed negation lexicon and reporting which sources were unavailable.
###CELL code###
#@title 1.1 Clone the repository and set up paths
import os, sys
from pathlib import Path

REPO_URL = "https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models.git"
REPO_NAME = REPO_URL.rstrip("/").split("/")[-1].replace(".git", "")

IN_COLAB = Path("/content").exists()
ROOT = Path("/content") / REPO_NAME if IN_COLAB else (Path.cwd() if (Path.cwd() / "src" / "config.py").exists() else Path.cwd().parent)

if IN_COLAB and not ROOT.exists():
    os.system(f"git clone --depth 1 {REPO_URL} {ROOT}")

if not (ROOT / "src" / "config.py").exists():
    raise SystemExit(f"repo not found at {ROOT}. Check the clone above.")

os.chdir(ROOT)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

print("working dir:", os.getcwd())
print("repo root  :", ROOT)
###CELL code###
#@title 1.2 Install dependencies
if IN_COLAB:
    %pip install -q -r requirements.txt
    # Bangla-specific normalisers; the pipeline has pure-Python fallbacks
    %pip install -q bnunicodenormalizer bnnumerizer
else:
    print("local run - install with: pip install -r requirements.txt")
###CELL code###
#@title 1.3 (Optional) Mount Google Drive so outputs survive the session
SAVE_TO_DRIVE = False  #@param {type:"boolean"}

OUT = None
if SAVE_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/CSE432_bangla_sentiment")
    OUT.mkdir(parents=True, exist_ok=True)
    print("outputs will be copied to", OUT)
else:
    print("Drive not mounted. Download outputs from section 6 before the session ends.")
###CELL md###
---
## 2. Kaggle credentials

Only needed for `saifsunny/bangla-nlp-words` and the labelled sentiment corpora.

**How to get `kaggle.json`:** kaggle.com → your avatar → *Settings* → *API* → *Create New Token*. A `kaggle.json` file downloads.

Pick **one** of the three options below. The token stays in this runtime and is never written into the repository.
###CELL code###
#@title 2.0 Helper — installs a kaggle.json with the permissions Kaggle requires
def _install_kaggle_json(src):
    """Copy kaggle.json into ~/.kaggle/ with mode 600, else Kaggle warns."""
    import json, shutil, stat
    src = Path(src)
    try:
        json.loads(src.read_text())          # fail fast on a malformed file
    except Exception as exc:
        print("not valid JSON:", exc)
        return False
    dest = Path.home() / ".kaggle"
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "kaggle.json"
    shutil.copy(src, target)
    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    os.environ["KAGGLE_CONFIG_DIR"] = str(dest)
    print("installed ->", target)
    return True
###CELL code###
#@title 2.1 Option A — upload kaggle.json through the file picker (easiest)
USE_FILE_PICKER = False  #@param {type:"boolean"}

if USE_FILE_PICKER and IN_COLAB:
    from google.colab import files
    uploaded = files.upload()
    for name, blob in uploaded.items():
        if name.endswith("kaggle.json"):
            tmp = Path("/content") / name
            tmp.write_bytes(blob)
            _install_kaggle_json(tmp)
elif USE_FILE_PICKER:
    print("file picker is Colab-only")
else:
    print("skipped (set USE_FILE_PICKER = True to use this option)")
###CELL code###
#@title 2.2 Option B — type the token into Colab form fields
USE_FORM = False  #@param {type:"boolean"}
kaggle_username = ""  #@param {type:"string"}
kaggle_key = ""  #@param {type:"string"}

if USE_FORM:
    if kaggle_username and kaggle_key:
        import json
        dest = Path.home() / ".kaggle"
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / "kaggle.json"
        target.write_text(json.dumps({"username": kaggle_username, "key": kaggle_key}))
        os.chmod(target, 0o600)
        os.environ["KAGGLE_CONFIG_DIR"] = str(dest)
        print("installed ->", target)
    else:
        print("fill in both fields first, then re-run this cell")
else:
    print("skipped (set USE_FORM = True to use this option)")
###CELL code###
#@title 2.3 Option C — read kaggle.json from Google Drive
USE_DRIVE_JSON = False  #@param {type:"boolean"}
drive_json_path = "/content/drive/MyDrive/kaggle.json"  #@param {type:"string"}

if USE_DRIVE_JSON:
    p = Path(drive_json_path)
    if p.exists():
        _install_kaggle_json(p)
    else:
        print("not found:", p, "- mount Drive in 1.3 and check the path")
else:
    print("skipped (set USE_DRIVE_JSON = True to use this option)")
###CELL code###
#@title 2.4 Report credential state and what it implies
have_creds = (Path.home() / ".kaggle" / "kaggle.json").exists()
print("Kaggle credentials:", "AVAILABLE" if have_creds else "ABSENT")
if not have_creds:
    print("  -> BanglaGEC + bangla-nlp-catalog still download (public HF mirrors).")
    print("  -> bangla-nlp-words and the labelled sentiment sources are skipped, so")
    print("     Notebooks 02/03 will have no training data until you add a token.")
###CELL md###
---
## 3. Download the datasets

Two channels per source: Hugging Face first (no credentials), Kaggle second.

BanglaGEC is ~7.07M sentence pairs — several GB. `MAX_ROWS` caps each split so the download finishes in minutes. 200k rows already saturate the frequent grammar patterns we mine; raise it for broader rule coverage at the cost of runtime.
###CELL code###
#@title 3.1 Download everything
MAX_ROWS = 200000  #@param {type:"integer"}
SKIP_KAGGLE = False  #@param {type:"boolean"}

args = f"--max-rows {MAX_ROWS}"
if SKIP_KAGGLE:
    args += " --skip-kaggle"

os.system(f"{sys.executable} src/download_datasets.py {args}")
###CELL code###
#@title 3.2 What actually arrived
import pandas as pd
from config import DATASETS, RAW_DIR

rows = []
for name, spec in DATASETS.items():
    d = RAW_DIR / name
    files = sorted(d.rglob("*")) if d.exists() else []
    files = [f for f in files if f.is_file()]
    csvs = [f for f in files if f.suffix == ".csv"]
    size = sum(f.stat().st_size for f in files)
    rows.append({
        "dataset": name, "role": spec["role"], "csvs": len(csvs),
        "MB": round(size / 1e6, 1), "status": "OK" if csvs else "MISSING",
    })
status = pd.DataFrame(rows)
print(status.to_string(index=False))

missing = status.loc[status.status == "MISSING", "dataset"].tolist()
if missing:
    print("\nMISSING:", missing)
    if "banglagec" in missing:
        print("  -> grammar-rule mining will be skipped; the rest of the pipeline still runs.")
    if {"revbangla", "largesentiment", "daraz_bd"} & set(missing):
        print("  -> a labelled sentiment source is missing. Notebooks 02/03 need at least")
        print("     one. Add Kaggle credentials in section 2 and re-run 3.1.")
###CELL md###
---
## 4. Run the pre-processing pipeline

Stages as implemented in `src/preprocess.py`:

| Stage | What happens | Dataset that drives it |
|---|---|---|
| B | Mine source→target substitutions from BanglaGEC, keep rules with support ≥ 5 | **banglagec** |
| C | Build stopwords + negation lexicon | **bangla-nlp-words** |
| D | Unicode normalisation, Bangla→ASCII digits, URL/mention masking, character-run collapse, emoji retained as polarity tokens | — |
| E | Prefix `NEG_` on tokens inside a 3-token window after a negator | **bangla-nlp-words** |
| F | Harmonise every source label to `{negative, neutral, positive}` | all supervised |
| G | Length filter, Bangla-script ratio ≥ 0.4, exact-duplicate removal | **banglagec** |
| H | Stratified split on `(label, source)`; majority class capped at 60% | — |
| I | Row counts + SHA-256 manifest | — |

Stage H matters more than it looks. Stratifying on **source** as well as label stops one platform from landing entirely in the test split — otherwise the test score measures platform shift rather than sentiment skill.
###CELL code###
#@title 4.1 Execute the pipeline
BANGLAGEC_SAMPLE = 200000  #@param {type:"integer"}
APPLY_GRAMMAR_RULES = True  #@param {type:"boolean"}

import config, preprocess
import importlib
importlib.reload(config)
importlib.reload(preprocess)
from preprocess import run_pipeline

stats = run_pipeline(sample=BANGLAGEC_SAMPLE, apply_rules=APPLY_GRAMMAR_RULES)
print("\nstats:", stats)
###CELL code###
#@title 4.2 Sanity-check the outputs
from config import ARTIFACTS, PREPROC_DIR

print("=== artefacts written ===")
for p in sorted(PREPROC_DIR.glob("*")):
    if p.is_file():
        print(f"  {p.name:<32} {p.stat().st_size / 1024:>10,.1f} KB")

if ARTIFACTS["clean_train"].exists():
    train = pd.read_csv(ARTIFACTS["clean_train"])
    test = pd.read_csv(ARTIFACTS["clean_test"])
    val = pd.read_csv(ARTIFACTS["clean_val"])

    print("\n=== label distribution per split ===")
    print(pd.crosstab(train["label"], "train"))
    print(pd.crosstab(val["label"], "val"))
    print(pd.crosstab(test["label"], "test"))

    print("\n=== rows per source (train) ===")
    print(train["source"].value_counts().to_string())

    print("\n=== negation coverage ===")
    print(f"  train rows containing NEG_: {train['has_negation'].mean():.2%}")
    print(f"  test  rows containing NEG_: {test['has_negation'].mean():.2%}")

    print("\n=== token length distribution (train) ===")
    print(train["n_tokens"].describe().round(2).to_string())

    print("\n=== 5 sample rows ===")
    with pd.option_context("display.max_colwidth", 130):
        print(train[["text", "label", "source"]].sample(5, random_state=42).to_string(index=False))

    # Leakage guard: identical text must not appear in two splits.
    overlap = len(set(train["text"]) & set(test["text"]))
    overlap_val = len(set(train["text"]) & set(val["text"]))
    print(f"\n=== leakage check === train/test overlap: {overlap}, "
          f"train/val overlap: {overlap_val}  (both must be 0)")
else:
    print("\nNo splits written. See the guidance printed by 4.1 about missing labelled data.")
###CELL md###
---
## 5. Inspect what the three specified datasets contributed

These outputs are the evidence quoted in `docs/03_methodology.md`.
###CELL code###
#@title 5.1 Grammar rules mined from BanglaGEC
if ARTIFACTS["grammar_rules"].exists():
    rules = pd.read_csv(ARTIFACTS["grammar_rules"])
    print(f"{len(rules):,} rules with support >= 5\n")
    print("=== rule types ===")
    print(rules["rule_type"].value_counts().to_string())
    print("\n=== 20 highest-support rules ===")
    with pd.option_context("display.max_colwidth", 55):
        print(rules.head(20)[["source_token", "target_token", "support", "rule_type"]].to_string(index=False))
    print("\n=== example sentences ===")
    with pd.option_context("display.max_colwidth", 120):
        print(rules.head(5)[["source_token", "example"]].to_string(index=False))
else:
    print("no rules mined (BanglaGEC unavailable)")
###CELL code###
#@title 5.2 Lexicons built from bangla-nlp-words
neg_path, sw_path = ARTIFACTS["negation_lexicon"], ARTIFACTS["stopwords"]
negators = [x for x in neg_path.read_text(encoding="utf-8").split("\n") if x.strip()] if neg_path.exists() else []
stopwords = [x for x in sw_path.read_text(encoding="utf-8").split("\n") if x.strip()] if sw_path.exists() else []

print(f"negators : {len(negators)}")
print("  ", " | ".join(negators[:40]))
print(f"\nstopwords: {len(stopwords)}")
print("  ", " | ".join(stopwords[:40]))
print("\nNegators are deliberately NOT removed as stopwords - they carry the polarity flip.")
###CELL code###
#@title 5.3 Negation tagging in action
from preprocess import tag_negation

demo = [
    "পণ্যটি ভালো না",
    "পণ্যটি ভালো",
    "এটা একেবারেই পছন্দ হয়নি",
    "দাম অনেক বেশি কিন্তু মান ভালো নয়",
]
negset = set(negators)
for s in demo:
    print(f"  {s}\n    -> {tag_negation(s, negset)}\n")
###CELL code###
#@title 5.4 Literature-review entries filtered from bangla-nlp-catalog
os.system(f"{sys.executable} src/catalog_to_lr.py --top 40")

if ARTIFACTS["lr_table"].exists():
    lr = pd.read_csv(ARTIFACTS["lr_table"])
    print(f"\n{len(lr):,} sentiment-related catalog entries captured")
    print("rendered tables: docs/generated/prior_work.md")
###CELL md###
---
## 6. Persist the outputs

CSVs are deliberately **not** committed to Git: they are large, and the upstream licences require redistribution through the original hosts. The manifest is the reproducibility record instead — same seed and same sources produce the same SHA-256.
###CELL code###
#@title 6.1 Zip the pre-processed dataset
if IN_COLAB:
    os.system("cd dataset && zip -q -r preprocessed.zip preprocessed")
    os.system("ls -lh dataset/preprocessed.zip")
else:
    print("local run - files are already on disk under dataset/preprocessed/")
###CELL code###
#@title 6.2 Copy to Drive or download to your machine
if IN_COLAB:
    if OUT is not None:
        import shutil
        for f in PREPROC_DIR.glob("*"):
            if f.is_file():
                shutil.copy(f, OUT / f.name)
        print("copied to Drive:", OUT)
    else:
        from google.colab import files
        if Path("dataset/preprocessed.zip").exists():
            files.download("dataset/preprocessed.zip")
        files.download(str(ARTIFACTS["data_manifest"]))
        print("check your browser downloads")
else:
    print("nothing to download")
###CELL code###
#@title 6.3 Print the reproducibility manifest
if ARTIFACTS["data_manifest"].exists():
    manifest = pd.read_csv(ARTIFACTS["data_manifest"])
    print(manifest.to_string(index=False))
    print("\nQuote these hashes in the report to demonstrate the splits are reproducible.")
###CELL md###
---
## Done — next step

Open **`notebooks/02_bert_finetuning.ipynb`** with a **T4 GPU** runtime.

On Colab either re-download the artefacts from Drive in Notebook 02, or run Notebook 02 immediately after this one in the same session so `dataset/preprocessed/` is still on disk.
