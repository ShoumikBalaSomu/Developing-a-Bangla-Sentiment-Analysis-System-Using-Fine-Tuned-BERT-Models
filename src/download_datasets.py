"""
download_datasets.py
====================
Fetches every dataset in the registry into dataset/raw/.

Two channels per dataset, tried in order:
  1. Hugging Face `datasets`  - no credentials, works out of the box on Colab
  2. Kaggle CLI               - needs kaggle.json, gives the original files

The script never aborts the whole run because one source is missing. It
records what succeeded and what did not, then exits 0 if at least the
grammar + catalog resources (both public on HF) arrived, so Notebook 01 can
still demonstrate the BanglaGEC rule-mining stage.

Usage:
    python src/download_datasets.py                 # everything
    python src/download_datasets.py --only banglagec bangla_nlp_catalog
    python src/download_datasets.py --skip-kaggle   # HF sources only

CSE432 - Natural Language Processing
Authors: 232-15-657 Shoumik Bala Somu | 232-15-644 Raisha Khan Raka
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATASETS, RAW_DIR, banner  # noqa: E402


# --------------------------------------------------------------------------
# Kaggle channel
# --------------------------------------------------------------------------
def kaggle_credentials_present() -> bool:
    """Look for kaggle.json in the usual locations (Windows + POSIX + Colab)."""
    home = Path.home()
    candidates = [
        home / ".kaggle" / "kaggle.json",
        Path("/root/.kaggle/kaggle.json"),
        Path("/content/kaggle.json"),
        Path.cwd() / "kaggle.json",
    ]
    return any(c.exists() for c in candidates)


def download_via_kaggle(slug: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
           "-d", slug, "-p", str(dest), "--unzip"]
    print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Retry zipped, then unzip ourselves: some Kaggle datasets reject --unzip.
        cmd_zip = [sys.executable, "-m", "kaggle", "datasets", "download",
                   "-d", slug, "-p", str(dest)]
        proc = subprocess.run(cmd_zip, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  [kaggle] FAILED {slug}: {proc.stderr.strip()[:300]}")
            return False
        for z in dest.glob("*.zip"):
            with zipfile.ZipFile(z) as fh:
                fh.extractall(dest)
            z.unlink()
    if proc.stderr.strip():
        print(f"  [kaggle] {proc.stderr.strip()[:200]}")
    return any(dest.rglob("*.csv")) or any(dest.rglob("*.json"))


# --------------------------------------------------------------------------
# Hugging Face channel
# --------------------------------------------------------------------------
def download_via_hf(hf_id: str, dest: Path, max_rows: int | None = None) -> bool:
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [hf] `datasets` not installed. pip install datasets")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(hf_id)
    except Exception as exc:  # noqa: BLE001 - many HF failure modes, all non-fatal
        print(f"  [hf] FAILED {hf_id}: {str(exc)[:300]}")
        return False

    n_files = 0
    for split, table in ds.items():
        if max_rows is not None and len(table) > max_rows:
            # Deterministic head sample: load_dataset order is stable for a
            # given dataset revision, so this is reproducible across runs.
            table = table.select(range(max_rows))
        out = dest / f"{split}.csv"
        table.to_csv(out, index=False)
        print(f"  [hf] {split}: {len(table):,} rows -> {out.name}")
        n_files += 1
    return n_files > 0


# --------------------------------------------------------------------------
# Per-dataset orchestration
# --------------------------------------------------------------------------
def fetch(name: str, spec: dict, skip_kaggle: bool, max_rows: int | None) -> dict:
    print(banner(f"Dataset: {name}  ({spec['role']})"))
    print(f"  {spec['notes']}")
    dest = RAW_DIR / name
    status = {"name": name, "role": spec["role"], "hf": "n/a", "kaggle": "n/a",
              "ok": False, "path": str(dest)}

    if dest.exists() and any(dest.rglob("*")):
        print("  [cache] already present, skipping download")
        status.update(ok=True, hf="cached", kaggle="cached")
        return status

    # Prefer HF when it exists and needs no auth.
    if spec.get("hf") and not skip_kaggle:
        status["hf"] = "ok" if download_via_hf(spec["hf"], dest, max_rows) else "failed"

    have_data = any(dest.rglob("*.csv")) if dest.exists() else False

    if not have_data and spec.get("kaggle") and not skip_kaggle:
        if spec["needs_kaggle_auth"] and not kaggle_credentials_present():
            print("  [kaggle] no kaggle.json found - skipping (auth required)")
            status["kaggle"] = "no-credentials"
        else:
            status["kaggle"] = "ok" if download_via_kaggle(spec["kaggle"], dest) else "failed"
    elif skip_kaggle:
        status["kaggle"] = "skipped"

    status["ok"] = bool(dest.exists() and any(dest.rglob("*")))
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="Download all project datasets.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="download only these registry keys")
    ap.add_argument("--skip-kaggle", action="store_true",
                    help="use Hugging Face sources only (no credentials needed)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows per split on HF downloads (e.g. 200000)")
    args = ap.parse_args()

    keys = args.only or list(DATASETS)
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        print(f"Unknown dataset keys: {unknown}\nValid keys: {list(DATASETS)}")
        return 2

    print(banner("Dataset acquisition"))
    print(f"  target dir : {RAW_DIR}")
    print(f"  kaggle auth: {'yes' if kaggle_credentials_present() else 'NO (kaggle.json missing)'}")

    rows = [fetch(k, DATASETS[k], args.skip_kaggle, args.max_rows) for k in keys]
    summary = pd.DataFrame(rows)
    print(banner("Acquisition summary"))
    print(summary.to_string(index=False))

    ok_names = set(summary.loc[summary.ok, "name"])
    # grammar + catalog are the minimum needed to demonstrate the pipeline
    minimum = {"banglagec"} & ok_names
    if not minimum:
        print("\n[!] BanglaGEC did not download. Grammar-rule mining will be skipped.\n"
              "    Re-run with Kaggle credentials, or with --max-rows to shrink it.")
    return 0


if __name__ == "__main__":
    # keep the tree clean if a previous partial run left an empty dir behind
    if RAW_DIR.exists() and not any(RAW_DIR.iterdir()):
        shutil.rmtree(RAW_DIR, ignore_errors=True)
    raise SystemExit(main())
