"""
evaluate.py
===========
Evaluation and analysis for the fine-tuned Bangla sentiment models.

Produces the figures and tables used in docs/04_analysis.md:
  - per-model metric table (accuracy, macro/weighted F1, per-class P/R/F1)
  - normalised confusion matrices
  - ROC curves + AUC (one-vs-rest, 3 class)
  - per-source breakdown  -> does the model generalise across platforms?
  - negation-subset accuracy -> did negation tagging actually help?
  - length-bucketed accuracy -> short comments are the hard case in Bangla
  - TF-IDF + Logistic Regression baseline for a non-transformer reference

Usage:
    python src/evaluate.py --model banglabert
    python src/evaluate.py --all
    python src/evaluate.py --baseline-only     # fast, CPU-only, no GPU needed

CSE432 - Natural Language Processing
Authors: 232-15-657 Shoumik Bala Somu | 232-15-644 Raisha Khan Raka
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ARTIFACTS, CHECKPOINT_DIR, LABELS_3, MODELS, NUM_LABELS, PREPROCESS,
    RESULTS_DIR, TRAIN, banner,
)

FIGSIZE = (7, 5)


def load_test() -> pd.DataFrame:
    p = Path(ARTIFACTS["clean_test"])
    if not p.exists():
        raise SystemExit(f"Missing {p}. Run the pre-processing stage first.")
    return pd.read_csv(p)


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_proba(model_key: str, df: pd.DataFrame) -> np.ndarray:
    """Return an (n, NUM_LABELS) probability matrix from a saved checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = CHECKPOINT_DIR / model_key.replace("/", "__") / "best"
    if not ckpt.exists():
        raise SystemExit(f"No checkpoint at {ckpt}. Train it first.")

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"  [eval] {model_key} on {device}")

    texts = df["text"].astype(str).tolist()
    probs: list[np.ndarray] = []
    bs = TRAIN["eval_batch_size"]
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = tok(texts[i:i + bs], truncation=True,
                        max_length=PREPROCESS["max_seq_len"],
                        padding=True, return_tensors="pt").to(device)
            logits = model(**batch).logits
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.vstack(probs)


# --------------------------------------------------------------------------
# Metric tables
# --------------------------------------------------------------------------
def metric_row(y_true, y_pred, y_proba=None, name: str = "") -> dict:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, matthews_corrcoef,
                                 precision_recall_fscore_support, roc_auc_score)

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(NUM_LABELS)), zero_division=0)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for i, lab in enumerate(LABELS_3):
        row[f"precision_{lab}"] = float(p[i])
        row[f"recall_{lab}"] = float(r[i])
        row[f"f1_{lab}"] = float(f1[i])
        row[f"support_{lab}"] = int(sup[i])
    if y_proba is not None:
        try:
            row["macro_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                             average="macro", labels=list(range(NUM_LABELS)))
        except ValueError:
            row["macro_auc"] = np.nan
    return row


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plot_confusion(y_true, y_pred, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_LABELS)))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=LABELS_3, yticklabels=LABELS_3)
    axes[0].set_title(f"{name} - counts")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=LABELS_3, yticklabels=LABELS_3, vmin=0, vmax=1)
    axes[1].set_title(f"{name} - row-normalised")
    for ax in axes:
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  [fig] {path.name}")


def plot_roc(y_true, y_proba, name: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    yb = label_binarize(y_true, classes=list(range(NUM_LABELS)))
    if yb.shape[1] == 1:                       # degenerate 2-class frame
        yb = np.hstack([1 - yb, yb])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, lab in enumerate(LABELS_3):
        fpr, tpr, _ = roc_curve(yb[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=2, label=f"{lab} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{name} - one-vs-rest ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  [fig] {path.name}")


def plot_grouped_bars(df: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    d = df.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(d["model"], d[metric], color="#4C72B0", edgecolor="black", linewidth=0.6)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.set_ylim(0, 1.0)
    for i, v in enumerate(d[metric]):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  [fig] {path.name}")


# --------------------------------------------------------------------------
# Slice analyses - the part that actually explains the numbers
# --------------------------------------------------------------------------
def slice_analysis(df_test: pd.DataFrame, y_true, y_pred) -> dict:
    """Break accuracy down by source, negation presence and text length.
    A single headline accuracy hides which slices the model is failing on."""
    d = df_test.copy()
    d["correct"] = (y_true == y_pred).astype(int)
    out: dict[str, object] = {}

    if "source" in d.columns:
        out["by_source"] = d.groupby("source")["correct"].agg(["mean", "count"]).round(4)
    if "has_negation" in d.columns:
        out["by_negation"] = d.groupby("has_negation")["correct"].agg(["mean", "count"]).round(4)

    bins = [0, 5, 10, 20, 40, 10_000]
    labels = ["1-5", "6-10", "11-20", "21-40", "41+"]
    d["len_bucket"] = pd.cut(d["text"].astype(str).str.split().str.len(),
                             bins=bins, labels=labels)
    out["by_length"] = d.groupby("len_bucket", observed=False)["correct"].agg(["mean", "count"]).round(4)

    for k, v in out.items():
        print(f"\n  --- accuracy {k} ---")
        print(v.to_string())
    return out


def error_examples(df_test: pd.DataFrame, y_true, y_pred, n: int = 20) -> pd.DataFrame:
    """Sample the highest-confidence wrong predictions. These are the rows to
    read by hand when writing the qualitative error analysis."""
    d = df_test.copy()
    d["true"] = y_true
    d["pred"] = y_pred
    wrong = d[d["true"] != d["pred"]].head(n * 5)
    if wrong.empty:
        return pd.DataFrame(columns=["text", "true_label", "pred_label"])
    sel = wrong.sample(n=min(n, len(wrong)), random_state=TRAIN["seed"])
    sel["true_label"] = [LABELS_3[i] for i in sel["true"]]
    sel["pred_label"] = [LABELS_3[i] for i in sel["pred"]]
    cols = ["text", "true_label", "pred_label"] + \
           ([c for c in ("source", "has_negation") if c in sel.columns])
    return sel[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# Non-transformer baseline
# --------------------------------------------------------------------------
def tfidf_baseline() -> dict:
    """TF-IDF (char + word) + Logistic Regression. Gives a reference point so
    the transformer numbers can be interpreted rather than read in isolation."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    df_train = pd.read_csv(ARTIFACTS["clean_train"])
    df_test = load_test()

    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=60_000,
                           sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=80_000,
                           sublinear_tf=True)
    from sklearn.pipeline import FeatureUnion
    pipe = Pipeline([
        ("features", FeatureUnion([("word", word), ("char", char)])),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced", n_jobs=-1)),
    ])
    print("  [baseline] fitting TF-IDF + LogisticRegression")
    pipe.fit(df_train["text"].astype(str), df_train["label_id"].astype(int))
    y_pred = pipe.predict(df_test["text"].astype(str))
    y_true = df_test["label_id"].astype(int).to_numpy()
    row = metric_row(y_true, y_pred, name="tfidf_logreg")
    print(f"  [baseline] accuracy={row['accuracy']:.4f} macro_f1={row['macro_f1']:.4f}")
    return row


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def evaluate_all(model_keys: list[str], baseline: bool = True) -> pd.DataFrame:
    df_test = load_test()
    y_true = df_test["label_id"].astype(int).to_numpy()
    rows = []

    if baseline:
        try:
            rows.append(tfidf_baseline())
        except Exception as exc:  # noqa: BLE001
            print(f"  [baseline] skipped: {str(exc)[:200]}")

    for key in model_keys:
        print(banner(f"Evaluating {key}"))
        try:
            proba = predict_proba(key, df_test)
        except SystemExit as exc:
            print(f"  [skip] {exc}")
            continue
        y_pred = proba.argmax(axis=1)
        row = metric_row(y_true, y_pred, proba, name=key)
        rows.append(row)

        safe = key.replace("/", "__")
        plot_confusion(y_true, y_pred, key, RESULTS_DIR / f"confusion_{safe}.png")
        plot_roc(y_true, proba, key, RESULTS_DIR / f"roc_{safe}.png")
        slices = slice_analysis(df_test, y_true, y_pred)
        for slice_name, slice_df in slices.items():
            if isinstance(slice_df, pd.DataFrame):
                slice_df.to_csv(RESULTS_DIR / f"slice_{slice_name}_{safe}.csv")
        errs = error_examples(df_test, y_true, y_pred)
        errs.to_csv(RESULTS_DIR / f"errors_{safe}.csv", index=False)

        (CHECKPOINT_DIR / safe / "eval_slices.json").write_text(
            json.dumps({k: v.to_dict() for k, v in slices.items()
                        if isinstance(v, pd.DataFrame)}, indent=2, default=str),
            encoding="utf-8")

    table = pd.DataFrame(rows)
    if table.empty:
        print("\n[!] Nothing evaluated. Train a model first:  python src/train.py --model banglabert")
        return table

    table = table.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    table.to_csv(ARTIFACTS["metrics"], index=False)
    print(banner("Metric summary (sorted by macro-F1)"))
    show = ["model", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "mcc"]
    show = [c for c in show if c in table.columns]
    print(table[show].round(4).to_string(index=False))

    if len(table) > 1:
        plot_grouped_bars(table, "macro_f1", "Macro-F1 by model",
                          RESULTS_DIR / "bar_macro_f1.png")
        plot_grouped_bars(table, "accuracy", "Accuracy by model",
                          RESULTS_DIR / "bar_accuracy.png")
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true", help="evaluate every registry model with a checkpoint")
    ap.add_argument("--baseline-only", action="store_true", help="TF-IDF baseline only (CPU, fast)")
    ap.add_argument("--no-baseline", action="store_true")
    a = ap.parse_args()

    if a.baseline_only:
        row = tfidf_baseline()
        pd.DataFrame([row]).to_csv(RESULTS_DIR / "metrics_baseline.csv", index=False)
        return 0

    if a.all:
        keys = [k for k in MODELS
                if (CHECKPOINT_DIR / k.replace("/", "__") / "best").exists()]
        if not keys:
            print("No checkpoints found under checkpoints/. Run src/train.py first.")
            return 1
    elif a.model:
        keys = [a.model]
    else:
        ap.error("pass --model, --all, or --baseline-only")
        return 2

    evaluate_all(keys, baseline=not a.no_baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
