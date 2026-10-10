"""
train.py
========
Fine-tunes the transformer registry on the pre-processed Bangla sentiment data.

Design decisions that matter for this dataset
---------------------------------------------
1. Discriminative learning rates. The pre-trained backbone gets
   TRAIN["learning_rate"] (2e-5) while the randomly-initialised classifier head
   gets TRAIN["head_learning_rate"] (6e-5). A single LR either under-trains the
   head or destabilises the backbone.
2. Inverse-frequency class weights. Bangla review corpora are strongly
   positive-skewed; unweighted cross-entropy collapses to the majority class.
3. Early stopping on macro-F1, not loss. Loss can keep falling while the
   minority class degrades, which is exactly the failure mode that makes a
   balanced-accuracy number look fine while the model is useless on negatives.
4. Label smoothing (0.05). Crowdsourced/lexicon-derived sentiment labels are
   noisy; smoothing stops the model from being over-confident on them.

Usage:
    python src/train.py --model csebuetnlp/banglabert
    python src/train.py --model banglabert --epochs 6
    python src/train.py --all

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
    ARTIFACTS, CHECKPOINT_DIR, ID2LABEL, LABEL2ID, LABELS_3, MODELS, NUM_LABELS,
    PREPROCESS, RESULTS_DIR, TRAIN, banner,
)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for key in ("clean_train", "clean_val", "clean_test"):
        if not Path(ARTIFACTS[key]).exists():
            raise SystemExit(
                f"Missing {ARTIFACTS[key]}. Run Notebook 01 (data pre-processing) "
                f"or `python src/preprocess.py` first.")
    return (pd.read_csv(ARTIFACTS["clean_train"]),
            pd.read_csv(ARTIFACTS["clean_val"]),
            pd.read_csv(ARTIFACTS["clean_test"]))


def class_weights(labels: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights, normalised to mean 1 so the effective LR is
    unchanged and only the *relative* class emphasis shifts."""
    counts = np.bincount(labels, minlength=NUM_LABELS).astype(float)
    counts[counts == 0] = 1.0
    w = len(labels) / (NUM_LABELS * counts)
    w = w / w.mean()
    return {i: float(x) for i, x in enumerate(w)}


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------
def build_model(model_key: str):
    """Return (model, tokenizer). AutoModelForSequenceClassification works for
    bert / roberta / electra families, which covers the whole registry."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    spec = MODELS[model_key] if model_key in MODELS else None
    hf_id = spec["hf_id"] if spec else model_key
    print(f"  loading {hf_id}")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="single_label_classification",
    )
    return model, tokenizer


def param_groups(model, backbone_lr: float, head_lr: float) -> list[dict]:
    """Separate the classifier head from the pre-trained backbone."""
    head_names = {"classifier", "cls", "score", "pooler", "dense"}
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = name.split(".")[0]
        (head if top in head_names else backbone).append(p)
    groups = [{"params": backbone, "lr": backbone_lr}]
    if head:
        groups.append({"params": head, "lr": head_lr})
    return groups


# --------------------------------------------------------------------------
# Weighted loss
# --------------------------------------------------------------------------
class WeightedTrainer:
    """Trainer subclass factory. Defined lazily so importing this module does
    not require torch (useful for the docs/analysis-only paths)."""

    @staticmethod
    def get(weights: dict[int, float]):
        import torch
        from transformers import Trainer

        w = torch.tensor([weights[i] for i in range(NUM_LABELS)],
                         dtype=torch.float)

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False,
                             num_items_in_batch=None):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                dev = logits.device
                loss_f = torch.nn.CrossEntropyLoss(
                    weight=w.to(dev),
                    label_smoothing=TRAIN["label_smoothing"],
                )
                loss = loss_f(logits.view(-1, NUM_LABELS), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        return _WeightedTrainer


# --------------------------------------------------------------------------
# Metrics + callbacks
# --------------------------------------------------------------------------
def compute_metrics_factory():
    from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels, preds, average=None, labels=list(range(NUM_LABELS)), zero_division=0)
        return {
            "accuracy": accuracy_score(labels, preds),
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_precision": float(np.mean(p)),
            "macro_recall": float(np.mean(r)),
            **{f"f1_{LABELS_3[i]}": float(f1[i]) for i in range(NUM_LABELS)},
        }

    return compute_metrics


def make_callbacks():
    from transformers import EarlyStoppingCallback
    cbs = [EarlyStoppingCallback(early_stopping_patience=TRAIN["early_stopping_patience"])]
    return cbs


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------
def tokenise(df_train, df_val, df_test, tokenizer, max_len: int):
    from datasets import Dataset

    def to_ds(df):
        return Dataset.from_dict({"text": df["text"].astype(str).tolist(),
                                  "label": df["label_id"].astype(int).tolist()})

    def enc(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=max_len,
                        padding=False)
        out["labels"] = batch["label"]
        return out

    raw = {"train": to_ds(df_train), "val": to_ds(df_val), "test": to_ds(df_test)}
    cols = ["input_ids", "attention_mask", "labels"]
    # ELECTRA/XLM-R variants may also emit token_type_ids; keep if present.
    enc_raw = {k: v.map(enc, batched=True) for k, v in raw.items()}
    extra = [c for c in enc_raw["train"].column_names if c.startswith("token_type")]
    keep = cols + extra
    return {k: v.remove_columns([c for c in v.column_names if c not in keep])
            for k, v in enc_raw.items()}


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def _training_arguments(out_dir: Path, epochs: int, fp16_ok: bool, n_train: int = 1000):
    """Build TrainingArguments, tolerating transformers 4 vs 5 differences."""
    import inspect
    from transformers import TrainingArguments

    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters

    common = dict(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN["batch_size"],
        gradient_accumulation_steps=TRAIN.get("gradient_accumulation_steps", 1),
        per_device_eval_batch_size=TRAIN["eval_batch_size"],
        learning_rate=TRAIN["learning_rate"],
        weight_decay=TRAIN["weight_decay"],
        max_grad_norm=TRAIN["max_grad_norm"],
        fp16=fp16_ok,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=TRAIN["save_total_limit"],
        logging_steps=50,
        seed=TRAIN["seed"],
        report_to=TRAIN["report_to"],
        dataloader_num_workers=0,   # keeps Colab disk/RAM usage bounded
    )
    if "warmup_ratio" in params:
        common["warmup_ratio"] = TRAIN["warmup_ratio"]
    else:
        steps_per_epoch = max(1, n_train // TRAIN["batch_size"])
        common["warmup_steps"] = max(1, int(steps_per_epoch * epochs * TRAIN["warmup_ratio"]))

    if "eval_strategy" in params:
        common["eval_strategy"] = "epoch"
    else:
        common["evaluation_strategy"] = "epoch"

    return TrainingArguments(**common)


def train_one(model_key: str, epochs: int | None = None,
              subset: int | None = None, use_weights: bool | None = None) -> dict:
    import torch
    from transformers import DataCollatorWithPadding, Trainer, set_seed

    set_seed(TRAIN["seed"])
    epochs = epochs or TRAIN["epochs"]
    use_weights = TRAIN["class_weight"] if use_weights is None else use_weights

    print(banner(f"Fine-tuning: {model_key}"))
    df_train, df_val, df_test = load_splits()
    if subset:
        df_train = df_train.head(subset)
        df_val = df_val.head(min(subset // 4, len(df_val)))
        print(f"  [debug] subset mode: train={len(df_train)} val={len(df_val)}")

    print(f"  train={len(df_train):,} val={len(df_val):,} test={len(df_test):,}")
    weights = class_weights(df_train["label_id"].to_numpy())
    print(f"  class weights: {weights}")

    model, tokenizer = build_model(model_key)
    max_len = PREPROCESS["max_seq_len"]
    enc = tokenise(df_train, df_val, df_test, tokenizer, max_len)

    out_dir = CHECKPOINT_DIR / model_key.replace("/", "__")
    args = _training_arguments(out_dir, epochs,
                               TRAIN["fp16"] and torch.cuda.is_available(),
                               len(df_train))

    # Discriminative LR is applied via an explicit optimizer so that the head
    # and the backbone are not forced to share one learning rate.
    optimizer = torch.optim.AdamW(
        param_groups(model, TRAIN["learning_rate"], TRAIN["head_learning_rate"]),
        lr=TRAIN["learning_rate"],
        weight_decay=TRAIN["weight_decay"],
    )

    trainer_cls = WeightedTrainer.get(weights) if use_weights else Trainer
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=enc["train"],
        eval_dataset=enc["val"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics_factory(),
        callbacks=make_callbacks(),
        optimizers=(optimizer, None),
    )
    try:
        # transformers >= 4.46
        trainer = trainer_cls(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        # transformers < 4.46
        trainer = trainer_cls(tokenizer=tokenizer, **trainer_kwargs)

    trainer.train()

    print(banner(f"Evaluating {model_key} on held-out test split"))
    test_metrics = trainer.predict(enc["test"]).metrics
    flat = {k.replace("test_", ""): v for k, v in test_metrics.items()
            if k.startswith("test_")}
    flat.update({"model_key": model_key,
                 "hf_id": MODELS.get(model_key, {}).get("hf_id", model_key),
                 "epochs": epochs,
                 "n_train": len(df_train), "n_val": len(df_val),
                 "n_test": len(df_test)})

    trainer.save_model(str(out_dir / "best"))
    tokenizer.save_pretrained(str(out_dir / "best"))
    (out_dir / "metrics.json").write_text(json.dumps(flat, indent=2, default=str),
                                          encoding="utf-8")
    print(json.dumps({k: v for k, v in flat.items()
                      if isinstance(v, (int, float, str))}, indent=2, default=str))
    del trainer, model
    import gc
    gc.collect()
    return flat


def append_metrics(flat: dict) -> None:
    """Accumulate per-model results into results/metrics_all_models.csv."""
    path = Path(ARTIFACTS["metrics"])
    flat["model"] = flat.get("model", flat.get("model_key"))
    row = pd.DataFrame([flat])
    if path.exists():
        old = pd.read_csv(path)
        key_col = "model_key" if "model_key" in old.columns else ("model" if "model" in old.columns else None)
        if key_col:
            old = old[old[key_col] != flat.get("model_key", flat.get("model"))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(path, index=False)
    print(f"  [metrics] appended -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="registry key or raw HF model id")
    ap.add_argument("--all", action="store_true", help="train every registry model")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None,
                    help="train on N rows only - smoke test the pipeline")
    ap.add_argument("--no-class-weights", action="store_true")
    a = ap.parse_args()

    if a.all:
        keys = list(MODELS)
    elif a.model:
        keys = [a.model]
    else:
        ap.error("pass --model <key|hf_id> or --all")
        return 2

    for k in keys:
        flat = train_one(k, epochs=a.epochs, subset=a.subset,
                         use_weights=not a.no_class_weights)
        append_metrics(flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
