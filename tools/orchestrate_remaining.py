"""
orchestrate_remaining.py
========================
Runs the full suite of remaining model trainings and evaluations sequentially:
1. Wait for bangla_bert_base training (if running)
2. Evaluate bangla_bert_base
3. Train and evaluate banglishbert (1 epoch)
4. Train and evaluate mbert (1 epoch)
5. Train and evaluate xlmr (1 epoch)
6. Train and evaluate ablation model (banglabert without grammar rules)
7. Build consolidated leaderboard and group comparison tables
"""

import sys
import time
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ARTIFACTS, CHECKPOINT_DIR, MODELS, RESULTS_DIR, PREPROC_DIR
from evaluate import evaluate_all, metric_row, predict_proba, load_test, tfidf_baseline

def run_cmd(cmd_list):
    print(f"\n[RUN] {' '.join(cmd_list)}")
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=str(ROOT))
    elapsed = time.time() - t0
    print(f"[DONE] code {res.returncode} in {elapsed/60:.2f} mins\n")
    return res.returncode == 0

def check_and_eval(model_key):
    print(f"\n=== Evaluating {model_key} ===")
    run_cmd([sys.executable, "src/evaluate.py", "--model", model_key, "--no-baseline"])

def build_group_comparison():
    print("\n=== Generating Group Comparison ===")
    metrics_path = Path(ARTIFACTS["metrics"])
    if not metrics_path.exists():
        print("metrics_all_models.csv not found")
        return
    m = pd.read_csv(metrics_path)
    BANGLA = {"banglabert", "bangla_bert_base", "banglishbert"}
    MULTI = {"mbert", "xlmr"}

    rows = []
    # Bangla-specific
    sub_b = m[m["model"].isin(BANGLA)]
    if not sub_b.empty:
        rows.append({
            "group": "Bangla-specific",
            "models": ", ".join(sorted(sub_b["model"].unique())),
            "best_macro_f1": sub_b["macro_f1"].max(),
            "mean_macro_f1": sub_b["macro_f1"].mean(),
            "best_accuracy": sub_b["accuracy"].max(),
        })

    # Multilingual
    sub_m = m[m["model"].isin(MULTI)]
    if not sub_m.empty:
        rows.append({
            "group": "Multilingual",
            "models": ", ".join(sorted(sub_m["model"].unique())),
            "best_macro_f1": sub_m["macro_f1"].max(),
            "mean_macro_f1": sub_m["macro_f1"].mean(),
            "best_accuracy": sub_m["accuracy"].max(),
        })

    # Classical
    sub_c = m[m["model"] == "tfidf_logreg"]
    if not sub_c.empty:
        rows.append({
            "group": "Classical",
            "models": "tfidf_logreg",
            "best_macro_f1": sub_c["macro_f1"].max(),
            "mean_macro_f1": sub_c["macro_f1"].mean(),
            "best_accuracy": sub_c["accuracy"].max(),
        })

    comp = pd.DataFrame(rows).round(4)
    print(comp.to_string(index=False))
    comp.to_csv(RESULTS_DIR / "group_comparison.csv", index=False)

def main():
    # 1. Wait for bangla_bert_base checkpoint if not yet ready
    ckpt_bbb = CHECKPOINT_DIR / "bangla_bert_base" / "best"
    while not ckpt_bbb.exists():
        # Check if process is still running
        ps = subprocess.run(["pgrep", "-f", "train.py --model bangla_bert_base"], capture_output=True, text=True)
        if not ps.stdout.strip():
            print("bangla_bert_base process not running; launching training...")
            run_cmd([sys.executable, "src/train.py", "--model", "bangla_bert_base", "--epochs", "1"])
            break
        print("Waiting for bangla_bert_base training to finish...", flush=True)
        time.sleep(30)

    check_and_eval("bangla_bert_base")

    # 2. Train banglishbert
    ckpt_bb = CHECKPOINT_DIR / "banglishbert" / "best"
    if not ckpt_bb.exists():
        run_cmd([sys.executable, "src/train.py", "--model", "banglishbert", "--epochs", "1"])
    check_and_eval("banglishbert")

    # 3. Train mbert
    ckpt_mb = CHECKPOINT_DIR / "mbert" / "best"
    if not ckpt_mb.exists():
        run_cmd([sys.executable, "src/train.py", "--model", "mbert", "--epochs", "1"])
    check_and_eval("mbert")

    # 4. Train xlmr
    ckpt_xl = CHECKPOINT_DIR / "xlmr" / "best"
    if not ckpt_xl.exists():
        run_cmd([sys.executable, "src/train.py", "--model", "xlmr", "--epochs", "1"])
    check_and_eval("xlmr")

    # 5. Ablation: BanglaBERT without grammar rules
    ckpt_ablation = CHECKPOINT_DIR / "banglabert_no_rules" / "best"
    if not ckpt_ablation.exists():
        print("\n=== Running Ablation: banglabert without grammar rules ===")
        # Backup original data files
        import shutil
        orig_train = ARTIFACTS["clean_train"]
        orig_val = ARTIFACTS["clean_val"]
        orig_test = ARTIFACTS["clean_test"]

        # Run preprocessing without rules
        run_cmd([sys.executable, "src/preprocess.py", "--no-rules"])
        # Train banglabert on no-rules data
        run_cmd([sys.executable, "src/train.py", "--model", "banglabert", "--epochs", "1"])
        # Move checkpoint to banglabert_no_rules
        src_best = CHECKPOINT_DIR / "banglabert" / "best"
        tgt_best = CHECKPOINT_DIR / "banglabert_no_rules" / "best"
        tgt_best.parent.mkdir(parents=True, exist_ok=True)
        if src_best.exists():
            shutil.copytree(src_best, tgt_best, dirs_exist_ok=True)
            # Evaluate no-rules
            df_test_norules = pd.read_csv(ARTIFACTS["clean_test"])
            y_true = df_test_norules["label_id"].to_numpy()
            proba = predict_proba("banglabert_no_rules", df_test_norules)
            y_pred = proba.argmax(axis=1)
            row = metric_row(y_true, y_pred, proba, name="banglabert_no_rules")
            pd.DataFrame([row]).to_csv(RESULTS_DIR / "metrics_ablation_no_rules.csv", index=False)

        # Restore standard preprocessed data with rules
        print("Restoring standard preprocessed dataset with grammar rules...")
        run_cmd([sys.executable, "src/preprocess.py"])
        # Restore banglabert 2-epoch checkpoint
        backup_dir = CHECKPOINT_DIR / "banglabert_backup_2epoch"
        if backup_dir.exists():
            print("Restoring 2-epoch banglabert checkpoint...")
            shutil.copytree(backup_dir, CHECKPOINT_DIR / "banglabert", dirs_exist_ok=True)


    # 6. Build consolidated tables
    build_group_comparison()
    print("\nALL REMAINING EXPERIMENTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
