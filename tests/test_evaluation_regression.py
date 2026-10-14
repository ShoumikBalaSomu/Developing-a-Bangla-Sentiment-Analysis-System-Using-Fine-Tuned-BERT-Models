"""
test_evaluation_regression.py
==============================
Automated regression tests and evaluation quality gates for trained sentiment models.
Enforces that model accuracy and macro F1 do not degrade beyond statistical thresholds
against established empirical baselines (ml-eval-regression).

CSE432 - Natural Language Processing
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
METRICS_FILE = RESULTS_DIR / "metrics_all_models.csv"
ABLATION_FILE = RESULTS_DIR / "metrics_ablation_no_rules.csv"

# Canonical baseline values from full training run (tolerance = 1.5% absolute)
TOLERANCE = 0.015

EXPECTED_BASELINES = {
    "tfidf_logreg": {
        "macro_f1": 0.6341,
        "accuracy": 0.8854,
    },
    "banglishbert": {
        "macro_f1": 0.6065,
        "accuracy": 0.8278,
    },
    "banglabert": {
        "macro_f1": 0.5985,
        "accuracy": 0.8300,
    },
    "mbert": {
        "macro_f1": 0.5843,
        "accuracy": 0.7932,
    },
    "bangla_bert_base": {
        "macro_f1": 0.5806,
        "accuracy": 0.7731,
    },
    "xlmr": {
        "macro_f1": 0.5760,
        "accuracy": 0.7968,
    },
}

EXPECTED_SUPPORT = {
    "negative": 389.0,
    "neutral": 21.0,
    "positive": 978.0,
    "total": 1388.0,
}


class TestEvaluationRegression(unittest.TestCase):
    """Regression test gate for model benchmark metrics."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(
            cls,
            METRICS_FILE.exists(),
            f"Metrics file missing at {METRICS_FILE}",
        )
        cls.records: dict[str, dict[str, float]] = {}
        with open(METRICS_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model"]
                cls.records[model_name] = {}
                for k, v in row.items():
                    if k != "model" and v != "" and v is not None:
                        try:
                            cls.records[model_name][k] = float(v)
                        except ValueError:
                            pass

    def test_metrics_file_is_populated(self):
        """Ensure metrics table has recorded entries."""
        self.assertGreaterEqual(
            len(self.records),
            6,
            f"Expected at least 6 models in metrics table, found {len(self.records)}",
        )

    def test_all_expected_models_present(self):
        """Ensure all canonical model keys are evaluated."""
        for model_key in EXPECTED_BASELINES:
            self.assertIn(
                model_key,
                self.records,
                f"Missing baseline model '{model_key}' in {METRICS_FILE}",
            )

    def test_regression_macro_f1_within_gate(self):
        """Assert Macro F1 does not regress beyond tolerance threshold."""
        for model_key, expected in EXPECTED_BASELINES.items():
            actual_f1 = self.records[model_key]["macro_f1"]
            target_f1 = expected["macro_f1"]
            delta = abs(actual_f1 - target_f1)
            self.assertLessEqual(
                delta,
                TOLERANCE,
                f"Model '{model_key}' Macro F1 ({actual_f1:.4f}) deviates by {delta:.4f} > {TOLERANCE} from baseline ({target_f1:.4f})",
            )

    def test_regression_accuracy_within_gate(self):
        """Assert Accuracy does not regress beyond tolerance threshold."""
        for model_key, expected in EXPECTED_BASELINES.items():
            actual_acc = self.records[model_key]["accuracy"]
            target_acc = expected["accuracy"]
            delta = abs(actual_acc - target_acc)
            self.assertLessEqual(
                delta,
                TOLERANCE,
                f"Model '{model_key}' Accuracy ({actual_acc:.4f}) deviates by {delta:.4f} > {TOLERANCE} from baseline ({target_acc:.4f})",
            )

    def test_metric_bounds_valid(self):
        """Validate that all probability and ratio metrics are in [0.0, 1.0]."""
        bounded_fields = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
            "precision_negative",
            "recall_negative",
            "f1_negative",
            "precision_neutral",
            "recall_neutral",
            "f1_neutral",
            "precision_positive",
            "recall_positive",
            "f1_positive",
        ]
        for model_key, metrics in self.records.items():
            for field in bounded_fields:
                if field in metrics:
                    val = metrics[field]
                    self.assertTrue(
                        0.0 <= val <= 1.0,
                        f"{model_key} {field} ({val}) is outside [0.0, 1.0]",
                    )

    def test_mcc_bounds_valid(self):
        """Matthews Correlation Coefficient must be between -1.0 and 1.0."""
        for model_key, metrics in self.records.items():
            if "mcc" in metrics:
                val = metrics["mcc"]
                self.assertTrue(
                    -1.0 <= val <= 1.0,
                    f"{model_key} mcc ({val}) is outside [-1.0, 1.0]",
                )

    def test_class_support_totals(self):
        """Validate test split class supports sum exactly to 1,388 samples."""
        for model_key, metrics in self.records.items():
            neg_sup = metrics.get("support_negative", 0.0)
            neu_sup = metrics.get("support_neutral", 0.0)
            pos_sup = metrics.get("support_positive", 0.0)
            total = neg_sup + neu_sup + pos_sup
            self.assertAlmostEqual(
                total,
                EXPECTED_SUPPORT["total"],
                places=1,
                msg=f"{model_key} total support ({total}) != {EXPECTED_SUPPORT['total']}",
            )
            self.assertAlmostEqual(neg_sup, EXPECTED_SUPPORT["negative"], places=1)
            self.assertAlmostEqual(neu_sup, EXPECTED_SUPPORT["neutral"], places=1)
            self.assertAlmostEqual(pos_sup, EXPECTED_SUPPORT["positive"], places=1)

    def test_ablation_macro_f1_baseline(self):
        """Validate BanglaGEC ablation baseline without grammar rules matches reported empirical metrics."""
        self.assertTrue(
            ABLATION_FILE.exists(),
            f"Ablation metrics file missing at {ABLATION_FILE}",
        )
        ablation_records = {}
        with open(ABLATION_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ablation_records[row["model"]] = {
                    k: float(v) for k, v in row.items() if k != "model" and v != "" and v is not None
                }
        self.assertIn("banglabert_no_rules", ablation_records)
        metrics = ablation_records["banglabert_no_rules"]
        # Expected: macro_f1 = 0.6228, accuracy = 0.8634, mcc = 0.7198
        self.assertAlmostEqual(metrics["macro_f1"], 0.6228, delta=TOLERANCE)
        self.assertAlmostEqual(metrics["accuracy"], 0.8634, delta=TOLERANCE)
        self.assertAlmostEqual(metrics["mcc"], 0.7198, delta=TOLERANCE)


if __name__ == "__main__":
    unittest.main()
