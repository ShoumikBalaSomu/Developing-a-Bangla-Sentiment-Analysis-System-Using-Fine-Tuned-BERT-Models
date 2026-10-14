"""
test_data_integrity.py
======================
Automated dataset provenance and cryptographic integrity test suite (ml-dataset-provenance).
Verifies SHA-256 hashes against data_manifest.csv, split sizes, label validity,
and data cleanliness.

CSE432 - Natural Language Processing
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
PREPROC_DIR = DATASET_DIR / "preprocessed"
MANIFEST_FILE = PREPROC_DIR / "data_manifest.csv"

EXPECTED_ROW_COUNTS = {
    "train_clean.csv": 5797,
    "val_clean.csv": 1388,
    "test_clean.csv": 1388,
}

VALID_LABELS_STR = {"negative", "neutral", "positive"}
VALID_LABEL_IDS = {0, 1, 2}
VALID_BINARY_LABELS_STR = {"negative", "positive"}
VALID_BINARY_LABEL_IDS = {0, 1}


class TestDataIntegrity(unittest.TestCase):
    """Test suite for dataset provenance, cryptographic digests, and invariants."""

    @classmethod
    def setUpClass(cls):
        cls.assertTrue(
            cls,
            MANIFEST_FILE.exists(),
            f"Data manifest missing at {MANIFEST_FILE}",
        )
        cls.manifest: dict[str, dict[str, str]] = {}
        with open(MANIFEST_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                artifact = row["artifact"]
                cls.manifest[artifact] = row

    def test_manifest_contains_core_artifacts(self):
        """Validate core preprocessed artifacts are catalogued."""
        core_artifacts = [
            "clean_train",
            "clean_val",
            "clean_test",
            "grammar_rules",
            "negation_lexicon",
            "stopwords",
            "vocab_wellformed",
        ]
        for art in core_artifacts:
            self.assertIn(
                art,
                self.manifest,
                f"Core artifact '{art}' not found in manifest",
            )

    def test_sha256_digests_match_manifest(self):
        """Verify SHA-256 cryptographic hashes of all files in manifest."""
        for artifact, meta in self.manifest.items():
            filename = meta.get("file")
            expected_hash = meta.get("sha256")
            if not filename or filename == "-" or not expected_hash or expected_hash == "-":
                continue

            target_file = PREPROC_DIR / filename
            if not target_file.exists():
                target_file = ROOT / "results" / filename

            self.assertTrue(
                target_file.exists(),
                f"File '{filename}' recorded in manifest does not exist on disk",
            )

            hasher = hashlib.sha256()
            with open(target_file, "rb") as f:
                while chunk := f.read(65536):
                    hasher.update(chunk)
            actual_hash = hasher.hexdigest()

            # Ignore data_manifest self-hash if it records its own previous state
            if artifact == "data_manifest":
                continue

            self.assertEqual(
                actual_hash,
                expected_hash,
                f"SHA-256 hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}",
            )

    def test_row_counts_and_split_integrity(self):
        """Verify train/val/test row counts match 70/15/15 stratified proportions."""
        for filename, expected_count in EXPECTED_ROW_COUNTS.items():
            csv_path = PREPROC_DIR / filename
            self.assertTrue(csv_path.exists(), f"Split file missing: {csv_path}")
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                self.assertEqual(
                    len(rows),
                    expected_count,
                    f"Row count mismatch in {filename}: expected {expected_count}, got {len(rows)}",
                )

    def test_labels_in_canonical_domain(self):
        """Verify all label strings and label IDs in all splits belong strictly to canonical schemes."""
        for filename in EXPECTED_ROW_COUNTS:
            csv_path = PREPROC_DIR / filename
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    label_str = row.get("label")
                    label_id_str = row.get("label_id")
                    bin_label_str = row.get("binary_label")
                    bin_label_id_str = row.get("binary_label_id")

                    self.assertIn(
                        label_str,
                        VALID_LABELS_STR,
                        f"Invalid label '{label_str}' in {filename} row {idx}",
                    )
                    self.assertIn(
                        int(label_id_str),
                        VALID_LABEL_IDS,
                        f"Invalid label_id '{label_id_str}' in {filename} row {idx}",
                    )
                    self.assertIn(
                        bin_label_str,
                        VALID_BINARY_LABELS_STR,
                        f"Invalid binary_label '{bin_label_str}' in {filename} row {idx}",
                    )
                    self.assertIn(
                        int(bin_label_id_str),
                        VALID_BINARY_LABEL_IDS,
                        f"Invalid binary_label_id '{bin_label_id_str}' in {filename} row {idx}",
                    )

    def test_clean_texts_are_non_empty(self):
        """Verify text column contains valid non-empty strings."""
        for filename in EXPECTED_ROW_COUNTS:
            csv_path = PREPROC_DIR / filename
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    text = row.get("text", "").strip()
                    self.assertGreater(
                        len(text),
                        0,
                        f"Found empty text in {filename} row {idx}",
                    )


if __name__ == "__main__":
    unittest.main()
