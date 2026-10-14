#!/usr/bin/env python3
"""
sanitize_artifacts.py
=====================
Automated artifact sanitizer and credential scanner for zero-secret release gates.
Enforces OWASP Top 10 and ML Security directives (ml-artifact-sanitizer).

CSE432 - Natural Language Processing
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent.parent

# Directories to ignore during scan
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode",
    ".ipynb_checkpoints",
    "node_modules",
    "raw",          # Uncommitted 3GB raw corpora (BanglaGEC)
    "checkpoints",  # Model checkpoints
    "runs",         # Tensorboard runs
}

EXCLUDED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".docx",
    ".zip",
    ".tar",
    ".gz",
    ".pyc",
    ".bin",
    ".safetensors",
}

# Max file size to scan (skip files larger than 10MB to avoid scanning raw text dumps)
MAX_FILE_BYTES = 10 * 1024 * 1024

# Regex patterns for sensitive tokens and private credentials
SECRET_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("HuggingFace User Token", re.compile(r"\bhf_[A-Za-z0-9]{34,}\b")),
    ("GitHub Personal Access Token", re.compile(r"\bghp_[A-Za-z0-9]{36}\b")),
    ("GitHub Fine-Grained Token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{80,}\b")),
    ("Kaggle API Key in JSON/Config", re.compile(r'["\'](?:key|KAGGLE_KEY)["\']\s*:\s*["\'][a-f0-9]{32}["\']', re.IGNORECASE)),
    ("AWS Access Key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS Secret Key Assignment", re.compile(r'(?i)aws_secret_access_key\s*=\s*["\'][A-Za-z0-9/+=]{40}["\']')),
    ("OpenSSH / RSA Private Key", re.compile(r"-----BEGIN (?:RSA|OPENSSH|DSA|EC|PGP)?\s*PRIVATE KEY-----")),
    ("Slack API Token", re.compile(r"\bxox[baprs]-[0-9a-zA-Z]{10,48}\b")),
    ("OpenAI API Key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b")),
    ("Workstation Agent Path Leak", re.compile(r"(?:qwen-agent[/\\]lUinoenfmc|D:[/\\]Documents[/\\]qwen-agent)")),
]


def mask_secret(matched_text: str) -> str:
    """Mask sensitive string preserving length clues."""
    if len(matched_text) <= 6:
        return "***"
    return matched_text[:3] + "..." + matched_text[-3:]


def scan_file(file_path: Path) -> list[tuple[int, str, str]]:
    """Scan a single file for known credential patterns."""
    findings = []
    try:
        if file_path.stat().st_size > MAX_FILE_BYTES:
            return findings

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                # Skip comments explaining patterns or test files checking patterns
                if "SECRET_PATTERNS" in line or "mask_secret" in line:
                    continue
                for name, pattern in SECRET_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        findings.append((line_no, name, mask_secret(match.group(0))))
    except Exception as exc:
        print(f"[WARN] Could not read {file_path}: {exc}", file=sys.stderr)
    return findings


def run_scan(root_dir: Path) -> int:
    """Scan root_dir recursively and return number of violations."""
    total_scanned = 0
    all_violations: list[tuple[Path, int, str, str]] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Exclude directories in-place
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for fname in filenames:
            path = Path(dirpath) / fname
            if path.suffix.lower() in EXCLUDED_EXTENSIONS:
                continue

            # Don't self-flag the sanitizer script itself or the security test fixture
            if path.name in ("sanitize_artifacts.py", "test_artifact_security.py"):
                continue

            total_scanned += 1
            findings = scan_file(path)
            for line_no, name, masked in findings:
                all_violations.append((path.relative_to(root_dir), line_no, name, masked))

    print(f"\n[Artifact & Secret Sanitizer] Scanned {total_scanned} files across repository.")

    if not all_violations:
        print("[SUCCESS] Zero hardcoded secrets or credentials detected. Gate: PASS (SANITIZED_FOR_RELEASE)\n")
        return 0

    print(f"[CRITICAL FAIL] Found {len(all_violations)} potential secret leaks:", file=sys.stderr)
    for rel_path, line_no, name, masked in all_violations:
        print(f"  - {rel_path}:{line_no} [{name}]: {masked}", file=sys.stderr)
    return 1


def main() -> None:
    exit_code = run_scan(ROOT)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
