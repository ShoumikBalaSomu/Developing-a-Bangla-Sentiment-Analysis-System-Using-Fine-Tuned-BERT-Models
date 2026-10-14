"""
test_artifact_security.py
=========================
Automated security test enforcing OWASP Top 10 and ML secret sanitization directives
(ml-artifact-sanitizer). Validates that no API keys, cloud secrets, or private tokens
are present in source code, configuration files, or notebooks.

CSE432 - Natural Language Processing
"""

from __future__ import annotations

from pathlib import Path
import unittest

from tools.sanitize_artifacts import ROOT, run_scan


class TestArtifactSecurity(unittest.TestCase):
    """Enforces zero hardcoded secrets and sanitized release artifacts."""

    def test_zero_secret_leakage_in_repository(self):
        """Scan workspace for leaked credentials; must pass with zero violations."""
        exit_code = run_scan(ROOT)
        self.assertEqual(
            exit_code,
            0,
            "Security audit failed: leaked credentials or tokens detected in repository files.",
        )


if __name__ == "__main__":
    unittest.main()
