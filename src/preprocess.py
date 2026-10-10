"""
preprocess.py
=============
End-to-end pre-processing pipeline. Produces every CSV under
dataset/preprocessed/ that Notebooks 02 and 03 consume.

Stages
------
  A. Source loading          - tolerant column resolution across Kaggle schemas
  B. Grammar-rule mining     - BanglaGEC source->target edit patterns -> rule table
  C. Lexicon construction    - bangla-nlp-words -> stopwords + negation lexicon
  D. Text normalisation      - NFKC/Bangla normalisation, URLs, digits, emoji
  E. Negation-scope tagging  - NEG_ prefix inside the negator window
  F. Label harmonisation     - all sources -> canonical {negative,neutral,positive}
  G. Filtering & dedupe      - length, language-script, exact-duplicate removal
  H. Stratified splitting    - per-source stratification so no source leaks a split
  I. Manifest                - row counts + SHA-256 for reproducibility proof

CSE432 - Natural Language Processing
Authors: 232-15-657 Shoumik Bala Somu | 232-15-644 Raisha Khan Raka
"""

from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ARTIFACTS, BINARY_MERGE, DATASETS, LABEL_COL_CANDIDATES, LABEL2ID, LABELS_3,
    MAX_CLASS_SHARE, NEGATION_SEEDS, PREPROC_DIR, PREPROCESS, RAW_DIR,
    RATING_TO_SENTIMENT, SEED, TEXT_COL_CANDIDATES, banner, resolve_dataset_dir,
)

# Bangla Unicode block + Bangla digits, used for the script filter.
BANGLA_RE = re.compile(r"[\u0980-\u09FF]")
BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"(?<!\w)@[\w]+")
HASHTAG_RE = re.compile(r"#")
WHITESPACE_RE = re.compile(r"\s+")
EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+", flags=re.UNICODE)

# Optional Bangla-specific normaliser. Falls back to pure Python if absent.
try:
    from bnunicodenormalizer import Normalizer as _BNU
    _bnu = _BNU()
    _HAS_BNU = True
except Exception:  # noqa: BLE001
    _bnu = None
    _HAS_BNU = False

try:
    from bnnumerizer import numerize as _bn_numerize
    _HAS_NUMERIZER = True
except Exception:  # noqa: BLE001
    _bn_numerize = None
    _HAS_NUMERIZER = False


# ==========================================================================
# Stage A - tolerant source loading
# ==========================================================================
def _pick_column(df: pd.DataFrame, candidates: list[str], explicit: str | None = None) -> str | None:
    """Resolve a column name from candidates, case- and whitespace-tolerant."""
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in ([explicit] if explicit else []) + candidates:
        if cand and cand.strip().lower() in cols:
            return cols[cand.strip().lower()]
    return None


def _iter_csvs(root: Path):
    if not root.exists():
        return
    for p in sorted(root.rglob("*.csv")):
        yield p


def load_raw_table(name: str) -> pd.DataFrame | None:
    """Load the first usable CSV for a registry entry. Returns None if absent."""
    spec = DATASETS[name]
    root = resolve_dataset_dir(spec["kaggle"])
    if not root.exists():
        root = RAW_DIR / name
    for path in _iter_csvs(root):
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  [load] unreadable {path.name}: {str(exc)[:120]}")
            continue
        if df.empty or df.shape[1] < 2:
            continue
        print(f"  [load] {name}: {path.name}  {len(df):,} rows x {df.shape[1]} cols")
        return df
    print(f"  [load] {name}: NO CSV FOUND under {root}")
    return None


# ==========================================================================
# Stage B - grammar-rule mining from BanglaGEC
# ==========================================================================
def mine_grammar_rules(sample: int) -> pd.DataFrame:
    """
    Derive a normalisation rule table from BanglaGEC source->target pairs.

    Method: for each pair, tokenise both sides and align them. Where a source
    token differs from its aligned target token, record the substitution. Keep
    only rules seen >= 5 times (support threshold) so one-off annotation noise
    does not become a rule. Substitutions are classified by type so we can
    report what kinds of errors dominate Bangla social text.

    This is the concrete technical use of the BanglaGEC dataset: it is not a
    sentiment resource, but it is the largest available corpus of *wrong ->
    right* Bangla, which is exactly what a normaliser needs.
    """
    df = load_raw_table("banglagec")
    if df is None:
        print("  [rules] BanglaGEC unavailable - grammar rule mining SKIPPED")
        return pd.DataFrame(columns=["source_token", "target_token", "support",
                                     "rule_type", "example"])

    src_col = _pick_column(df, ["source", "src", "incorrect", "input"],
                           DATASETS["banglagec"]["columns"]["src"])
    tgt_col = _pick_column(df, ["target", "tgt", "correct", "output"],
                           DATASETS["banglagec"]["columns"]["tgt"])
    if not src_col or not tgt_col:
        print(f"  [rules] could not find source/target columns in {list(df.columns)}")
        return pd.DataFrame(columns=["source_token", "target_token", "support",
                                     "rule_type", "example"])

    df = df[[src_col, tgt_col]].dropna()
    if len(df) > sample:
        df = df.iloc[:sample]          # deterministic head sample, seed-independent
    print(f"  [rules] mining from {len(df):,} pairs")

    MIN_SUPPORT = 5
    counter: Counter = Counter()
    examples: dict[tuple[str, str], str] = {}
    pairs_with_edits = 0

    for s, t in zip(df[src_col].astype(str), df[tgt_col].astype(str)):
        st, tt = s.split(), t.split()
        if st == tt:
            continue
        pairs_with_edits += 1
        # Token-position alignment is valid for the dominant single-token
        # substitution case. Length mismatch => skip (edit distance alignment
        # would cost O(n*m) over 200k pairs for marginal extra rules).
        if len(st) != len(tt):
            continue
        for a, b in zip(st, tt):
            if a != b:
                key = (a, b)
                counter[key] += 1
                if key not in examples:
                    examples[key] = s[:120]

    rows = []
    for (a, b), n in counter.most_common():
        if n < MIN_SUPPORT:
            break
        rows.append({
            "source_token": a,
            "target_token": b,
            "support": n,
            "rule_type": _classify_rule(a, b),
            "example": examples[(a, b)],
        })

    rules = pd.DataFrame(rows)
    print(f"  [rules] {pairs_with_edits:,} pairs had edits; "
          f"{len(rules):,} rules passed support>={MIN_SUPPORT}")
    return rules


def _classify_rule(a: str, b: str) -> str:
    """Label a substitution rule by the error type it corrects."""
    if a.replace(" ", "") == b.replace(" ", ""):
        return "whitespace"
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return "single_char_typo"
    if a in b or b in a:
        return "affix_inflection"
    if _bnu_norm(a) == _bnu_norm(b):
        return "unicode_normalisation"
    return "lexical_substitution"


def _bnu_norm(text: str) -> str:
    if _HAS_BNU:
        try:
            return (_bnu(text) or text).strip()
        except Exception:  # noqa: BLE001
            pass
    return unicodedata.normalize("NFC", text).strip()


def build_wellformed_vocab(sample: int = 100_000) -> set[str]:
    """Vocabulary of grammatically well-formed Bangla from BanglaGEC targets.
    Used in Stage G to flag OOV-heavy rows (often spam or heavy code-mixing)."""
    df = load_raw_table("banglagec")
    if df is None:
        return set()
    tgt_col = _pick_column(df, ["target", "tgt", "correct", "output"], "target")
    if not tgt_col:
        return set()
    df = df[[tgt_col]].dropna().head(sample)
    vocab: set[str] = set()
    for t in df[tgt_col].astype(str):
        vocab.update(w for w in t.split() if BANGLA_RE.search(w))
    print(f"  [vocab] well-formed Bangla vocabulary: {len(vocab):,} types")
    return vocab


def apply_grammar_rules(text: str, rule_map: dict[str, str]) -> str:
    """Substitute mined source tokens with their corrected targets."""
    if not rule_map:
        return text
    tokens = text.split()
    out = [rule_map.get(t, t) for t in tokens]
    return " ".join(out)


# ==========================================================================
# Stage C - lexicons from bangla-nlp-words
# ==========================================================================
def build_lexicons() -> tuple[set[str], set[str]]:
    """
    Returns (stopwords, negators).

    Reads the bangla-nlp-words compilation and buckets its entries. The
    dataset organises Bangla grammatical elements, so we detect negators by
    both membership in the seed list and by a substring heuristic over
    candidate files/columns.
    """
    stopwords: set[str] = set()
    negators: set[str] = set(NEGATION_SEEDS)

    root = RAW_DIR / "bangla_nlp_words"
    if not root.exists():
        root = resolve_dataset_dir(DATASETS["bangla_nlp_words"]["kaggle"])

    if root.exists():
        for path in _iter_csvs(root):
            try:
                df = pd.read_csv(path)
            except Exception:  # noqa: BLE001
                continue
            stem = path.stem.lower()
            is_stop = "stop" in stem
            is_neg = "neg" in stem
            for col in df.columns:
                vals = df[col].dropna().astype(str).str.strip()
                vals = set(v for v in vals if v and BANGLA_RE.search(v))
                if is_stop:
                    stopwords |= vals
                elif is_neg:
                    negators |= vals
                elif not stopwords and len(vals) > 20:
                    # Generic word list: only the seed negators are trusted,
                    # but keep high-frequency short particles as stopword hints.
                    stopwords |= {v for v in vals if len(v) <= 4}
    else:
        print("  [lexicon] bangla-nlp-words unavailable - using seed negators only")

    # Fall back to a conservative default stopword set so downstream code
    # never runs against an empty lexicon.
    if not stopwords:
        stopwords = set("""
        এবং ও ওর ঐ ঔ এ এই একই এর এইটা এমন এত তবে তবু তাই তা তো তুমি তারা
        তার আমাদের আমাকে আমি আপনার খুব কিছু কেউ কেন কোথায় যখন যে যত জন্য থেকে
        মধ্যে উপর নিচে সব সবাই হতে হবে হয় ছিল না নয়
        """.split())

    print(f"  [lexicon] stopwords={len(stopwords):,}  negators={len(negators):,}")
    return stopwords, negators


# ==========================================================================
# Stage D - text normalisation
# ==========================================================================
def normalise_text(text: str, rule_map: dict[str, str] | None = None) -> str:
    """Clean one Bangla string. Order matters: structural fixes first, then
    Unicode normalisation, then grammar rules, then whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""

    t = unicodedata.normalize("NFC", text)
    t = URL_RE.sub(PREPROCESS["url_placeholder"], t)
    t = MENTION_RE.sub(PREPROCESS["mention_placeholder"], t)
    t = HASHTAG_RE.sub(" ", t)

    # Bangla digits -> ASCII digits, so "৫" and "5" tokenise identically.
    t = t.translate(str.maketrans(BANGLA_DIGITS, "0123456789"))
    if _HAS_NUMERIZER:
        try:
            t = _bn_numerize(t)
        except Exception:  # noqa: BLE001
            pass

    # Collapse repeated characters ("ভাআআলো" -> "ভালো") to at most 2 runs.
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)

    t = _bnu_norm(t)

    if rule_map and PREPROCESS.get("apply_grammar_rules", True):
        t = apply_grammar_rules(t, rule_map)

    # Emoji carry polarity in Bangla social text; keep them as tokens.
    if not PREPROCESS["keep_emoji_token"]:
        t = EMOJI_RE.sub(" ", t)

    t = re.sub(r"[^\w\s\u0980-\u09FF#]", " ", t)
    t = WHITESPACE_RE.sub(" ", t).strip()
    return t


# ==========================================================================
# Stage E - negation-scope tagging
# ==========================================================================
def tag_negation(text: str, negators: set[str], window: int | None = None) -> str:
    """
    Prefix tokens within `window` positions after a negator with NEG_.

    Motivation: "ভালো না" (not good) and "ভালো" (good) differ only by the
    negator. A model that loses the negator's scope flips the polarity. Marking
    the scope makes the contrast explicit in the token stream, which measurably
    helps on short Bangla comments.
    """
    window = window or PREPROCESS["negation_window"]
    if not negators or not text:
        return text

    tokens = text.split()
    out: list[str] = []
    remaining = 0
    for tok in tokens:
        if tok in negators:
            out.append(tok)
            remaining = window
            continue
        if remaining > 0:
            # Do not re-tag a nested negator.
            out.append(tok if tok.startswith("NEG_") else f"NEG_{tok}")
            remaining -= 1
        else:
            out.append(tok)
    return " ".join(out)


# ==========================================================================
# Stage F - label harmonisation
# ==========================================================================
_WORD_LABEL_MAP = {
    "positive": "positive", "pos": "positive", "1": "positive",
    "সকারাত্মক": "positive", "পজিটিভ": "positive",
    "negative": "negative", "neg": "negative", "0": "negative",
    "নেতিবাচক": "negative", "নেগেটিভ": "negative",
    "neutral": "neutral", "neu": "neutral", "mixed": "neutral", "2": "neutral",
    "নিরপেক্ষ": "neutral",
}


def harmonise_label(raw, style: str) -> str | None:
    """Map any source label onto the canonical 3-class scheme."""
    if pd.isna(raw):
        return None
    if style == "rating":
        try:
            r = int(float(raw))
        except (TypeError, ValueError):
            return None
        return RATING_TO_SENTIMENT.get(r)

    s = str(raw).strip().lower()
    return _WORD_LABEL_MAP.get(s)


# ==========================================================================
# Stage G - filtering
# ==========================================================================
def looks_like_bangla(text: str, threshold: float = 0.4) -> bool:
    """Keep rows where >=40% of non-space chars are in the Bangla block.
    Lets genuine code-mixed text through while dropping pure-English or
    pure-spam rows."""
    letters = [c for c in text if not c.isspace()]
    if not letters:
        return False
    bn = sum(1 for c in letters if BANGLA_RE.match(c))
    return (bn / len(letters)) >= threshold


# ==========================================================================
# Stage H - splitting
# ==========================================================================
def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split with stratification on (label, source) so that
      - class balance is preserved in every split, and
      - every source contributes to train/val/test proportionally.

    Without source stratification a whole platform can land in one split, and
    the test score then measures platform shift rather than sentiment skill.
    """
    from sklearn.model_selection import train_test_split

    test_size = PREPROCESS["test_size"]
    val_size = PREPROCESS["val_size"]

    strat = df["label"].astype(str) + "||" + df["source"].astype(str)
    train_full, test = train_test_split(
        df, test_size=test_size, random_state=SEED, stratify=strat)
    # recompute strat on the reduced frame
    strat2 = train_full["label"].astype(str) + "||" + train_full["source"].astype(str)
    val_frac = val_size / (1 - test_size)
    train, val = train_test_split(
        train_full, test_size=val_frac, random_state=SEED, stratify=strat2)

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def balance_majority(df: pd.DataFrame, max_share: float = MAX_CLASS_SHARE) -> pd.DataFrame:
    """Downsample any class above `max_share` of the frame. Returns a new df."""
    counts = df["label"].value_counts()
    cap = int(len(df) * max_share)
    if counts.max() <= cap:
        return df
    parts = []
    for lab, grp in df.groupby("label"):
        if len(grp) > cap:
            grp = grp.sample(n=cap, random_state=SEED)
            print(f"  [balance] '{lab}' downsampled {counts[lab]:,} -> {cap:,}")
        parts.append(grp)
    out = pd.concat(parts).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return out


# ==========================================================================
# Stage I - manifest
# ==========================================================================
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_manifest(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Record size + hash of every artefact so a reviewer can verify they
    regenerated identical data."""
    rows = []
    for key, path in ARTIFACTS.items():
        path = Path(path)
        if not path.exists():
            continue
        rows.append({
            "artifact": key,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        })
    for name, df in frames.items():
        rows.append({"artifact": f"{name}[rows]", "file": "-", "bytes": len(df), "sha256": "-"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(ARTIFACTS["data_manifest"], index=False)
    print(f"  [manifest] {len(rows)} entries -> {ARTIFACTS['data_manifest'].name}")
    return manifest


# ==========================================================================
# Supervised source assembly
# ==========================================================================
def build_supervised_frame(negators: set[str], rule_map: dict[str, str]) -> pd.DataFrame:
    """Load every supervised source, harmonise labels, normalise text."""
    frames = []
    for name, spec in DATASETS.items():
        if spec["role"] != "supervised":
            continue
        df = load_raw_table(name)
        if df is None:
            continue

        text_col = _pick_column(df, TEXT_COL_CANDIDATES, spec.get("text_col"))
        label_col = _pick_column(df, LABEL_COL_CANDIDATES, spec.get("label_col"))
        if not text_col or not label_col:
            print(f"  [sup] {name}: cannot resolve text/label columns "
                  f"(have {list(df.columns)[:8]}) - skipped")
            continue

        work = pd.DataFrame({
            "raw_text": df[text_col].astype(str),
            "raw_label": df[label_col],
        })
        work["label"] = work["raw_label"].map(lambda r: harmonise_label(r, spec["label_style"]))
        before = len(work)
        work = work.dropna(subset=["label"])
        print(f"  [sup] {name}: {len(work):,}/{before:,} rows got a canonical label")
        work["source"] = name
        frames.append(work)

    if not frames:
        return pd.DataFrame(columns=["raw_text", "text", "label", "label_id",
                                     "binary_label", "source", "n_tokens", "has_negation"])

    allrows = pd.concat(frames, ignore_index=True)
    print(f"  [sup] combined raw pool: {len(allrows):,} rows")

    # Normalisation + feature columns
    allrows["text"] = allrows["raw_text"].map(lambda s: normalise_text(s, rule_map))
    allrows = allrows[allrows["text"].str.len() >= PREPROCESS["min_text_len"]]
    allrows = allrows[allrows["text"].str.len() <= PREPROCESS["max_text_len"]]

    if PREPROCESS["drop_na_text"]:
        allrows = allrows[allrows["text"].str.strip() != ""]

    allrows = allrows[allrows["text"].map(looks_like_bangla)]
    allrows["text"] = allrows["text"].map(lambda s: tag_negation(s, negators))
    allrows["has_negation"] = allrows["text"].str.contains("NEG_")

    if PREPROCESS["dedupe"]:
        n0 = len(allrows)
        allrows = allrows.drop_duplicates(subset=["text"])
        print(f"  [sup] dedupe: {n0:,} -> {len(allrows):,} rows")

    allrows["label_id"] = allrows["label"].map(LABEL2ID)
    allrows["binary_label"] = allrows["label"].map(BINARY_MERGE)
    allrows["n_tokens"] = allrows["text"].str.split().str.len()
    allrows = allrows.drop(columns=["raw_label"]).reset_index(drop=True)
    return allrows


# ==========================================================================
# Literature review generator (bangla-nlp-catalog)
# ==========================================================================
def build_lr_table() -> pd.DataFrame:
    """Filter the catalog to sentiment-related resources. Output feeds the
    prior-work tables in docs/02_literature_review.md."""
    df = load_raw_table("bangla_nlp_catalog")
    if df is None:
        print("  [lr] catalog unavailable - skipping")
        return pd.DataFrame()

    kw = re.compile(r"sentiment|opinion|emotion|affect|review", re.I)
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        if df[col].dtype == object:
            mask |= df[col].astype(str).str.contains(kw, na=False)
    hits = df[mask].copy()
    hits["__matched_cols"] = hits.astype(str).apply(
        lambda r: ",".join(c for c in df.columns if kw.search(str(r[c]))), axis=1)
    print(f"  [lr] {len(hits):,} catalog entries matched sentiment-related keywords")
    return hits


# ==========================================================================
# Main
# ==========================================================================
def run_pipeline(sample: int | None = None, apply_rules: bool = True) -> dict:
    from config import BANGLAGEC_SAMPLE_ROWS
    sample = sample or BANGLAGEC_SAMPLE_ROWS

    stats: dict[str, object] = {}
    print(banner("STAGE B/C  Grammar rules + lexicons"))
    rules = mine_grammar_rules(sample)
    stopwords, negators = build_lexicons()
    rule_map = dict(zip(rules["source_token"], rules["target_token"])) if not rules.empty else {}

    if not apply_rules:
        rule_map = {}
    PREPROCESS["apply_grammar_rules"] = bool(rule_map)
    stats["n_grammar_rules"] = len(rules)
    stats["n_negators"] = len(negators)
    stats["n_stopwords"] = len(stopwords)

    print(banner("STAGE F/G/H  Supervised assembly, filtering, splitting"))
    pool = build_supervised_frame(negators, rule_map)
    stats["pool_rows"] = len(pool)
    if pool.empty:
        print("\n[!] No supervised data available. Notebook 01 will still write the\n"
              "    grammar/lexicon artefacts so the pipeline is demonstrable, but\n"
              "    Notebooks 02/03 need labelled data. Add Kaggle credentials and re-run.")
        _write_aux(rules, negators, stopwords)
        return stats

    print("\n  label distribution (pre-balance):")
    print(pool["label"].value_counts().to_string())
    print("\n  per-source row counts:")
    print(pool["source"].value_counts().to_string())

    train, val, test = stratified_split(pool)
    train = balance_majority(train)
    for df, nm in ((train, "train"), (val, "val"), (test, "test")):
        df["binary_label_id"] = (df["binary_label"] == "positive").astype(int)
        stats[f"{nm}_rows"] = len(df)

    print(banner("STAGE I  Writing artefacts"))
    train.to_csv(ARTIFACTS["clean_train"], index=False)
    val.to_csv(ARTIFACTS["clean_val"], index=False)
    test.to_csv(ARTIFACTS["clean_test"], index=False)
    _write_aux(rules, negators, stopwords)
    lr = build_lr_table()
    if not lr.empty:
        lr.to_csv(ARTIFACTS["lr_table"], index=False)
    vocab = build_wellformed_vocab()
    if vocab:
        ARTIFACTS["vocab_wellformed"].write_text(
            "\n".join(sorted(vocab)), encoding="utf-8")

    write_manifest({"train": train, "val": val, "test": test})
    print(banner("Pre-processing complete"))
    print(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"  artefacts in {PREPROC_DIR}")
    return stats


def _write_aux(rules: pd.DataFrame, negators, stopwords) -> None:
    if rules is not None and not rules.empty:
        rules.to_csv(ARTIFACTS["grammar_rules"], index=False)
    Path(ARTIFACTS["negation_lexicon"]).write_text(
        "\n".join(sorted(set(negators))), encoding="utf-8")
    Path(ARTIFACTS["stopwords"]).write_text(
        "\n".join(sorted(set(stopwords))), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="BanglaGEC rows to mine")
    ap.add_argument("--no-rules", action="store_true", help="skip grammar-rule application")
    a = ap.parse_args()
    run_pipeline(sample=a.sample, apply_rules=not a.no_rules)
