import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path as MplPath
from pathlib import Path

def draw_system_overview(output_paths):
    # Dimensions: 19.6 x 10.8 inches for wide inter-column corridors and zero label collisions
    fig = plt.figure(figsize=(19.6, 10.8), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 19.6)
    ax.set_ylim(0, 10.8)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    PALETTE = {
        "title_bg": "#0f172a",
        "c1_border": "#0284c7",
        "c1_bg": "#f0f9ff",
        "c1_badge": "#0369a1",
        "c1_box": "#e0f2fe",

        "c2_border": "#6366f1",
        "c2_bg": "#f5f3ff",
        "c2_badge": "#4f46e5",
        "c2_box": "#ede9fe",

        "c3_border": "#059669",
        "c3_bg": "#ecfdf5",
        "c3_badge": "#047857",
        "c3_box": "#d1fae5",

        "c4_border": "#d97706",
        "c4_bg": "#fffbeb",
        "c4_badge": "#b45309",
        "c4_box": "#fef3c7",

        "dark": "#0f172a",
        "muted": "#475569",
    }

    def draw_card(x, y, w, h, bg, border, lw=1.5, radius=0.16, zorder=2):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=border, facecolor=bg, zorder=zorder
        )
        ax.add_patch(rect)
        return rect

    def draw_ortho_arrow(points, color, lw=2.2, zorder=6):
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        path = MplPath(points, codes)
        patch = patches.FancyArrowPatch(
            path=path,
            arrowstyle="->,head_width=5,head_length=7",
            lw=lw,
            color=color,
            zorder=zorder,
            joinstyle="round",
            capstyle="round",
        )
        ax.add_patch(patch)
        return patch

    def draw_pill_label(x, y, text, text_color, border_color, bg_color="#ffffff", fontsize=7.4, zorder=10):
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color,
            zorder=zorder,
            bbox=dict(
                boxstyle="round,pad=0.24,rounding_size=0.14",
                facecolor=bg_color,
                edgecolor=border_color,
                linewidth=1.15,
            ),
        )

    # -------------------------------------------------------------
    # Column Setup (4 columns with 1.14-inch inter-column corridors)
    # -------------------------------------------------------------
    col_w = 3.74
    col_gap = 1.14
    start_x = 0.61
    cols_x = [start_x + i * (col_w + col_gap) for i in range(4)]
    total_w = (cols_x[-1] + col_w) - start_x
    base_y = 1.15
    card_h = 8.35

    # -------------------------------------------------------------
    # 0. Title Banner
    # -------------------------------------------------------------
    draw_card(start_x, 9.75, total_w, 0.80, PALETTE["title_bg"], "#1e293b", lw=1.5, radius=0.2)
    ax.text(start_x + total_w / 2, 10.25, "CSE432 — Bangla Sentiment Analysis System Pipeline & Architecture",
            ha="center", va="center", color="#ffffff", fontsize=16.0, fontweight="bold")
    ax.text(start_x + total_w / 2, 9.93, "End-to-End Workflow: Auxiliary Resource Mining  →  Preprocessing & Harmonisation  →  Unified Modeling  →  Empirical Evaluation",
            ha="center", va="center", color="#cbd5e1", fontsize=10.5)

    col_meta = [
        ("1. Auxiliary Resource Mining", PALETTE["c1_bg"], PALETTE["c1_border"], PALETTE["c1_badge"]),
        ("2. Preprocessing & Splits", PALETTE["c2_bg"], PALETTE["c2_border"], PALETTE["c2_badge"]),
        ("3. Unified Modeling (6 Models)", PALETTE["c3_bg"], PALETTE["c3_border"], PALETTE["c3_badge"]),
        ("4. Empirical Diagnostics & Slices", PALETTE["c4_bg"], PALETTE["c4_border"], PALETTE["c4_badge"]),
    ]

    for i, (col_title, bg, border, badge_bg) in enumerate(col_meta):
        cx = cols_x[i]
        draw_card(cx, base_y, col_w, card_h, bg, border, lw=1.8, radius=0.22, zorder=1)
        draw_card(cx, base_y + card_h - 0.54, col_w, 0.54, badge_bg, border, lw=1.0, radius=0.20, zorder=3)
        ax.text(cx + col_w / 2, base_y + card_h - 0.27, col_title, ha="center", va="center",
                color="#ffffff", fontsize=11.5, fontweight="bold", zorder=4)

    # =============================================================
    # Column 1: Auxiliary Resource Mining
    # =============================================================
    c1_x = cols_x[0] + 0.16
    c1_w = col_w - 0.32

    # 1A: BanglaGEC
    draw_card(c1_x, 6.55, c1_w, 2.20, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w / 2, 8.50, "BanglaGEC Corpus (nahidhub)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c1_x + c1_w / 2, 8.26, "~7.07M parallel sentence pairs", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.4, fontstyle="italic")
    ax.text(c1_x + c1_w / 2, 7.68, "[Stage B] Grammar Rule Mining\n• 200,000 deterministic head sample\n• Positional substitution mapping\n• Minimum support threshold ≥ 5",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.4, linespacing=1.32)
    draw_card(c1_x + 0.14, 6.67, c1_w - 0.28, 0.44, "#ffffff", "#0284c7", lw=1.1)
    ax.text(c1_x + c1_w / 2, 6.89, "grammar_rules.csv (2,238 rules)", ha="center", va="center",
            color="#0369a1", fontsize=8.8, fontweight="bold")

    # 1B: Bangla NLP Words
    draw_card(c1_x, 3.90, c1_w, 2.42, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w / 2, 6.07, "Bangla NLP Words (saifsunny)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c1_x + c1_w / 2, 5.83, "Grammatical particles & stopwords", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.4, fontstyle="italic")
    ax.text(c1_x + c1_w / 2, 5.26, "[Stage C] Lexicon Construction\n• Stopwords extraction (813 words)\n• Negator detection (14 tokens)\n• Negators strictly preserved from deletion",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.4, linespacing=1.32)
    draw_card(c1_x + 0.14, 4.02, c1_w - 0.28, 0.66, "#ffffff", "#0284c7", lw=1.1)
    ax.text(c1_x + c1_w / 2, 4.48, "negation_lexicon.txt (14 negators)", ha="center", va="center",
            color="#0369a1", fontsize=8.4, fontweight="bold")
    ax.text(c1_x + c1_w / 2, 4.20, "bangla_stopwords.txt (813 words)", ha="center", va="center",
            color="#0369a1", fontsize=8.4, fontweight="bold")

    # 1C: Bangla NLP Catalog
    draw_card(c1_x, 1.30, c1_w, 2.38, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w / 2, 3.43, "Bangla NLP Catalog (kishormorol)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c1_x + c1_w / 2, 3.19, "813 papers, 63 datasets, 26 tasks", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.4, fontstyle="italic")
    ax.text(c1_x + c1_w / 2, 2.63, "[Catalog] Literature Review Generation\n• Filter: sentiment | opinion | emotion\n• Year-ranked metadata synthesis\n• Auditable prior work selection",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.4, linespacing=1.32)
    draw_card(c1_x + 0.14, 1.42, c1_w - 0.28, 0.66, "#ffffff", "#0284c7", lw=1.1)
    ax.text(c1_x + c1_w / 2, 1.88, "literature_review_table.csv (166 items)", ha="center", va="center",
            color="#0369a1", fontsize=8.3, fontweight="bold")
    ax.text(c1_x + c1_w / 2, 1.60, "prior_work.md (Markdown synthesis)", ha="center", va="center",
            color="#0369a1", fontsize=8.3, fontweight="bold")

    # =============================================================
    # Column 2: Preprocessing & Splits
    # =============================================================
    c2_x = cols_x[1] + 0.16
    c2_w = col_w - 0.32
    c2_mid = c2_x + c2_w / 2

    # 2A: Supervised Sources
    draw_card(c2_x, 7.35, c2_w, 1.40, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_mid, 8.50, "Raw Supervised Corpora", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c2_mid, 7.90, "• Daraz BD (e-commerce customer ratings)\n• RevBangla (curated product polarities)\n• LargeSentiment (social media text)\nCandidate Pool: 113,959 total rows",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.3, linespacing=1.28)

    draw_ortho_arrow([(c2_mid, 7.35), (c2_mid, 7.03)], "#4f46e5", lw=1.8, zorder=5)

    # 2B: Normalisation & Negation Scope
    draw_card(c2_x, 5.25, c2_w, 1.78, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_mid, 6.78, "[Stage D & E] Normalisation Pipeline", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.2, fontweight="bold")
    ax.text(c2_mid, 5.98, "• Unicode NFC + bnunicodenormalizer\n• URL / User mention tokenization\n• Bangla digits → ASCII; char collapse\n• Negation tagging: 3-token forward NEG_\n• Substitution: grammar_rules.csv",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.3, linespacing=1.28)

    draw_ortho_arrow([(c2_mid, 5.25), (c2_mid, 4.93)], "#4f46e5", lw=1.8, zorder=5)

    # 2C: Harmonisation & Filtering
    draw_card(c2_x, 3.35, c2_w, 1.58, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_mid, 4.68, "[Stage F & G] Harmonisation & Filter", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.2, fontweight="bold")
    ax.text(c2_mid, 3.98, "• Canonical 3-class: {neg, neu, pos}\n• Rating map: 1-2→neg, 3→neu, 4-5→pos\n• Length 3–1000 chars; Bangla ratio ≥ 0.4\n• Exact-match deduplication (8,573 clean)",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.3, linespacing=1.28)

    draw_ortho_arrow([(c2_mid, 3.35), (c2_mid, 3.03)], "#4f46e5", lw=1.8, zorder=5)

    # 2D: Joint Stratified Splitting
    draw_card(c2_x, 1.30, c2_w, 1.73, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_mid, 2.80, "[Stage H] Stratified Splitting", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.2, fontweight="bold")
    ax.text(c2_mid, 2.48, "Stratified on (label, source) • 60% majority cap",
            ha="center", va="center", color=PALETTE["muted"], fontsize=7.9)
    draw_card(c2_x + 0.12, 1.40, c2_w - 0.24, 0.90, "#ffffff", "#6366f1", lw=1.1)
    ax.text(c2_mid, 2.08, "train_clean.csv : 5,797 rows (70%)", ha="center", va="center",
            color="#4338ca", fontsize=8.4, fontweight="bold")
    ax.text(c2_mid, 1.82, "val_clean.csv   : 1,388 rows (15%)", ha="center", va="center",
            color="#4338ca", fontsize=8.4, fontweight="bold")
    ax.text(c2_mid, 1.56, "test_clean.csv  : 1,388 rows (15%)", ha="center", va="center",
            color="#b91c1c", fontsize=8.4, fontweight="bold")

    # =============================================================
    # Column 3: Unified Modeling Framework
    # =============================================================
    c3_x = cols_x[2] + 0.16
    c3_w = col_w - 0.32
    c3_mid = c3_x + c3_w / 2

    # 3A: Training Recipe
    draw_card(c3_x, 6.60, c3_w, 2.15, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_mid, 8.50, "Unified Training Protocol", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c3_mid, 7.52, "• AdamW with 2 discriminative LR groups\n• Backbone LR: 2e-5 | Head LR: 6e-5\n• Batch: 16 | Epochs: 4 | Warmup: 10%\n• Max Seq Len: 128 | Label Smoothing: 0.05\n• Inverse-frequency class weights (mean=1)\n• Early stopping: 2 epochs on Macro-F1",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.2, linespacing=1.30)

    draw_ortho_arrow([(c3_mid, 6.60), (c3_mid, 6.37)], "#059669", lw=1.8, zorder=5)

    # 3B: Bangla-Specific Models
    draw_card(c3_x, 4.55, c3_w, 1.82, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_mid, 6.13, "Language-Specific Backbones", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.0, fontweight="bold")

    draw_card(c3_x + 0.10, 5.32, c3_w - 0.20, 0.56, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_mid, 5.60, "BanglaBERT (csebuetnlp/banglabert)\nELECTRA Discriminator • 27.5 GB Bangla corpus",
            ha="center", va="center", color="#065f46", fontsize=7.6, fontweight="bold", linespacing=1.25)

    draw_card(c3_x + 0.10, 4.65, c3_w - 0.20, 0.56, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_mid, 4.93, "BanglishBERT (csebuetnlp/banglishbert)\nCode-Mixed Pretrained ELECTRA",
            ha="center", va="center", color="#065f46", fontsize=7.6, fontweight="bold", linespacing=1.25)

    # 3C: Multilingual & BERT-Base
    draw_card(c3_x, 2.70, c3_w, 1.62, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_mid, 4.08, "BERT-Base & Multilingual Baselines", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.0, fontweight="bold")
    draw_card(c3_x + 0.10, 3.38, c3_w - 0.20, 0.48, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_mid, 3.62, "Bangla BERT Base (sagorsarker/bangla-bert-base)", ha="center", va="center",
            color="#065f46", fontsize=7.6, fontweight="bold")
    draw_card(c3_x + 0.10, 2.80, c3_w - 0.20, 0.48, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_mid, 3.04, "mBERT (~178M) & XLM-RoBERTa (~270M)", ha="center", va="center",
            color="#065f46", fontsize=7.6, fontweight="bold")

    # 3D: Classical Baseline
    draw_card(c3_x, 1.30, c3_w, 1.18, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_mid, 2.26, "Classical N-Gram Reference", ha="center", va="center",
            color=PALETTE["dark"], fontsize=9.8, fontweight="bold")
    draw_card(c3_x + 0.10, 1.40, c3_w - 0.20, 0.64, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_mid, 1.82, "TF-IDF + Logistic Regression", ha="center", va="center",
            color="#065f46", fontsize=8.4, fontweight="bold")
    ax.text(c3_mid, 1.56, "Word (60k) + Char (80k) = 140k n-gram features", ha="center", va="center",
            color=PALETTE["muted"], fontsize=7.5)

    # =============================================================
    # Column 4: Empirical Diagnostics & Slices
    # =============================================================
    c4_x = cols_x[3] + 0.16
    c4_w = col_w - 0.32
    c4_mid = c4_x + c4_w / 2

    # 4A: Leaderboard
    draw_card(c4_x, 6.45, c4_w, 2.30, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_mid, 8.50, "Test Split Leaderboard (1,388)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    draw_card(c4_x + 0.10, 6.55, c4_w - 0.20, 1.72, "#ffffff", "#f59e0b", lw=1.0)

    leader_rows = [
        ("TF-IDF Baseline", "0.6341", "88.54%"),
        ("BanglishBERT", "0.6065", "82.78%"),
        ("BanglaBERT", "0.5985", "83.00%"),
        ("mBERT", "0.5843", "79.32%"),
        ("Bangla BERT Base", "0.5806", "77.31%"),
        ("XLM-RoBERTa", "0.5760", "79.68%"),
    ]
    col_model_x = c4_x + 0.22
    col_f1_x = c4_x + 2.30
    col_acc_x = c4_x + c4_w - 0.22

    ax.text(col_model_x, 8.06, "Model", ha="left", va="center", fontweight="bold", fontsize=8.1, color=PALETTE["dark"])
    ax.text(col_f1_x, 8.06, "Macro-F1", ha="right", va="center", fontweight="bold", fontsize=8.1, color=PALETTE["dark"])
    ax.text(col_acc_x, 8.06, "Accuracy", ha="right", va="center", fontweight="bold", fontsize=8.1, color=PALETTE["dark"])
    ax.plot([c4_x + 0.18, c4_x + c4_w - 0.18], [7.91, 7.91], color="#fde68a", lw=1.0, zorder=4)

    for idx, (m_name, f1, acc) in enumerate(leader_rows):
        ry = 7.74 - idx * 0.21
        ax.text(col_model_x, ry, m_name, ha="left", va="center", fontsize=7.9, color=PALETTE["dark"])
        ax.text(col_f1_x, ry, f1, ha="right", va="center", fontsize=7.9, fontweight="bold", color="#b45309")
        ax.text(col_acc_x, ry, acc, ha="right", va="center", fontsize=7.9, color=PALETTE["muted"])

    # 4B: Robustness Slices
    draw_card(c4_x, 4.55, c4_w, 1.68, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_mid, 6.00, "3 Robustness Slices", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c4_mid, 5.22, "• Source: RevBangla (91.0%) vs Daraz (73.7%)\n  (17.26% spread exposes noisy star ratings)\n• Negation: Non-neg (85.7%) vs Negated (65.6%)\n  (20.10% deficit reveals post-verbal syntax)\n• Length: 1–5 tokens (88.3%) down to 41+ (61.1%)",
            ha="center", va="center", color=PALETTE["dark"], fontsize=7.8, linespacing=1.26)

    # 4C: Ablation
    draw_card(c4_x, 2.85, c4_w, 1.50, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_mid, 4.12, "BanglaGEC Rule Ablation", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    ax.text(c4_mid, 3.45, "• With Rules: 0.5985 F1 (83.00% Acc)\n• Without Rules: 0.6228 F1 (86.34% Acc)\n• Net Delta: −2.43 pp Macro-F1 (−3.34 pp Acc)\nFinding: 51.5% lexical substitution rules hurt\nby regularizing informal polarity keywords.",
            ha="center", va="center", color=PALETTE["dark"], fontsize=7.8, linespacing=1.26)

    # 4D: Diagnostic Figures & Artefacts
    draw_card(c4_x, 1.30, c4_w, 1.35, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_mid, 2.42, "13 Visual Diagnostic Plots", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.3, fontweight="bold")
    draw_card(c4_x + 0.10, 1.40, c4_w - 0.20, 0.80, "#ffffff", "#f59e0b", lw=1.0)
    ax.text(c4_mid, 1.80, "• 5 Confusion Matrices + 5 ROC Curves\n• Macro-F1 & Accuracy Comparison Bars\n• Stratified Label Distribution Plot\n• 25 Hand-Audited Error Cases",
            ha="center", va="center", color="#92400e", fontsize=7.8, fontweight="bold", linespacing=1.26)

    # =============================================================
    # Clean Orthogonal Inter-Column Pipeline Connectors
    # =============================================================
    gap12_mid = (cols_x[0] + col_w + cols_x[1]) / 2
    gap23_mid = (cols_x[1] + col_w + cols_x[2]) / 2
    gap34_mid = (cols_x[2] + col_w + cols_x[3]) / 2

    # 1. Grammar Rules (Col 1) -> Stage D & E Normalisation (Col 2)
    draw_ortho_arrow(
        [(c1_x + c1_w - 0.14, 6.89), (gap12_mid, 6.89), (gap12_mid, 6.35), (c2_x, 6.35)],
        color="#0284c7", lw=2.2, zorder=6
    )
    draw_pill_label(gap12_mid, 7.28, "Mined Rules\n(2,238)", "#0369a1", "#0284c7", "#f0f9ff", fontsize=7.3)

    # 2. Negation Lexicon (Col 1) -> Stage D & E Normalisation (Col 2)
    draw_ortho_arrow(
        [(c1_x + c1_w - 0.14, 4.35), (gap12_mid, 4.35), (gap12_mid, 5.65), (c2_x, 5.65)],
        color="#0284c7", lw=2.2, zorder=6
    )
    draw_pill_label(gap12_mid, 5.00, "Lexicon\n(14 neg)", "#0369a1", "#0284c7", "#f0f9ff", fontsize=7.3)

    # 3. Train & Val Splits (Col 2) -> Unified Training Protocol (Col 3)
    draw_ortho_arrow(
        [(c2_x + c2_w - 0.12, 1.95), (gap23_mid, 1.95), (gap23_mid, 7.55), (c3_x, 7.55)],
        color="#4f46e5", lw=2.3, zorder=6
    )
    draw_pill_label(gap23_mid, 4.75, "Train & Val\n(7,185 rows)", "#4338ca", "#6366f1", "#f5f3ff", fontsize=7.3)

    # 4. Trained Checkpoints (Col 3) -> Evaluation Leaderboard (Col 4)
    draw_ortho_arrow(
        [(c3_x + c3_w, 7.55), (c4_x, 7.55)],
        color="#059669", lw=2.3, zorder=6
    )
    draw_pill_label(gap34_mid, 8.02, "Evaluated\nCheckpoints", "#047857", "#059669", "#ecfdf5", fontsize=7.3)

    # 5. Held-Out Test Split (Col 2 bottom) -> Bottom Evaluation Bus -> Col 4 bottom
    bus_y = 0.52
    draw_ortho_arrow(
        [(c2_mid, 1.40), (c2_mid, bus_y), (c4_mid, bus_y), (c4_mid, 1.15)],
        color="#dc2626", lw=2.3, zorder=5
    )

    bus_center_x = (c2_mid + c4_mid) / 2
    bus_w = 6.80
    draw_card(bus_center_x - bus_w / 2, bus_y - 0.24, bus_w, 0.48,
              "#fef2f2", "#dc2626", lw=1.4, radius=0.14, zorder=7)
    ax.text(bus_center_x, bus_y,
            "Held-Out Test Split (1,388 samples) — Zero Cross-Platform Leakage",
            ha="center", va="center", color="#991b1b", fontsize=8.8, fontweight="bold", zorder=8)

    plt.tight_layout()
    for p in output_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    targets = [
        ROOT / "results" / "system_overview.png",
        ROOT / "docs" / "assets" / "images" / "system_overview.png",
    ]
    draw_system_overview(targets)
