import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_system_overview(output_paths):
    # Dimensions: 17 x 10 inches for expansive, crisp typography
    fig = plt.figure(figsize=(17, 10.2), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    # Clean background
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
        "arrow": "#334155"
    }

    def draw_card(x, y, w, h, bg, border, lw=1.5, radius=0.18, zorder=2):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            linewidth=lw, edgecolor=border, facecolor=bg, zorder=zorder
        )
        ax.add_patch(rect)
        return rect

    # -------------------------------------------------------------
    # 0. Title Banner
    # -------------------------------------------------------------
    draw_card(0.5, 9.25, 16.0, 0.75, PALETTE["title_bg"], "#1e293b", lw=1.5, radius=0.2)
    ax.text(8.5, 9.72, "CSE432 — Bangla Sentiment Analysis System Pipeline & Architecture",
            ha="center", va="center", color="#ffffff", fontsize=15.5, fontweight="bold")
    ax.text(8.5, 9.42, "End-to-End Workflow: Auxiliary Resource Mining → Preprocessing & Harmonisation → Unified Modeling → Empirical Evaluation",
            ha="center", va="center", color="#94a3b8", fontsize=10.5)

    # -------------------------------------------------------------
    # Column Setup
    # -------------------------------------------------------------
    col_w = 3.65
    col_gap = 0.46
    start_x = 0.5
    cols_x = [start_x + i * (col_w + col_gap) for i in range(4)]
    base_y = 0.95
    card_h = 8.05

    col_meta = [
        ("1. Auxiliary Resource Mining", PALETTE["c1_bg"], PALETTE["c1_border"], PALETTE["c1_badge"]),
        ("2. Preprocessing & Splits", PALETTE["c2_bg"], PALETTE["c2_border"], PALETTE["c2_badge"]),
        ("3. Unified Modeling (6 Models)", PALETTE["c3_bg"], PALETTE["c3_border"], PALETTE["c3_badge"]),
        ("4. Empirical Diagnostics & Slices", PALETTE["c4_bg"], PALETTE["c4_border"], PALETTE["c4_badge"]),
    ]

    for i, (col_title, bg, border, badge_bg) in enumerate(col_meta):
        cx = cols_x[i]
        # Main Column Container
        draw_card(cx, base_y, col_w, card_h, bg, border, lw=1.8, radius=0.22, zorder=1)
        # Column Title Header
        draw_card(cx, base_y + card_h - 0.52, col_w, 0.52, badge_bg, border, lw=1.0, radius=0.20, zorder=3)
        ax.text(cx + col_w/2, base_y + card_h - 0.26, col_title, ha="center", va="center",
                color="#ffffff", fontsize=11.5, fontweight="bold", zorder=4)

    # =============================================================
    # Column 1: Auxiliary Resource Mining
    # =============================================================
    c1_x = cols_x[0] + 0.16
    c1_w = col_w - 0.32

    # 1A: BanglaGEC
    draw_card(c1_x, 6.25, c1_w, 2.05, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w/2, 8.05, "BanglaGEC Corpus (nahidhub)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c1_x + c1_w/2, 7.82, "~7.07M parallel sentence pairs", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.5, fontstyle="italic")
    ax.text(c1_x + c1_w/2, 7.30, "⚙️ [Stage B] Grammar Rule Mining\n• 200,000 deterministic head sample\n• Positional substitution mapping\n• Minimum support threshold ≥ 5",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)
    draw_card(c1_x + 0.15, 6.35, c1_w - 0.3, 0.42, "#ffffff", "#0284c7", lw=1.0)
    ax.text(c1_x + c1_w/2, 6.56, "grammar_rules.csv (2,238 rules)", ha="center", va="center",
            color="#0369a1", fontsize=9.0, fontweight="bold")

    # 1B: Bangla NLP Words
    draw_card(c1_x, 3.65, c1_w, 2.35, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w/2, 5.75, "Bangla NLP Words (saifsunny)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c1_x + c1_w/2, 5.52, "Grammatical particles & stopwords", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.5, fontstyle="italic")
    ax.text(c1_x + c1_w/2, 5.00, "⚙️ [Stage C] Lexicon Construction\n• Stopwords extraction (813 words)\n• Negator detection (14 tokens)\n• Negators strictly preserved from deletion",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)
    draw_card(c1_x + 0.15, 3.75, c1_w - 0.3, 0.65, "#ffffff", "#0284c7", lw=1.0)
    ax.text(c1_x + c1_w/2, 4.20, "negation_lexicon.txt (14 negators)", ha="center", va="center",
            color="#0369a1", fontsize=8.5, fontweight="bold")
    ax.text(c1_x + c1_w/2, 3.92, "bangla_stopwords.txt (813 words)", ha="center", va="center",
            color="#0369a1", fontsize=8.5, fontweight="bold")

    # 1C: Bangla NLP Catalog
    draw_card(c1_x, 1.10, c1_w, 2.30, PALETTE["c1_box"], PALETTE["c1_border"], lw=1.2)
    ax.text(c1_x + c1_w/2, 3.15, "Bangla NLP Catalog (kishormorol)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c1_x + c1_w/2, 2.92, "813 papers, 63 datasets, 26 tasks", ha="center", va="center",
            color=PALETTE["muted"], fontsize=8.5, fontstyle="italic")
    ax.text(c1_x + c1_w/2, 2.40, "⚙️ Literature Review Generation\n• Filter: sentiment | opinion | emotion\n• Year-ranked metadata synthesis\n• Auditable prior work selection",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)
    draw_card(c1_x + 0.15, 1.20, c1_w - 0.3, 0.65, "#ffffff", "#0284c7", lw=1.0)
    ax.text(c1_x + c1_w/2, 1.63, "literature_review_table.csv (166 items)", ha="center", va="center",
            color="#0369a1", fontsize=8.5, fontweight="bold")
    ax.text(c1_x + c1_w/2, 1.35, "prior_work.md (Markdown synthesis)", ha="center", va="center",
            color="#0369a1", fontsize=8.5, fontweight="bold")

    # =============================================================
    # Column 2: Preprocessing & Splits
    # =============================================================
    c2_x = cols_x[1] + 0.16
    c2_w = col_w - 0.32

    # 2A: Supervised Sources
    draw_card(c2_x, 6.95, c2_w, 1.35, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_x + c2_w/2, 8.05, "Raw Supervised Corpora", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 7.50, "• Daraz BD (e-commerce customer ratings)\n• RevBangla (curated product polarities)\n• LargeSentiment (social media text)\nCandidate Pool: 113,959 total rows",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)

    # 2B: Normalisation & Negation Scope
    draw_card(c2_x, 4.95, c2_w, 1.75, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_x + c2_w/2, 6.45, "[Stage D & E] Normalisation Pipeline", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 5.70, "• Unicode NFC + bnunicodenormalizer\n• URL / User mention tokenization\n• Bangla digits → ASCII; char collapse\n• Negation tagging: 3-token forward NEG_\n• Substitution: grammar_rules.csv",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)

    # 2C: Harmonisation & Filtering
    draw_card(c2_x, 3.10, c2_w, 1.60, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_x + c2_w/2, 4.45, "[Stage F & G] Harmonisation & Filter", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 3.75, "• Canonical 3-class: {neg, neu, pos}\n• Rating map: 1-2→neg, 3→neu, 4-5→pos\n• Length 3–1000 chars; Bangla ratio ≥ 0.4\n• Exact-match deduplication",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.5)

    # 2D: Joint Stratified Splitting
    draw_card(c2_x, 1.10, c2_w, 1.75, PALETTE["c2_box"], PALETTE["c2_border"], lw=1.2)
    ax.text(c2_x + c2_w/2, 2.60, "[Stage H] Stratified Splitting", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 2.25, "Stratified on (label, source) • 60% majority cap",
            ha="center", va="center", color=PALETTE["muted"], fontsize=8.0)
    draw_card(c2_x + 0.12, 1.20, c2_w - 0.24, 0.78, "#ffffff", "#6366f1", lw=1.0)
    ax.text(c2_x + c2_w/2, 1.75, "train_clean.csv : 5,797 rows (70%)", ha="center", va="center",
            color="#4338ca", fontsize=8.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 1.51, "val_clean.csv   : 1,388 rows (15%)", ha="center", va="center",
            color="#4338ca", fontsize=8.5, fontweight="bold")
    ax.text(c2_x + c2_w/2, 1.28, "test_clean.csv  : 1,388 rows (15%)", ha="center", va="center",
            color="#b91c1c", fontsize=8.5, fontweight="bold")

    # =============================================================
    # Column 3: Unified Modeling Framework
    # =============================================================
    c3_x = cols_x[2] + 0.16
    c3_w = col_w - 0.32

    # 3A: Training Recipe
    draw_card(c3_x, 6.30, c3_w, 2.00, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_x + c3_w/2, 8.05, "Unified Training Protocol", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c3_x + c3_w/2, 7.20, "• AdamW with 2 discriminative LR groups\n• Backbone LR: 2e-5 | Head LR: 6e-5\n• Batch: 16 | Epochs: 4 | Warmup: 10%\n• Max Seq Len: 128 | Label Smoothing: 0.05\n• Inverse-frequency class weights (mean=1)\n• Early stopping: 2 epochs on Macro-F1",
            ha="center", va="center", color=PALETTE["dark"], fontsize=8.3)

    # 3B: Bangla-Specific Models
    draw_card(c3_x, 4.30, c3_w, 1.75, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_x + c3_w/2, 5.82, "Language-Specific Backbones", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.0, fontweight="bold")
    
    draw_card(c3_x + 0.1, 5.10, c3_w - 0.2, 0.45, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_x + c3_w/2, 5.32, "BanglaBERT (csebuetnlp/banglabert)\nELECTRA Discriminator • 27.5 GB Bangla corpus",
            ha="center", va="center", color="#065f46", fontsize=7.5, fontweight="bold")
    
    draw_card(c3_x + 0.1, 4.40, c3_w - 0.2, 0.45, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_x + c3_w/2, 4.62, "BanglishBERT (csebuetnlp/banglishbert)\nCode-Mixed Pretrained ELECTRA",
            ha="center", va="center", color="#065f46", fontsize=7.5, fontweight="bold")

    # 3C: Multilingual & BERT-Base
    draw_card(c3_x, 2.50, c3_w, 1.55, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_x + c3_w/2, 3.82, "BERT-Base & Multilingual Baselines", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.0, fontweight="bold")
    draw_card(c3_x + 0.1, 3.10, c3_w - 0.2, 0.45, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_x + c3_w/2, 3.32, "Bangla BERT Base (sagorsarker/bangla-bert-base)", ha="center", va="center",
            color="#065f46", fontsize=7.6, fontweight="bold")
    draw_card(c3_x + 0.1, 2.60, c3_w - 0.2, 0.45, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_x + c3_w/2, 2.82, "mBERT (~178M) & XLM-RoBERTa (~270M)", ha="center", va="center",
            color="#065f46", fontsize=7.6, fontweight="bold")

    # 3D: Classical Baseline
    draw_card(c3_x, 1.10, c3_w, 1.15, PALETTE["c3_box"], PALETTE["c3_border"], lw=1.2)
    ax.text(c3_x + c3_w/2, 2.05, "Classical N-Gram Reference", ha="center", va="center",
            color=PALETTE["dark"], fontsize=9.8, fontweight="bold")
    draw_card(c3_x + 0.1, 1.20, c3_w - 0.2, 0.65, "#ffffff", "#10b981", lw=1.0)
    ax.text(c3_x + c3_w/2, 1.60, "TF-IDF + Logistic Regression", ha="center", va="center",
            color="#065f46", fontsize=8.5, fontweight="bold")
    ax.text(c3_x + c3_w/2, 1.35, "Word (60k) + Char (80k) = 140k n-gram features", ha="center", va="center",
            color=PALETTE["muted"], fontsize=7.5)

    # =============================================================
    # Column 4: Empirical Diagnostics & Slices
    # =============================================================
    c4_x = cols_x[3] + 0.16
    c4_w = col_w - 0.32

    # 4A: Leaderboard
    draw_card(c4_x, 6.20, c4_w, 2.10, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_x + c4_w/2, 8.05, "Test Split Leaderboard (1,388)", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    leader_rows = [
        ("TF-IDF Baseline", "0.6341", "88.54%"),
        ("BanglishBERT", "0.6065", "82.78%"),
        ("BanglaBERT", "0.5985", "83.00%"),
        ("mBERT", "0.5843", "79.32%"),
        ("Bangla BERT Base", "0.5806", "77.31%"),
        ("XLM-RoBERTa", "0.5760", "79.68%"),
    ]
    ax.text(c4_x + 0.20, 7.68, "Model", fontweight="bold", fontsize=8.2, color=PALETTE["dark"])
    ax.text(c4_x + 1.95, 7.68, "Macro-F1", fontweight="bold", fontsize=8.2, color=PALETTE["dark"])
    ax.text(c4_x + 2.80, 7.68, "Accuracy", fontweight="bold", fontsize=8.2, color=PALETTE["dark"])
    for idx, (m_name, f1, acc) in enumerate(leader_rows):
        ry = 7.42 - idx * 0.21
        ax.text(c4_x + 0.20, ry, m_name, fontsize=8.0, color=PALETTE["dark"])
        ax.text(c4_x + 2.05, ry, f1, fontsize=8.0, fontweight="bold", color="#b45309")
        ax.text(c4_x + 2.85, ry, acc, fontsize=8.0, color=PALETTE["muted"])

    # 4B: Robustness Slices
    draw_card(c4_x, 4.35, c4_w, 1.65, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_x + c4_w/2, 5.78, "3 Robustness Slices", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c4_x + c4_w/2, 5.00, "• Source: RevBangla (91.0%) vs Daraz (73.7%)\n  (17.26% spread exposes noisy star ratings)\n• Negation: Non-neg (85.7%) vs Negated (65.6%)\n  (20.10% deficit reveals post-verbal syntax)\n• Length: 1-5 tokens (88.3%) down to 41+ (61.1%)",
            ha="center", va="center", color=PALETTE["dark"], fontsize=7.8)

    # 4C: Ablation
    draw_card(c4_x, 2.70, c4_w, 1.45, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_x + c4_w/2, 3.90, "BanglaGEC Rule Ablation", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    ax.text(c4_x + c4_w/2, 3.28, "• With Rules: 0.5985 F1 (83.00% Acc)\n• Without Rules: 0.6228 F1 (86.34% Acc)\n• Net Delta: −2.43 pp Macro-F1 (−3.34 pp Acc)\nFinding: 51.5% lexical substitution rules hurt\nby regularizing informal polarity keywords.",
            ha="center", va="center", color=PALETTE["dark"], fontsize=7.8)

    # 4D: Diagnostic Figures & Artefacts
    draw_card(c4_x, 1.10, c4_w, 1.40, PALETTE["c4_box"], PALETTE["c4_border"], lw=1.2)
    ax.text(c4_x + c4_w/2, 2.28, "13 Visual Diagnostic Plots", ha="center", va="center",
            color=PALETTE["dark"], fontsize=10.5, fontweight="bold")
    draw_card(c4_x + 0.1, 1.20, c4_w - 0.2, 0.85, "#ffffff", "#f59e0b", lw=1.0)
    ax.text(c4_x + c4_w/2, 1.70, "• 5 Confusion Matrices + 5 ROC Curves\n• Macro-F1 & Accuracy Comparison Bars\n• Stratified Label Distribution Plot\n• 25 Hand-Audited Error Cases",
            ha="center", va="center", color="#92400e", fontsize=7.8, fontweight="bold")

    # =============================================================
    # Clean Inter-Column Pipeline Arrows
    # =============================================================
    # 1. Grammar Rules -> Preprocessing (clean horizontal)
    arr1 = patches.FancyArrowPatch((c1_x + c1_w, 6.56), (c2_x, 6.56),
                                  arrowstyle="->,head_width=4,head_length=6",
                                  lw=2.0, color="#0284c7")
    ax.add_patch(arr1)
    ax.text((c1_x + c1_w + c2_x)/2, 6.72, "Mined Rules\n(2,238)", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#0369a1")

    # 2. Negation Lexicon -> Preprocessing (clean horizontal)
    arr2 = patches.FancyArrowPatch((c1_x + c1_w, 4.20), (c2_x, 5.20),
                                  arrowstyle="->,head_width=4,head_length=6",
                                  connectionstyle="arc3,rad=-0.15",
                                  lw=2.0, color="#0284c7")
    ax.add_patch(arr2)
    ax.text((c1_x + c1_w + c2_x)/2, 4.45, "Lexicon\n(14 neg)", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#0369a1")

    # 3. Training & Val Splits -> Unified Modeling (Horizontal connector via bridge)
    arr3 = patches.FancyArrowPatch((c2_x + c2_w, 1.75), (c3_x, 7.30),
                                  arrowstyle="->,head_width=4,head_length=6",
                                  connectionstyle="arc3,rad=0.22",
                                  lw=2.2, color="#4f46e5")
    ax.add_patch(arr3)
    ax.text((c2_x + c2_w + c3_x)/2 - 0.05, 4.60, "Train & Val Splits\n(7,185 rows)",
            ha="center", va="center", fontsize=8.0, fontweight="bold", color="#4338ca")

    # 4. Trained Checkpoints -> Evaluation Leaderboard (clean horizontal)
    arr4 = patches.FancyArrowPatch((c3_x + c3_w, 7.25), (c4_x, 7.25),
                                  arrowstyle="->,head_width=4,head_length=6",
                                  lw=2.2, color="#059669")
    ax.add_patch(arr4)
    ax.text((c3_x + c3_w + c4_x)/2, 7.45, "Evaluated\nCheckpoints", ha="center", va="bottom",
            fontsize=8.0, fontweight="bold", color="#047857")

    # 5. Held-Out Test Set -> Dedicated Bottom Evaluation Bus
    # Draw a prominent, clean bottom pipeline ribbon
    test_bus_y = 0.45
    draw_card(c2_x + c2_w - 0.2, test_bus_y - 0.15, (c4_x - c2_x - c2_w) + 0.4, 0.45,
              "#fee2e2", "#ef4444", lw=1.2, radius=0.12)
    ax.text((c2_x + c2_w + c4_x)/2, test_bus_y + 0.07,
            "Held-Out Test Split (1,388 samples) — Zero Cross-Platform Leakage",
            ha="center", va="center", color="#991b1b", fontsize=8.5, fontweight="bold")

    # Arrow from test_clean down to the bus
    arr_test_down = patches.FancyArrowPatch((c2_x + c2_w/2, 1.10), ((c2_x + c2_w/2), test_bus_y + 0.30),
                                           arrowstyle="->,head_width=3.5,head_length=5",
                                           connectionstyle="arc3,rad=0.0",
                                           lw=1.8, color="#ef4444", linestyle="solid")
    ax.add_patch(arr_test_down)

    # Arrow from the bus up to Column 4 Evaluation
    arr_test_up = patches.FancyArrowPatch((c4_x + c4_w/2, test_bus_y + 0.30), (c4_x + c4_w/2, 1.10),
                                         arrowstyle="<-,head_width=3.5,head_length=5",
                                         connectionstyle="arc3,rad=0.0",
                                         lw=1.8, color="#ef4444", linestyle="solid")
    ax.add_patch(arr_test_up)

    plt.tight_layout()
    for p in output_paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close()

if __name__ == "__main__":
    targets = [
        "results/system_overview.png",
        "docs/assets/images/system_overview.png",
    ]
    draw_system_overview(targets)
