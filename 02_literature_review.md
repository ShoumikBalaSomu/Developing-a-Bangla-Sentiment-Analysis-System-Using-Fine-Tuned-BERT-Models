# 2. Literature Review

---

## 2.1 How This Review Was Constructed

Most student literature reviews are assembled by hand, which makes them unfalsifiable: there is no record of what was searched or what was excluded, so the selection of prior work cannot be checked for cherry-picking.

This review is generated from a machine-readable catalog. The `kishormorol/bangla-nlp-catalog` dataset indexes **813 papers, 63 datasets, 20 models and 9 tools across 26 tasks**, each tagged by task. `src/catalog_to_lr.py` filters it with the regex `sentiment|opinion|emotion|affect|review`, sorts by year, groups by resource type, and emits `docs/generated/prior_work.md`.

To regenerate with a different scope:

```bash
python src/catalog_to_lr.py                          # sentiment/opinion/emotion/review
python src/catalog_to_lr.py --keyword "hate|offensive" --top 50
```

The filtered table is written to `dataset/preprocessed/literature_review_table.csv`, and the generated Markdown is included from `docs/generated/prior_work.md`. **Regenerate that file before submission** so the review reflects the catalog version actually used.

The narrative sections below summarise the *verified* literature. Where a specific number is quoted, its source is named.

---

## 2.2 Transformer Language Models for Bangla

### BanglaBERT

Bhattacharjee et al. introduced BanglaBERT, a BERT-based model pretrained on **Bangla2B+**, a 27.5 GB Bangla corpus crawled from eleven sources. The paper reports that BanglaBERT outperforms multilingual baselines on Bangla NLU benchmarks while being smaller, and established the BanglaGLUE benchmark suite. The released discriminator checkpoint is `csebuetnlp/banglabert` (ELECTRA architecture).

This is the strongest argument for language-specific pretraining in Bangla: a model that has only ever seen Bangla learns Bangla morphology — the agglutinative suffixes, the compound verb forms — more efficiently than a model that must spread capacity across a hundred languages.

### Bangla BERT Base

`sagorsarker/bangla-bert-base` follows the `bert-base-uncased` architecture: **12 layers, 768 hidden units, 12 attention heads, ~110M parameters**, trained for one million steps on Bangla text. It provides a BERT-family (rather than ELECTRA-family) comparison point against BanglaBERT, which matters because ELECTRA's replaced-token-detection objective and BERT's masked-language-modelling objective produce differently-behaved representations for the same downstream task.

### BanglishBERT

`csebuetnlp/banglishbert` is pretrained for Bangla–English code-mixed text. Code-mixing is pervasive in Bangladeshi social media and e-commerce reviews — a single sentence may switch scripts mid-clause. Including this model tests whether code-mixed pretraining helps when the evaluation data contains romanised Bangla.

### Multilingual baselines

`bert-base-multilingual-cased` (mBERT) and `xlm-roberta-base` (XLM-R) are the standard cross-lingual references. They require no Bangla-specific pretraining and often perform surprisingly well, which is exactly why they must be included: **a language-specific model that cannot beat mBERT is not worth its engineering cost.** Any claim that BanglaBERT is "better" is meaningless without this comparison.

### Related findings in the literature

Kowsher et al.'s Bangla-BERT work reports state-of-the-art results across several Bangla datasets and substantial improvement on binary linguistic tasks. A study of Bangla hate-speech classification found that among transformer models, **BanglaBERT produced the best performance for both subtasks and outperformed both mBERT and XLM-RoBERTa despite being smaller**, which the authors attribute to language-specific pretraining. Chakma and Hasan leveraged BanglaBERT specifically for *low-resource* Bangla sentiment analysis, the setting closest to this project.

> ⚠️ **Citation caution.** The specific accuracy figures reported by these papers were obtained on *their* datasets with *their* preprocessing and splits. They are not comparable to this project's numbers and are not used as a performance target anywhere in `docs/04_analysis.md`.

---

## 2.3 Bangla Sentiment Analysis Resources

### Labelled corpora

| Resource | Scale | Classes | Notes |
|---|---|---|---|
| **RevBangla** (`saifsunny/revbangla`) | Bangla e-commerce product comments | positive / negative | `Label_Sentiment` column. Also published on Mendeley Data. |
| **Large Sentiment Analysis Bangla** (`tahmidmir/largesentimentdata`) | large-scale | positive / negative / neutral | Four columns: Platform, Comment, Sentiment, Label. Sourced from Twitter, YouTube and Google. |
| **Bangladesh Daraz Reviews** (`robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis`) | Daraz Bangladesh product reviews | star rating → mapped | Requires rating→sentiment mapping. |
| **Daraz/Pickaboo corpus** (Rashid et al., 2024, *Data in Brief*) | 78,130 reviews across 152 product categories | sentiment + emotion | Includes Bangla and English reviews; one of the largest published Bangladeshi e-commerce corpora. |
| **BanglishRev** (Shamael et al., 2024, arXiv:2412.13161) | 1.74M written reviews from 3.2M ratings, 128k products | sentiment | Largest Bangla-English code-mixed review dataset reported to date. |
| **BANEmo** (IEEE, 2026) | 15,000 comments | labelled for sentiment | Harvested from social media and newspapers. |
| **BMotion** (`eftekhar/bangla-textual-emotion-dataset`) | ~30k texts | 6 emotion classes | Emotion rather than polarity; collected by scraping and translation. |

### Supporting linguistic resources

| Resource | Content | Used in this project for |
|---|---|---|
| **BanglaGEC** (`nahidhub/banglagec`) | ~7.07M parallel sentence pairs; train 5,730,275 / validation 636,702 / test 707,448 rows | Mining grammar-normalisation rules |
| **Bangla NLP words** (`saifsunny/bangla-nlp-words`) | Bangla grammatical elements: stopwords, particles, punctuation | Negation lexicon + stopwords |
| **Bangla NLP Catalog** (`kishormorol/bangla-nlp-catalog`) | 813 papers, 63 datasets, 20 models, 9 tools, 26 tasks | Literature-review generation |
| **BanglaSenti** (GitHub, `fahad35`) | 61,582 Bangla words labelled positive / negative / neutral | Not used; a lexicon-based alternative documented for contrast in §2.5 |

BanglaGEC is described by its authors as a large-scale parallel corpus for Bangla Grammatical Error Correction. Subsequent work has used LLMs to improve Bangla GEC by categorising error types and generating synthetic data (Bhattacharyya & Bhattacharya, arXiv:2406.14284), and transformer-based token classification for Bangla grammatical error *detection* (arXiv:2411.08344). None of this literature treats BanglaGEC as a sentiment resource, and this project does not either.

---

## 2.4 Classical and Hybrid Approaches

Before transformers, Bangla sentiment analysis relied on:

- **Lexicon-based methods.** Sentiment score computed by summing polarities of matched words. The BanglaSenti resource (61,582 words) supports this directly. Fast, interpretable, language-specific — but brittle: no compositional understanding, so negation, intensifiers and sarcasm are systematically mishandled.
- **Machine learning on hand-crafted features.** TF-IDF or bag-of-words plus SVM / Logistic Regression / Naive Bayes. Rejuan et al. applied machine-learning and deep-learning approaches to Bangla online-shopping customer sentiment. Performance is bounded by feature engineering and degrades on unseen vocabulary.
- **Stacked contractive auto-encoders** on machine-translated Bangla reviews, reported in *Journal of Ambient Intelligence and Humanized Computing* (2023). The authors note that prior opinion-mining methods in Bangla "reported limited performance."
- **Ensemble deep learning for aspect-sentiment-opinion extraction.** BanglaASTE (arXiv:2511.21381) targets the finer-grained triplet-extraction task rather than document-level polarity.

This project retains a **TF-IDF (word + character n-gram) + Logistic Regression** baseline deliberately. Character n-grams matter for Bangla: they capture morphological suffixes that word-level features miss, and they are robust to the spelling variation that dominates social-media text. The baseline exists so the transformer results can be *interpreted* — a transformer that fails to beat it clearly is not earning its compute cost.

---

## 2.5 Research Gap

Four gaps motivate this project:

**G1 — Negation is acknowledged but rarely handled explicitly.** Lexicon methods fail on negation by construction. Transformer methods usually absorb it implicitly from pretraining, but no Bangla sentiment study the catalog surfaces reports a dedicated negation-subset evaluation. This project tags negation scope and measures accuracy on a negated subset separately (`docs/04_analysis.md` §4.2), so the effect is quantified rather than assumed.

**G2 — Results are not comparable across studies.** Different datasets, preprocessing, label schemes, splits and metrics mean published accuracies cannot be ranked against each other. This project fixes one recipe across all models and reports macro-F1 and MCC rather than accuracy alone.

**G3 — Normalisation is typically hand-written.** Bangla social text is orthographically noisy, and most pipelines apply a small hand-authored rule set. This project mines the rule table from BanglaGEC at support ≥ 5, making it data-derived, extensible, and reproducible from a recorded sample size.

**G4 — Source leakage in evaluation.** When multiple platforms are pooled, a random split can place one platform entirely in training or entirely in test. The reported score then measures domain shift, not sentiment skill. This project stratifies on `(label, source)` jointly and reports per-source accuracy so leakage would be visible.

---

## 2.6 Positioning of This Work

| Aspect | Common practice | This project |
|---|---|---|
| Preprocessing | Hand-written rules | Rules mined from BanglaGEC, support ≥ 5 |
| Negation | Implicit / ignored | Explicit scope tagging + dedicated slice evaluation |
| Label scheme | Per-dataset | Harmonised to `{negative, neutral, positive}` with documented mapping |
| Splitting | Random | Stratified on `(label, source)` |
| Metrics | Accuracy | Macro-F1, balanced accuracy, MCC, per-class P/R/F1, one-vs-rest AUC |
| Baselines | Sometimes omitted | TF-IDF (word + char) + Logistic Regression always included |
| Model comparison | Single model | 5 transformers under one identical recipe |
| Literature review | Hand-picked | Generated from a machine-readable catalog, reproducible |

The central empirical question is **G2's**: does language-specific Bangla pretraining (BanglaBERT, bangla-bert-base) beat multilingual pretraining (mBERT, XLM-R) on harmonised Bangla e-commerce and social-media sentiment, under an identical fine-tuning recipe? Section 3 of `docs/04_analysis.md` answers it.

---

## 2.7 References

Full BibTeX entries are in [`references.bib`](references.bib). Key sources:

1. Bhattacharjee, A., Samin, K. I., Islam, M. S., Rahman, A., Rahman, M. S. (2021). *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.* arXiv:2101.00204.
2. Kowsher, M. et al. (2022). *Bangla-BERT: Transformer-Based Efficient Model for Sentiment Analysis and Language Understanding.* IEEE.
3. Chakma, A., Hasan, M. (2023). *LowResource at BLP-2023 Task 2: Leveraging BanglaBert for Low Resource Sentiment Analysis of Bangla Language.* CoRR abs/2311.12735.
4. Bhattacharyya, P., Bhattacharya, A. (2024). *Leveraging LLMs for Bangla Grammar Error Correction: Error Categorization, Synthetic Data, and Model Evaluation.* arXiv:2406.14284.
5. Shamael, M. N. et al. (2024). *BanglishRev: A Large-Scale Bangla-English and Code-Mixed Product Review Dataset for Sentiment Analysis.* arXiv:2412.13161.
6. Rashid, M. R. A. et al. (2024). *A comprehensive dataset for sentiment and emotion classification from Bangladesh e-commerce reviews.* Data in Brief.
7. Alam, F. et al. (2021). *A Review of Bangla Natural Language Processing Tasks and the Utility of Transformer Models.* arXiv:2107.03844.
8. Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
9. Clark, K., Luong, M.-T., Le, Q. V., Manning, C. D. (2020). *ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.* ICLR.
10. Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL. (XLM-RoBERTa)

> **Auto-generated prior-work tables:** see [`generated/prior_work.md`](generated/prior_work.md), produced by `src/catalog_to_lr.py`. That file is regenerated from the catalog and should be re-run before submission.
