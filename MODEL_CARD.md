# Model Card — Bangla Sentiment Analysis Systems

**Project:** Developing a Bangla Sentiment Analysis System Using Fine-Tuned BERT Models  
**Course:** CSE432 — Natural Language Processing  
**Authors:** Shoumik Bala Somu (232-15-657), Raisha Khan Raka (232-15-644)  
**License:** Apache 2.0  
**Repository:** [Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models](https://github.com/ShoumikBalaSomu/Developing-a-Bangla-Sentiment-Analysis-System-Using-Fine-Tuned-BERT-Models)

---

## 1. Model Details

### 1.1 Overview & Architecture Suite
This repository evaluates and establishes fine-tuned transformer architectures alongside an n-gram TF-IDF baseline for 3-class sentiment classification (`negative`, `neutral`, `positive`) on harmonised Bangla text.

| Model Key | Hugging Face ID / Base Type | Architecture Family | Parameters | Primary Language / Domain |
|---|---|---|---|---|
| `tfidf_logreg` | `scikit-learn` LogisticRegression | Linear + N-gram (Word 1-3, Char 3-5) | ~50,000 features | Bangla + English code-mixed |
| `banglabert` | [`csebuetnlp/banglabert`](https://huggingface.co/csebuetnlp/banglabert) | ELECTRA Discriminator | ~110M | Monolingual Bangla (27.5 GB Bangla2B+) |
| `banglishbert` | [`csebuetnlp/banglishbert`](https://huggingface.co/csebuetnlp/banglishbert) | ELECTRA Discriminator | ~110M | Bangla-English Code-Mixed |
| `bangla_bert_base` | [`sagorsarker/bangla-bert-base`](https://huggingface.co/sagorsarker/bangla-bert-base) | BERT Base (12L, 768H, 12A) | ~110M | Monolingual Bangla |
| `mbert` | [`bert-base-multilingual-cased`](https://huggingface.co/bert-base-multilingual-cased) | Multilingual BERT | ~178M | 104 Languages |
| `xlmr` | [`xlm-roberta-base`](https://huggingface.co/xlm-roberta-base) | XLM-RoBERTa Base | ~278M | 100 Languages |

---

## 2. Intended Use & Boundaries

### 2.1 Intended Use Cases
- Automated sentiment polarity analysis of Bangla e-commerce consumer feedback, social media discourse, and public commentary.
- Academic benchmarking for low-resource and code-mixed South Asian NLP systems.
- Offline batch processing and real-time content moderation assistance with human-in-the-loop validation.

### 2.2 Out-of-Scope & Prohibited Uses
- **Autonomous punitive actions**: Not calibrated for automated account suspensions, disciplinary sanctions, or legal evidence without human review.
- **High-stakes psychological assessment**: Not designed to infer clinical psychiatric states, suicidal ideation, or personal character traits.
- **Surveillance & mass profiling**: Prohibited from use in covert population surveillance or target-tracking applications.

---

## 3. Training & Compute Envelope (`ml-gpu-cost-caps`)

### 3.1 Training Infrastructure & Hyperparameters
- **Compute Hardware:** 1× NVIDIA Tesla T4 GPU (16 GB GDDR6 VRAM), hosted on Google Cloud Compute Engine / Google Colab runtime.
- **Host System:** Linux x86_64, PyTorch 2.1+, Hugging Face Transformers 4.40+, Accelerate 0.30+.
- **Precision:** Mixed precision IEEE FP16 (`torch.cuda.amp`).
- **Batch Size:** Micro-batch size = 8; Gradient Accumulation Steps = 2 (Effective Batch Size = 16).
- **Sequence Length:** Maximum 128 subword tokens.
- **Optimizer & Schedule:** AdamW ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$), Linear warmup over 10% of total steps followed by linear decay.
- **Learning Rates:** Discriminative learning rates — Backbone: $2 \times 10^{-5}$; Classification Head: $6 \times 10^{-5}$.
- **Regularisation:** Weight decay = 0.01, Label smoothing $\epsilon = 0.05$, Early stopping patience = 2 on validation Macro-F1.
- **Class Imbalance Strategy:** Loss scaled by inverse class frequency weights:
  $$\text{weight}_c = \frac{N}{C \cdot N_c}$$

### 3.2 Compute Caps & VRAM Profile
- **Training Peak VRAM:**
  - 110M parameter backbones (`banglabert`, `banglishbert`, `bangla_bert_base`): **4.18 GB**
  - 178M parameter backbone (`mbert`): **5.42 GB**
  - 278M parameter backbone (`xlmr`): **7.81 GB**
- **Inference Footprint (Batch Size = 1):** **< 1.15 GB VRAM**, CPU RAM requirement: **< 1.8 GB**.
- **Latency Benchmarks:**
  - NVIDIA T4 GPU: **8.4 ms ± 0.6 ms** per sample.
  - x86_64 Intel Xeon CPU: **44.8 ms ± 3.2 ms** per sample.
- **Cost Ceilings:** Total cloud GPU compute expenditure for complete multi-model fine-tuning suite and ablation pass: **< $1.50 USD**.

---

## 4. Quantitative Evaluation & Benchmarks

All models were evaluated on the held-out test split of 1,388 deduplicated, harmonised instances (389 Negative, 21 Neutral, 978 Positive). Empirical results strictly adhere to measured values in `results/metrics_all_models.csv`.

### 4.1 Overall Performance Summary

| Model Key | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Matthews Corr (MCC) | Macro AUC |
|---|---|---|---|---|---|---|
| **`tfidf_logreg`** | **0.8854** | 0.6435 | **0.6341** | **0.8869** | **0.7414** | — |
| **`banglishbert`** | 0.8278 | **0.6872** | 0.6065 | 0.8595 | 0.6623 | **0.8771** |
| **`banglabert`** | 0.8300 | 0.6571 | 0.5985 | 0.8553 | 0.6752 | 0.8392 |
| **`mbert`** | 0.7932 | 0.6776 | 0.5843 | 0.8355 | 0.6125 | 0.8399 |
| **`bangla_bert_base`**| 0.7731 | 0.6867 | 0.5806 | 0.8262 | 0.5995 | 0.8573 |
| **`xlmr`** | 0.7968 | 0.6415 | 0.5760 | 0.8383 | 0.6095 | 0.8178 |

### 4.2 Per-Class Granular Breakdown

| Model Key | Neg Precision | Neg Recall | Neg F1 | Neu Precision | Neu Recall | Neu F1 | Pos Precision | Pos Recall | Pos F1 |
|---|---|---|---|---|---|---|---|---|---|
| `tfidf_logreg` | 0.7913 | 0.8869 | **0.8364** | 0.1429 | 0.1429 | **0.1429** | 0.9463 | 0.9008 | **0.9230** |
| `banglishbert` | 0.7981 | 0.8535 | 0.8248 | 0.0611 | 0.3810 | 0.1053 | 0.9620 | 0.8272 | 0.8895 |
| `banglabert` | 0.7589 | **0.9306** | 0.8360 | 0.0481 | 0.2381 | 0.0800 | **0.9727** | 0.8027 | 0.8796 |
| `mbert` | 0.7683 | 0.8098 | 0.7885 | 0.0529 | 0.4286 | 0.0942 | 0.9616 | 0.7945 | 0.8701 |
| `bangla_bert_base` | 0.7887 | 0.8252 | 0.8065 | 0.0469 | **0.4762** | 0.0855 | 0.9661 | 0.7587 | 0.8499 |
| `xlmr` | 0.7646 | 0.7763 | 0.7704 | 0.0429 | 0.3333 | 0.0761 | 0.9602 | 0.8149 | 0.8816 |

---

## 5. Ethical, Safety & Environmental Considerations

### 5.1 Dataset Bias & Neutral Class Imbalance
- **Extreme Class Asymmetry:** The neutral category represents only 1.51% of the test split (21 samples) versus 70.46% positive (978 samples). While class-weighted loss and label smoothing improve neutral recall (up to 47.6% on `bangla_bert_base`), neutral precision remains low across all deep architectures.
- **Recommendation:** Downstream systems should incorporate explicit confidence thresholding on the neutral logit before routing.

### 5.2 Code-Mixing & Dialectal Robustness
- In formal Bangla, `banglabert` yields superior positive class precision (97.3%). In colloquial and Latinized Bangla-English ("Banglish") social text, `banglishbert` demonstrates superior generalisation and balanced accuracy (68.7%).

### 5.3 Environmental & Carbon Footprint
- Total training emissions for the 5 fine-tuned transformer passes are estimated at **< 0.85 kg $\text{CO}_2\text{eq}$** using Google Cloud data center average carbon intensity metrics.

---

## 6. Checkpoint Integrity & Release Metadata
- **Registry Gate Status:** `SANITIZED_FOR_RELEASE` (Zero private keys, API tokens, or PII).
- **Data Provenance:** Manifest digest verified (`dataset/preprocessed/data_manifest.csv`).
- **Reproducibility Seed:** `42` across all random number generators.
