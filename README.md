# LLM Source Attribution & Stylometry Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Executive Summary
An end-to-end supervised Natural Language Processing (NLP) and Machine Learning pipeline designed for **AI text provenance and stylometric attribution**. Given a synthetic text sample, the model analyzes lexical patterns, sublinear n-gram distributions, and structural formatting habits to identify which Large Language Model (LLM) variant generated the response.

---

## Pipeline Architecture

```mermaid
flowchart LR
    A["Raw Corpus (9,719 Samples)"] --> B["Stratified Split (80% / 20%)"]
    B --> C["Sublinear TF-IDF (15,000 Features)"]
    C --> D["Logistic Regression (Balanced)"]
    D --> E["Evaluation (33.49% Accuracy)"]
```

---

## Visual Results

| Class Distribution | Confusion Matrix |
| :---: | :---: |
| <img src="class_distribution.png" width="420"/> | <img src="confusion_matrix.png" width="420"/> |

---

## Methodology & Implementation

| Pipeline Stage | Implementation Details |
| :--- | :--- |
| **Dataset** | 9,719 labeled synthetic responses (`ai_responses.parquet`) paired with prompt metadata (`ai_prompts.jsonl`). |
| **Validation Strategy** | Stratified 80/20 train-test split (`random_state=42`) to preserve class proportions across all target LLM variants. |
| **Feature Extraction** | Sublinear TF-IDF vectorization capturing top `15,000` unigram and bigram lexical signatures. |
| **Classification Model** | L2-regularized Logistic Regression with balanced class weights (`class_weight="balanced"`). |
| **Evaluation Metrics** | Accuracy, Precision, Recall, Macro F1-Score, and Multi-Class Confusion Matrix. |

---

## Experimental Results
* **Test Accuracy:** **33.49%** across multi-class LLM variants (outperforming the random-chance baseline on fine-grained same-family model variants).
* **Key Insight:** Stylometric n-gram features effectively separate distinct model generations, whereas closely related model checkpoints exhibit overlapping lexical distributions—visualized in the confusion matrix above.

---

## Repository Structure

```text
llm-source-attribution/
├── llm_attribution_pipeline.ipynb   # Exploratory analysis, TF-IDF vectorization, training & confusion matrix
├── ai_responses.parquet             # Primary labeled text corpus (9,719 samples)
├── ai_prompts.jsonl                 # Associated prompt registry metadata
├── class_distribution.png           # Dataset class balance chart
├── confusion_matrix.png             # Model classification performance heatmap
├── requirements.txt                 # Reproducible Python environment dependencies
├── LICENSE                          # MIT open-source license
└── README.md                        # Project documentation and experimental findings
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/faizannadeem4449-dotcom/llm-source-attribution.git](https://github.com/faizannadeem4449-dotcom/llm-source-attribution.git)
cd llm-source-attribution

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook llm_attribution_pipeline.ipynb
```
