# LLM Source Attribution & Stylometry Classifier

## Project Overview
An end-to-end supervised Machine Learning pipeline that identifies which AI model generated a given text response by analyzing lexical patterns, n-gram distributions, and structural formatting.

## Dataset & Pipeline
- **Corpus:** 9,719 labeled synthetic responses across multiple LLM variants (`ai_responses.parquet` & `ai_prompts.jsonl`).
- **Feature Engineering:** Sublinear TF-IDF vectorization (`15,000` unigram & bigram features).
- **Model:** Class-balanced Logistic Regression classifier evaluated on an 80/20 stratified split.
- **Test Accuracy:** 33.49%

## Repository Structure
- `llm_attribution_pipeline.ipynb`: Data loading, exploratory plots, model training, and confusion matrix evaluation.
- `ai_responses.parquet`: Labeled text corpus.
- `ai_prompts.jsonl`: Associated prompt metadata.
