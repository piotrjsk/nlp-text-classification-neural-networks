# NLP Text Classification Engine: PyTorch & DistilBERT

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)

## Project Overview

This repository features an end-to-end Natural Language Processing (NLP) pipeline for **Sentiment Analysis and Text Classification** built with **PyTorch** and **Hugging Face Transformers**. 

The project demonstrates fine-tuning a pre-trained **DistilBERT** Transformer architecture (`DistilBertForSequenceClassification`) alongside custom data processing pipelines. The workflow covers everything from raw text exploration and subword tokenization to model training, hyperparameter optimization, and comprehensive evaluation metrics.

The dataset is automatically ingested directly via Hugging Face's `datasets` library, eliminating the need for manual dataset downloads or external file hosting.

## End-to-End Pipeline Architecture

The pipeline follows a modular, reproducible workflow designed for modern NLP tasks:

1. **Automated Data Ingestion & EDA**
   * Downloads and caches the target text dataset seamlessly via Hugging Face `datasets`.
   * Analyzes class distributions, token counts, and sequence lengths.
   * Generates custom **WordCloud** visualizations and frequency distributions using `matplotlib` and `seaborn`.

2. **Text Preprocessing & Tokenization (`DistilBertTokenizerFast`)**
   * Converts raw text into token IDs, attention masks, and input tensors using fast Rust-based subword tokenizers.
   * Handles sequence truncation and dynamic padding to optimize PyTorch batch processing.

3. **PyTorch Data Pipeline & DataLoader**
   * Encapsulates tokenized sequences into a custom PyTorch `Dataset` abstraction.
   * Leverages PyTorch `DataLoader` for multi-threaded mini-batch generation during training and evaluation loops.

4. **Model Architecture & Fine-Tuning**
   * Fine-tunes `DistilBertForSequenceClassification` using AdamW (`torch.optim.AdamW`).
   * Implements custom PyTorch training loops with step-by-step progress logging (`tqdm`), loss tracking, and evaluation validation.

5. **Model Evaluation & Error Analysis**
   * Computes classification metrics using `scikit-learn` (`classification_report`, precision, recall, F1-score).
   * Generates confusion matrices to highlight class-level prediction accuracy and misclassifications.

## Pipeline Breakdown & Key Modules

| Pipeline Stage | Components / Libraries Used | Key Functionality |
| :--- | :--- | :--- |
| **Data Ingestion** | `datasets (load_dataset)` | Automated dataset streaming and caching |
| **EDA & Visuals** | `matplotlib`, `seaborn`, `wordcloud` | Corpus analysis, word clouds, length distribution |
| **Preprocessing** | `transformers (DistilBertTokenizerFast)` | Fast subword tokenization, attention masks |
| **Modeling** | `torch.nn`, `transformers (DistilBERT)` | Deep Learning classification architecture |
| **Optimization** | `torch.optim (AdamW)` | Gradient-based parameter optimization |
| **Metrics** | `scikit-learn` | Precision, Recall, F1-score, Confusion Matrix |

## Repository Structure

```text
.
├── text_classification.ipynb    # Primary Jupyter Notebook containing the full pipeline
├── requirements.txt             # Project dependencies and minimum versions
├── .gitignore                   # Ignore rules for checkpoints, cache, and virtualenvs
└── README.md                    # Project documentation
```

## Key Technical Skills Demonstrated
* **Deep Learning & NLP:** Fine-tuning Transformer models (DistilBERT), understanding attention mechanisms, sequence classification.
* **PyTorch Ecosystem:** Custom Dataset implementation, DataLoader batching, training/validation loops, gradient tracking, device management (GPU/CUDA acceleration).
* **Text Processing & EDA:** Subword tokenization, sequence length optimization, stopword handling, and visual text analytics.
* **Model Evaluation:** F1-score analysis, confusion matrix generation, multi-class/binary classification reporting.

## How to Run

```bash
# Clone the repository
git clone https://github.com/piotrjsk/cnn-facial-emotion-recognition.git
cd cnn-facial-emotion-recognition

# Install dependencies
pip install -r requirements.txt

```

---
*Developed by Piotr Jasiak & Mateusz Panek | [LinkedIn Profile](https://www.linkedin.com/in/piotrjasiak)*