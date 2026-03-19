# AI-Powered Speech & NLP Analysis Pipeline

This repository contains an end-to-end, modular machine learning pipeline capable of transcribing audio, extracting named entities, classifying intents, providing explainability (via SHAP), and logging experiment tracking metrics natively.

## 🌟 Features
- **Speech-to-Text (ASR):** Powered by OpenAI's Whisper (batch & streaming support).
- **NLP Pipeline:** Zero-shot intent classification and BERT-based Entity Extraction.
- **Data Augmentation:** Audio segmentation and noise injection via `librosa`.
- **Explainability:** SHAP feature importance plots for transparent NLP decisions.
- **Monitoring & Evaluation:** MLflow experiment tracking integration with Precision, Recall, and ROC-AUC metrics.

---

## 📁 Repository Structure
```
├── configs/
│   └── default.yaml         # Project configuration settings
├── data/
│   ├── train/               # Train audio splits
│   └── test/                # Test audio splits
├── experiments/             # Explainability plots (SHAP outputs)
├── notebooks/               # Jupyter notebooks for sandbox exploration
├── src/
│   ├── audio/               # Audio segmentation, augmentation, splits
│   ├── transcription/       # Whisper ASR wrapper
│   ├── nlp/                 # Entity extraction, Intent classifier
│   ├── explainability/      # SHAP integration and plotting utilities
│   ├── evaluation/          # ROC-AUC, accuracy, precision, metrics
│   └── pipeline/            # End-to-End Orchestrator & Tracking
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

---

## 🚀 Pipeline Overview

1. **Audio Ingestion:** `src.audio.processor.py` loads and optionally injects white-noise to the incoming 16kHz audio array.
2. **ASR Transcription:** The `src.transcription.asr.WhisperTranscription` wrapper converts spoken language to textual data.
3. **NLP Analysis:** The text passes through zero-shot intent classification (`facebook/bart-large-mnli`) and entity extraction (`dslim/bert-base-NER`).
4. **Explainability Logging:** The predicted intent is interpreted using `shap.Explainer` to identify the most critical keywords guiding the classification.
5. **Experiment Tracking:** The full workflow execution is logged explicitly through `mlflow` inside `src.pipeline.runner.py`.

---

## 💾 Dataset Workflow

The project expects a repository of `.wav`, `.mp3`, or `.flac` files located in the `data/` directory. 
By utilizing `src.audio.dataset.AudioDatasetManager`, the data can be parsed recursively and partitioned uniformly into `/train` and `/test` chunks for fine-tuning setups. 

---

## 🧠 Model Details

- **ASR Model:** `openai/whisper` (Base model by default, customizable via `configs/default.yaml`).
- **Classifier:** `facebook/bart-large-mnli` (Zero-shot classification pipeline).
- **Entity Extractor:** `dslim/bert-base-NER` (For Name, Location, Organization token extraction).

---

## 🔍 Explainability Insights

Machine Learning shouldn't be a black box. This pipeline embeds `SHAP` locally. When an audio file generates a transcript, that text is sent into a specialized explainer wrapper predicting which tokens influenced the intent classification.

Visual results (horizontal bar charts illustrating token score impacts against candidate labels) are compiled as `.png` plots embedded logically in the `experiments/` directory and subsequently synced as MLflow artifacts.

---

## 📊 Results Summary

During test workloads:
- Whisper demonstrates robust zero-shot transcription.
- MLflow logs hyperparameters organically.
- Zero-shot classification adapts strongly to dynamically typed configurations mapped in `configs/default.yaml` under `candidate_labels`.

To run the orchestrator independently, edit the path argument within the `__main__` entrypoint inside `src/pipeline/runner.py`.

## ⚙️ Quickstart Setup

1. Spin up your virtual environment via `venv` or `conda`.
2. Install pip constraints: `pip install -r requirements.txt`
3. Check and adjust labels in `configs/default.yaml`.
4. Run inference locally: 
   ```bash
   python -m src.pipeline.runner
   ```
