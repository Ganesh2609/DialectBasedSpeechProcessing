# Wise@DravidianLangTech 2026: Cross-Pipeline Dialect Embedding Transfer for Tamil Speech Classification and Recognition 🎯

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://github.com/huggingface/transformers)
[![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-Tamil-green.svg)](https://huggingface.co/Harveenchadha/vakyansh-wav2vec2-tamil-tam-250)
[![Whisper](https://img.shields.io/badge/OpenAI_Whisper-Small-green.svg)](https://github.com/openai/whisper)
[![DravidianLangTech 2026](https://img.shields.io/badge/DravidianLangTech%202026-Classification%201st%20Place-gold.svg)](https://sites.google.com/view/dravidianlangtech-2026)
[![F1 Score](https://img.shields.io/badge/F1%20Score-0.79-brightgreen.svg)](#results)
[![Dialect Classification](https://img.shields.io/badge/Task-Dialect%20Classification-purple.svg)](#subtask-1-dialect-classification)
[![ASR](https://img.shields.io/badge/Task-Speech%20Recognition-red.svg)](#subtask-2-automatic-speech-recognition)
[![Tamil](https://img.shields.io/badge/Language-Tamil-orange.svg)](https://en.wikipedia.org/wiki/Tamil_language)

This repository contains the implementation of our system for the **DravidianLangTech@ACL 2026** shared task on dialect-based speech recognition and classification in Tamil. Our approach achieved **1st place** in Subtask 1 (Dialect Classification) with a macro F1 score of **0.79**, and ranked 8th in Subtask 2 (ASR) with a WER of **0.90**.

## 📋 Overview

We address two subtasks for dialectal Tamil speech:
- **Subtask 1 — Dialect Classification**: Classifying speech into four Tamil dialect regions (Northern, Southern, Western, Central) using a Tamil Wav2Vec2 backbone with learned attentive pooling
- **Subtask 2 — Automatic Speech Recognition**: Transcribing dialectal Tamil speech using Whisper models with novel dialect conditioning strategies

Our system features a **tight integration** between both subtasks — dialect embeddings extracted from the trained classifier are injected into the Whisper ASR encoder for dialect-aware transcription.

## ✨ Key Features

- **Systematic Model Exploration**: 12 classification model variants combining 6 pooling strategies × 2 fine-tuning configurations
- **Learned Attentive Pooling**: Non-linear attention mechanism that discovers dialect-discriminative temporal frames
- **Dialect-Conditioned ASR**: Novel architectures injecting dialect embeddings via encoder residual injection and decoder cross-attention
- **Robust Preprocessing**: Two-stage pipeline with LUFS loudness normalization and Facebook Denoiser DNS64
- **Differential Learning Rates**: Separate learning rates for pretrained backbone and randomly initialized layers
- **Comprehensive Evaluation**: Macro F1 for classification, WER for ASR, tracked with `torchmetrics`

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/PLACEHOLDER/DialectSpeechProcessing.git
cd DialectSpeechProcessing

# Install requirements
pip install -r requirements.txt
```

## 📁 Project Structure

```
DialectSpeechProcessing/
│
├── 1 Dataset/                                  # Raw & intermediate processed audio
│
├── 2 Data Initial Preprocessing/
│   ├── 📔 generate_separate_csv.ipynb          # Per-dialect CSV generation from raw data
│   └── 📔 merge_all_csv.ipynb                  # Merge into unified transcripts.csv
│
├── 3 Preprocessing/
│   ├── 📔 1_audio_standardization.ipynb        # Channel selection, resampling, LUFS normalization
│   └── 📔 2_rnnoise_denoiser.ipynb             # Facebook Denoiser DNS64 speech enhancement
│
├── 4 Final Dataset/
│   ├── Train/                                  # Training audio + transcripts.csv
│   │   ├── Central_Dialect/
│   │   ├── Northern_Dialect/
│   │   ├── Southern_Dialect/
│   │   └── Western_Dialect/
│   └── Test/                                   # Test audio (unlabeled)
│
├── 5 Dialect Classification/
│   ├── 1_original_average_attentive_entire/
│   ├── 2_finetuned_average_attentive/
│   ├── 3_original_average/
│   ├── 4_original_attentive/
│   ├── 5_finetuned_average/
│   ├── 6_finetuned_attentive/
│   ├── 7_original_average_attentive/
│   ├── 8_original_learned_attentive/
│   ├── 9_original_learned_average_attentive/
│   ├── 10_finetuned_learned_attentive/
│   ├── 11_finetuned_learned_average_attentive/
│   ├── 12_finetuned_learned_attentive_entire/  # ⭐ Best (F1=0.79)
│   ├── Final_Results/                          # Comparison notebook + bar plots
│   └── features/                               # Pre-extracted feature analysis
│
├── 6 Automatic Speech Recognition/
│   ├── 📘 extract_dialect_embeddings.py        # Extract embeddings from trained classifier
│   ├── 1_dialect_conditioned_residual_medium/
│   ├── 2_dialect_conditioned_residual_small/
│   ├── 3_cloud_dialect_conditioned_residual_medium/
│   ├── 4_dialect_conditioned_cross_attention_small/
│   ├── 5_vanilla_whisper_small/
│   ├── 6_final_vanilla_whisper_small/          # ⭐ Final submission
│   │   ├── Central/
│   │   ├── Northern/
│   │   ├── Southern/
│   │   ├── Western/
│   │   └── Unified/
│   └── 7_cloud_final_vanilla_whisper_medium/
│       ├── Central/
│       ├── Northern/
│       ├── Southern/
│       ├── Western/
│       └── Unified/
│
├── 7 Final Results/                            # Competition submission files
│   ├── 📄 Wise_Classification_Run1-3.txt
│   ├── 📄 Wise_Recognition_Run1-2.txt
│   └── 📄 submission_data.txt
│
├── .gitignore
├── requirements.txt
└── 📖 README.md
```

Each model variant folder contains:
```
variant_folder/
├── 📘 model.py                 # Model architecture definition
├── 📘 dataset.py               # Data loading and preprocessing
├── 📘 trainer.py               # Training/evaluation loop
├── 📘 training_*.py            # Main training script with hyperparameters
├── 📘 transcription_generator.py  # (ASR only) Inference script
└── train_data/
    ├── checkpoints/            # Model checkpoints (gitignored)
    ├── logs/                   # Training logs
    └── graphs/                 # Training metric plots
```

### Dialect Classification — Variant Descriptions

All 12 variants use the Tamil Wav2Vec2 backbone (`Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`) which produces 768-dimensional frame-level representations. Each variant differs in **(a) backbone fine-tuning** — frozen (``original'') vs. unfreezing the top 4 transformer layers (``finetuned'') — and **(b) temporal pooling** — how the variable-length frame sequence is aggregated into a fixed utterance embedding for classification via a 3-layer MLP.

| # | Variant | Fine-tuning | Pooling | Description |
|---|---------|-------------|---------|-------------|
| 1 | `original_average_attentive_entire` | Frozen (XLSR-53) | Mean + Attentive | Concatenation of mean-pooled and attentive-pooled vectors (1536-dim) using XLSR-53 backbone |
| 2 | `finetuned_average_attentive` | Top 4 layers | Mean + Attentive | Same pooling as #1 but with Tamil Wav2Vec2 fine-tuning |
| 3 | `original_average` | Frozen | Mean | Simple masked mean pooling over all valid frames |
| 4 | `original_attentive` | Frozen | Attentive | Learnable linear projection computes softmax attention weights over frames |
| 5 | `finetuned_average` | Top 4 layers | Mean | Fine-tuned backbone with mean pooling |
| 6 | `finetuned_attentive` | Top 4 layers | Attentive | Fine-tuned backbone with linear attention pooling |
| 7 | `original_average_attentive` | Frozen | Mean + Attentive | Frozen Tamil Wav2Vec2 with concatenated mean + attentive (1536-dim) |
| 8 | `original_learned_attentive` | Frozen | Learned Attentive | Two-layer bottleneck network (768→192→1) with tanh — learns non-linear frame importance |
| 9 | `original_learned_average_attentive` | Frozen | Mean + Learned Attn | Concatenation of mean pooling and learned attentive pooling (1536-dim) |
| 10 | `finetuned_learned_attentive` | Top 4 layers | Learned Attentive | Fine-tuned backbone with non-linear learned attentive pooling |
| 11 | `finetuned_learned_average_attentive` | Top 4 layers | Mean + Learned Attn | Fine-tuned backbone with concatenated mean + learned attentive |
| **12** | **`finetuned_learned_attentive_entire`** | **Top 4 layers** | **Learned Attentive** | **⭐ Best model — learns which temporal frames carry dialect-discriminative cues (F1=0.79, Rank 1)** |

### ASR — Variant Descriptions

For ASR, dialect embeddings (768-dim) are first extracted from the best classification model's attentive pooling layer (before the MLP head) using `extract_dialect_embeddings.py`. These embeddings encode dialect-discriminative phonetic and prosodic information and are used to condition Whisper models.

| # | Variant | Model | Conditioning | Description |
|---|---------|-------|-------------|-------------|
| 1 | `dialect_conditioned_residual_medium` | Whisper-Tamil-Medium | Residual Injection | Projected dialect embedding added to encoder hidden states at every transformer layer — creates persistent dialect bias throughout encoding |
| 2 | `dialect_conditioned_residual_small` | Whisper-Tamil-Small | Residual Injection | Same residual injection strategy as #1 with smaller Whisper model |
| 3 | `cloud_dialect_conditioned_residual_medium` | Whisper-Tamil-Medium | Residual Injection | Cloud-trained variant of #1 with extended GPU training |
| 4 | `dialect_conditioned_cross_attention_small` | Whisper-Tamil-Small | Cross-Attention | Dialect embedding projected into a context token prepended to encoder output; decoder cross-attention attends to it alongside acoustic features. Uses staged training (projection warmup → decoder fine-tuning) |
| 5 | `vanilla_whisper_small` | Whisper-Tamil-Small | None | Standard fine-tuning baseline without any dialect conditioning |
| **6** | **`final_vanilla_whisper_small`** | **Whisper-Tamil-Small** | **None** | **⭐ Final submission — dialect-specific + unified fine-tuning across 4 regions (WER=0.90, Rank 8)** |
| 7 | `cloud_final_vanilla_whisper_medium` | Whisper-Tamil-Medium | None | Cloud-trained Whisper-Medium with per-dialect + unified fine-tuning |

> **Note:** The dialect-conditioned variants (#1–4) were not fully trained due to time constraints. The final submission used the vanilla Whisper-Tamil-Small model (#6) with greedy decoding, repetition penalty of 1.5, and n-gram repeat prevention.

## 🔧 Usage

### 1. Data Preprocessing

#### Stage 1: Audio Standardization
```python
jupyter notebook "3 Preprocessing/1_audio_standardization.ipynb"
```
Applies: best channel selection (SNR-based), 16 kHz resampling, DC offset removal, LUFS normalization to −23 LUFS, and safe peak limiting.

#### Stage 2: Neural Denoising
```python
jupyter notebook "3 Preprocessing/2_rnnoise_denoiser.ipynb"
```
Applies Facebook Denoiser DNS64 — a causal U-Net operating in the time domain for speech enhancement.

### 2. Dialect Classification (Subtask 1)

```python
# Train the best-performing model (Variant 12)
python "5 Dialect Classification/12_finetuned_learned_attentive_entire/training_wav2vec.py"
```

Training configuration:
- Backbone: `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`
- Pooling: Learned Attentive Pooling (768→192→1 bottleneck with tanh)
- Unfrozen Layers: Top 4 transformer layers
- Optimizer: AdamW (encoder lr=1e-5, classifier lr=1e-4)
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
- Batch Size: 4
- Epochs: 20

### 3. Automatic Speech Recognition (Subtask 2)

#### Extract Dialect Embeddings
```python
python "6 Automatic Speech Recognition/extract_dialect_embeddings.py"
```
Extracts 768-dim embeddings from the trained classifier's attentive pooling layer.

#### Train ASR Model
```python
# Dialect-conditioned (residual injection)
python "6 Automatic Speech Recognition/2_dialect_conditioned_residual_small/training_whisper.py"

# Vanilla Whisper baseline
python "6 Automatic Speech Recognition/5_vanilla_whisper_small/training_whisper.py"
```

Training configuration:
- Model: `vasista22/whisper-tamil-small`
- Optimizer: AdamW (lr=1e-5, weight_decay=1e-2)
- Scheduler: ReduceLROnPlateau
- Batch Size: 2
- Decoding: Greedy with repetition penalty (1.5) and n-gram blocking (size=3)

## 🏗️ Methodology

### Dialect Classification Architecture

```
Raw Audio → [Wav2Vec2 Tamil Backbone] → Frame-level features (T × 768)
         → [Learned Attentive Pooling] → Utterance embedding (768)
         → [MLP Classifier: 768→512→256→4] → Dialect prediction
```

**Learned Attentive Pooling** uses a two-layer bottleneck network (768 → 192 → 1) with tanh nonlinearity to learn non-linear importance scores over temporal frames, automatically discovering which frames carry dialect-discriminative information.

### Dialect-Conditioned ASR Architectures

**Residual Injection**: Projects dialect embedding (768-dim) and adds it to the Whisper encoder's hidden states at every transformer layer.

**Cross-Attention**: Projects dialect embedding into a single context token, concatenates it to the encoder output, and lets the decoder attend to it alongside acoustic features.

## 📊 Results

### Subtask 1: Dialect Classification

Our system achieved **1st place** with a significant margin of 0.26 F1 points over the 2nd place team:

| Rank | Team Name | Macro F1 |
|------|-----------|----------|
| **1** | **Wise** | **0.79** |
| 2 | Wave2Word | 0.53 |
| 3 | IIITK_SpeechScape | 0.48 |
| 4 | GigitAI | 0.45 |
| 5 | CHMOD_777 | 0.43 |

### Subtask 2: Automatic Speech Recognition

| Rank | Team Name | WER |
|------|-----------|-----|
| 1 | CHMOD_777 | 0.51 |
| 2 | CUET_InferX | 0.54 |
| 3 | Wave2Word | 0.55 |
| 4 | DLRG | 0.55 |
| 5 | IIITK_SpeechScape | 0.57 |
| **8** | **Wise** | **0.90** |

## 🗂️ Dataset

The Tamil Dialect Speech Dataset consists of spontaneous and read speech from native speakers across four dialect groups, recorded at 16 kHz:

| Dialect | Samples | Duration |
|---------|---------|----------|
| Southern | 1,427 | 2:44:30 |
| Northern | 1,696 | 3:29:15 |
| Western | 1,126 | 1:59:59 |
| Central | 885 | 1:08:18 |
| **Total** | **5,134** | **9:22:02** |

Test set: 2.05 hours of unlabeled audio.

## 🏆 Key Contributions

1. **Systematic Pooling Ablation**: Exhaustive comparison of 12 model variants across 6 pooling strategies and 2 fine-tuning configurations, identifying learned attentive pooling as optimal for dialect identification
2. **Dialect Embedding Transfer**: Novel pipeline connecting dialect classification to ASR through pre-extracted dialect embeddings
3. **Two Conditioning Strategies**: Residual injection (encoder-side) and cross-attention (decoder-side) approaches for dialect-aware ASR
4. **Robust Preprocessing**: LUFS normalization + neural denoising pipeline tailored for field recordings
5. **State-of-the-Art Performance**: 1st place in dialect classification with 0.26 F1 margin over 2nd place

## 👥 Authors

1. **Ganesh Sundhar S**
2. **Hari Krishnan N**
3. **Gnanasabesan G**
4. **Suriya KP**

**Affiliation**: Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India

## 🙏 Acknowledgments

We thank the organizers of the DravidianLangTech@ACL 2026 shared task for providing the dataset and evaluation framework.
