# Wise@DravidianLangTech 2026: Dialect-Aware Tamil Speech Classification and Recognition via Cross-Pipeline Embedding Transfer 🎯

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-4.30+-yellow.svg)](https://github.com/huggingface/transformers)
[![Wav2Vec2](https://img.shields.io/badge/Wav2Vec2-Tamil-green.svg)](https://huggingface.co/Harveenchadha/vakyansh-wav2vec2-tamil-tam-250)
[![Whisper](https://img.shields.io/badge/OpenAI_Whisper-Tamil-green.svg)](https://github.com/openai/whisper)
[![DravidianLangTech 2026](https://img.shields.io/badge/DravidianLangTech%202026-Classification%201st%20Place-gold.svg)](https://sites.google.com/view/dravidianlangtech-2026)
[![F1 Score](https://img.shields.io/badge/F1%20Score-0.79-brightgreen.svg)](#results)
[![Dialect Classification](https://img.shields.io/badge/Task-Dialect%20Classification-purple.svg)](#subtask-1-dialect-classification)
[![ASR](https://img.shields.io/badge/Task-Speech%20Recognition-red.svg)](#subtask-2-automatic-speech-recognition)
[![Tamil](https://img.shields.io/badge/Language-Tamil-orange.svg)](https://en.wikipedia.org/wiki/Tamil_language)

<p align="center">
  <img src="assets/architecture.png" alt="System Architecture" width="100%"/>
</p>
<p align="center"><i>Overview of the proposed dialect-aware Tamil speech processing architecture. Raw audio is preprocessed through channel selection, DC offset removal, LUFS normalization, and neural denoising. A Wav2Vec2-based dialect classifier extracts utterance-level dialect embeddings using temporal pooling strategies and an MLP head. These embeddings are transferred across pipelines to condition the Whisper-Tamil ASR model through residual or cross-attention injection, enabling dialect-aware transcription.</i></p>

---

This repository contains the implementation of the **Wise** system for the shared task on dialect-based speech processing in Tamil at **DravidianLangTech@ACL 2026**, addressing two subtasks: **(1)** four-way dialect region classification (Northern, Southern, Western, Central), and **(2)** dialectal Tamil ASR. All audio is preprocessed using loudness normalization followed by neural denoising to ensure consistent audio quality. For classification, we experiment with different model variants combining multilingual and Tamil-pretrained **Wav2Vec2** backbones with five temporal pooling strategies under frozen and partial fine-tuning settings. Our best configuration — learned attentive pooling with partial fine-tuning and a differentially-trained MLP head — achieves a macro F1 of **0.79**, securing **1st place** (0.26-point margin). For ASR, we propose two novel **dialect-conditioned Whisper** architectures (residual injection and cross-attention) that inject dialect embeddings from the trained classifier into the ASR pipeline. The best model achieved a WER of **0.90**, securing **8th place**.

**Keywords:** Dialect Embeddings, Temporal Pooling, Frozen Backbone, Residual & Cross Attention Injection, Wav2Vec2, Whisper

## 📋 Overview

Tamil is one of the oldest classical languages in the world, spoken by millions across South India, Sri Lanka, and Singapore. Its wide geographic spread has produced rich regional dialects that differ in pronunciation, rhythm, and vocabulary, making it challenging for automatic speech recognition and dialect identification systems to perform well.

Our system makes three key contributions:
1. A **preprocessing pipeline** combining LUFS normalization and neural denoising
2. A **study of 12 dialect classification variants** showing learned attentive pooling performs best
3. Two **dialect-conditioned Whisper architectures** that inject dialect embeddings into the ASR encoder

The system features a **tight integration** between both subtasks — dialect embeddings extracted from the trained classifier are injected into the Whisper ASR encoder for dialect-aware transcription.

## ✨ Key Features

- **Systematic Model Exploration**: 12 classification model variants combining 5 pooling strategies × 2 fine-tuning configurations × 2 backbones
- **Learned Attentive Pooling**: Non-linear attention mechanism that discovers dialect-discriminative temporal frames
- **Dialect-Conditioned ASR**: Novel architectures injecting dialect embeddings via encoder residual injection and decoder cross-attention
- **Robust Preprocessing**: Two-stage pipeline with LUFS loudness normalization and Facebook Denoiser DNS64
- **Differential Learning Rates**: Separate learning rates for pretrained backbone (1e-5) and randomly initialized layers (1e-4)
- **Cross-Pipeline Embedding Transfer**: Dialect embeddings from the classifier's attentive pooling layer are pre-computed and injected into Whisper

## 🗂️ Dataset

The dataset comprises spontaneous and read speech from native speakers across four regional dialect groups. All recordings are sampled at 16 kHz and captured in natural acoustic environments. The training partition contains 9.22 hours of transcribed speech.

| Dialect | Samples | Duration |
|---------|---------|----------|
| Southern | 1,427 | 2h 44m |
| Northern | 1,696 | 3h 29m |
| Western | 1,126 | 1h 59m |
| Central | 885 | 1h 08m |
| **Total** | **5,134** | **9h 22m** |

## 🏗️ Methodology

### Audio Preprocessing

The raw audio recordings exhibit considerable variation in recording conditions — including ambient noise, varying volume levels, multi-channel recordings, and DC bias. We design a two-stage preprocessing pipeline:

**Stage 1 — Audio Standardization:**
1. **Channel Selection**: For multi-channel recordings, the channel with the highest SNR is retained
2. **DC Offset Removal**: The mean of the waveform is subtracted to eliminate residual DC bias
3. **LUFS Normalization**: Loudness is normalized to −23 LUFS using the ITU-R BS.1770 standard, ensuring consistent perceptual loudness across utterances

**Stage 2 — Neural Denoising:**
Facebook's Denoiser DNS64 model — a causal U-Net encoder-decoder architecture operating directly in the time domain, preserving phase information for speech enhancement.

### Subtask 1: Dialect Classification

We conduct a systematic ablation study of 12 model variants. All variants share a common three-component architecture:

```
Raw Audio → [Wav2Vec2 Backbone] → Frame-level features (T × 768)
         → [Temporal Pooling]   → Utterance embedding (768)
         → [MLP Classifier: 768→512→256→4] → Dialect prediction
```

#### Speech Backbone

We experiment with two backbones:
- **Multilingual**: `facebook/wav2vec2-large-xlsr-53`
- **Tamil Fine-tuned**: `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`

And two fine-tuning strategies:
- **Frozen**: Entire backbone frozen; only pooling and classification layers are trained
- **Partial Fine-tuning**: Top 4 (of 12) transformer layers unfrozen and trained jointly, preventing catastrophic forgetting while allowing dialect-specific adaptation

#### Temporal Pooling Strategies

| Strategy | Description |
|----------|-------------|
| **Mean Pooling** | Masked average over all valid frames |
| **Attentive Pooling** | Learnable linear projection computes scalar attention weights, followed by softmax-weighted aggregation |
| **Learned Attentive Pooling** | Enhanced non-linear attention with a two-layer bottleneck network (768→192→1) |
| **Mean + Attentive** | Concatenation of mean-pooled and attentive-pooled vectors |
| **Learned Mean + Attentive** | Concatenation of mean-pooled and learned-attentive-pooled vectors |

#### Classification Head

The pooled utterance embedding passes through a three-layer MLP with progressively decreasing hidden dimensions, dropout of 0.3 after each ReLU activation.

### Subtask 2: Automatic Speech Recognition

We explore both dialect-conditioned and baseline approaches using **Whisper-Tamil** models (`vasista22/whisper-tamil-small` and `vasista22/whisper-tamil-medium`).

#### Dialect Embedding Extraction

For each audio sample, a forward pass through the best dialect classifier extracts the output of the learned attentive pooling layer (before the MLP head). These 768-dim embeddings capture dialect-discriminative phonetic, prosodic, and speaking rate characteristics.

#### Variant 1: Residual Injection

The dialect embedding is projected to Whisper encoder's hidden dimension via a linear layer + layer normalization, then injected into the encoder's residual stream at **every** transformer layer. This creates a persistent dialect bias throughout encoding, analogous to conditional embeddings in diffusion models.

#### Variant 2: Cross-Attention Injection

The dialect embedding is projected into a single context token and concatenated to the front of the encoder output. The decoder's cross-attention attends to this augmented sequence, conditioning transcription on dialect identity. A staged training schedule is used: projection warmup (frozen Whisper) → decoder fine-tuning.

#### Variant 3: Vanilla Whisper Baseline

Standard fine-tuning without dialect conditioning, serving as a controlled baseline.

## 📊 Results

### Subtask 1: Dialect Classification

Our best-performing variant used learned attentive pooling with partial fine-tuning of the top four Wav2Vec2 transformer layers, achieving **1st place** with a margin of 0.26 F1 over the next team:

| Rank | Team | Macro F1 |
|------|------|----------|
| **1** | **Wise** | **0.79** |
| 2 | Wave2Word | 0.53 |
| 3 | IIITK_SpeechScape | 0.48 |

Learned attentive pooling was particularly effective as it captures temporally localized dialect cues such as formant transitions and intonation rather than averaging representations across all frames.

### Subtask 2: Automatic Speech Recognition

| Rank | Team | WER |
|------|------|-----|
| 1 | CHMOD_777 | 0.51 |
| 2 | CUET_InferX | 0.54 |
| 3 | Wave2Word | 0.55 |
| **8** | **Wise** | **0.90** |

> **Note:** The dialect-conditioned Whisper models were only partially trained due to time and computational constraints. We expect that fully training them — particularly the cross-attention variant — would further reduce WER.

## 📁 Project Structure

```
DialectBasedSpeechProcessing/
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
├── assets/
│   └── architecture.png                        # System architecture diagram
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

All 12 variants use the Tamil Wav2Vec2 backbone (`Harveenchadha/vakyansh-wav2vec2-tamil-tam-250`) which produces 768-dimensional frame-level representations. Each variant differs in **(a) backbone fine-tuning** — frozen ("original") vs. unfreezing the top 4 transformer layers ("finetuned") — and **(b) temporal pooling** — how the variable-length frame sequence is aggregated into a fixed utterance embedding for classification via a 3-layer MLP.

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

## ⚙️ Experimental Setup

Experiments were conducted on an NVIDIA RTX 4080 Laptop GPU locally and on cloud instances equipped with NVIDIA RTX A5000 GPUs for computationally intensive workloads. A fixed random seed of 17 was used for reproducibility.

- **Optimizer**: AdamW for all models
- **Dialect Classification LR**: 1e-5 (encoder), 1e-4 (pooling + MLP)
- **ASR LR**: 1e-5 with weight decay 1e-2
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Evaluation**: Macro F1 (classification), WER (ASR) via `torchmetrics`

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
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)

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

## 🏆 Key Contributions

1. **Systematic Pooling Ablation**: Exhaustive comparison of 12 model variants across 5 pooling strategies and 2 fine-tuning configurations, identifying learned attentive pooling as optimal for dialect identification
2. **Cross-Pipeline Dialect Embedding Transfer**: Novel pipeline connecting dialect classification to ASR through pre-extracted dialect embeddings from the classifier's attentive pooling layer
3. **Two Novel Conditioning Strategies**: Residual injection (encoder-side, every layer) and cross-attention (decoder-side, context token) approaches for dialect-aware ASR
4. **Robust Preprocessing**: LUFS normalization + neural denoising pipeline tailored for field recordings with varying conditions
5. **State-of-the-Art Performance**: 1st place in dialect classification with 0.26 F1 margin over 2nd place

## ⚠️ Limitations

1. The training dataset is relatively small (9.22 hours of speech), limiting the ability of large speech models to fully adapt to dialectal variations
2. The dataset is class-imbalanced (Central: 885 vs. Northern: 1,696 samples), which may reduce recall for underrepresented dialect groups
3. Some transcriptions in the dataset contain minor inaccuracies or partially incorrect words, introducing noise during ASR training and evaluation

## 👥 Authors

- **Ganesh Sundhar S**
- **Hari Krishnan N**
- **Gnanasabesan G**
- **Suriya KP**
- **Jyothish Lal G** *(Supervisor)*

**Affiliation**: Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India

## 🙏 Acknowledgments

We thank the organizers of the DravidianLangTech@ACL 2026 shared task for curating the Tamil dialect speech corpus and providing the evaluation infrastructure.
