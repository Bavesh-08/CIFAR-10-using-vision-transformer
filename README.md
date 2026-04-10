# 🔭 Vision Transformer (ViT) — CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)
![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-61.1%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A clean, from-scratch implementation of the **Vision Transformer (ViT)** architecture using **TensorFlow / Keras**, trained on the CIFAR-10 dataset — no external data download required.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Hyperparameters](#hyperparameters)
- [How It Works](#how-it-works)
- [Improvements](#improvements)

---

## Overview

Vision Transformers apply the Transformer architecture — originally designed for NLP — directly to image patches. Instead of convolutional filters, ViT splits an image into a sequence of fixed-size patches, embeds them linearly, and processes them with standard multi-head self-attention.

This implementation covers the complete pipeline:

```
Image → Patch Embedding → [CLS] + Positional Encoding
      → 4 × Transformer Encoder Blocks
      → MLP Classification Head → 10 Classes
```

---

## Architecture

| Component | Details |
|---|---|
| Input | 32×32×3 (CIFAR-10) |
| Patch Size | 4×4 → 64 patches per image |
| Projection Dim | 64 |
| Transformer Blocks | 4 |
| Attention Heads | 4 (key_dim = 16) |
| MLP Hidden Dim | 128, GELU activation |
| Dropout | 0.1 |
| Output | Dense(10, softmax) |
| **Total Parameters** | **150,986** |

### Model Summary

```
Layer                        Output Shape          Params
─────────────────────────────────────────────────────────
input_layer                  (None, 32, 32, 3)          0
patch_embedding              (None, 64, 64)          3,136
pos_encoding (+ CLS token)   (None, 65, 64)          4,224
transformer_block_0          (None, 65, 64)         33,472
transformer_block_1          (None, 65, 64)         33,472
transformer_block_2          (None, 65, 64)         33,472
transformer_block_3          (None, 65, 64)         33,472
layer_normalization          (None, 65, 64)            128
[CLS] extraction             (None, 64)                  0
dense (MLP head)             (None, 128)             8,320
dense (classifier)           (None, 10)              1,290
─────────────────────────────────────────────────────────
Total                                              150,986
```

---

## Results

| Metric | Value |
|---|---|
| Final Train Accuracy | 63.99% |
| Final Val Accuracy | 63.00% |
| **Test Accuracy** | **61.13%** |
| Test Loss | 1.0722 |
| Epochs Run | 20 / 20 |
| Training Time | ~3 min (GPU) |

### Training Curve

```
Epoch  1 → Train: 30.5%  Val: 41.5%
Epoch  5 → Train: 52.7%  Val: 55.3%
Epoch 10 → Train: 58.0%  Val: 57.7%
Epoch 15 → Train: 61.5%  Val: 59.6%
Epoch 20 → Train: 64.0%  Val: 63.0%
```

> **Note:** ViT models typically underperform CNNs on small datasets without pretraining. Accuracy above 60% from scratch on CIFAR-10 (32×32) is a solid baseline. See [Improvements](#improvements) for ideas to push further.

---

## Project Structure

```
vision_transformer/
│
├── vision_transformer.py   # Full model + training + inference
└── README.md
```

All components live in a single file for clarity:

| Section | Description |
|---|---|
| `load_data()` | Loads and normalizes CIFAR-10 from `tf.keras.datasets` |
| `PatchEmbedding` | Extracts 4×4 patches and projects them to 64D |
| `PositionalEncoding` | Adds learnable [CLS] token + positional embeddings |
| `TransformerBlock` | LayerNorm → MHSA → residual → LayerNorm → MLP → residual |
| `build_vit()` | Assembles the full Keras model |
| `train()` | Trains with Adam, EarlyStopping, ReduceLROnPlateau |
| `predict_single()` | Single-image inference with class name + confidence |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/vision-transformer-cifar10.git
cd vision-transformer-cifar10

# Install dependencies
pip install tensorflow numpy
```

**Requirements:**
- Python 3.8+
- TensorFlow 2.x
- NumPy

The CIFAR-10 dataset (~160 MB) is downloaded automatically on first run via `tf.keras.datasets.cifar10`.

---

## Usage

### Train the model

```bash
python vision_transformer.py
```

This will:
1. Download CIFAR-10 automatically
2. Print the model summary
3. Train for up to 20 epochs (with early stopping)
4. Print test accuracy
5. Run a demo prediction on one test image

### Inference on a single image

```python
from vision_transformer import build_vit, predict_single
import numpy as np

model = build_vit()
model.load_weights("vit_weights.h5")  # if saved

# image: numpy array (32, 32, 3), values in [0, 1]
predict_single(model, image)
# → Prediction: cat  (55.6% confidence)
```

### Save / load weights

```python
# Save
model.save_weights("vit_cifar10.h5")

# Load
model = build_vit()
model.load_weights("vit_cifar10.h5")
```

---

## Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | 32 | Input image size |
| `PATCH_SIZE` | 4 | Size of each square patch |
| `PROJECTION_DIM` | 64 | Patch embedding dimension |
| `NUM_HEADS` | 4 | Attention heads per block |
| `TRANSFORMER_LAYERS` | 4 | Number of encoder blocks |
| `MLP_DIM` | 128 | Feed-forward hidden size |
| `DROPOUT_RATE` | 0.1 | Dropout throughout |
| `BATCH_SIZE` | 64 | Training batch size |
| `EPOCHS` | 20 | Max training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |

---

## How It Works

### 1. Patch Embedding
Each 32×32 image is divided into sixty-four non-overlapping 4×4 patches. Each patch is flattened into a 48-dimensional vector (4×4×3) and linearly projected to 64D using a `Dense` layer.

### 2. Positional Encoding
A learnable `[CLS]` token is prepended to the patch sequence (making it 65 tokens). Learnable positional embeddings are added so the model knows the spatial order of patches.

### 3. Transformer Encoder
Each block applies:
- **LayerNorm** → **Multi-Head Self-Attention** (4 heads) → **residual connection**
- **LayerNorm** → **MLP** (Dense 128, GELU, Dense 64) → **residual connection**

### 4. Classification
After all 4 blocks, only the `[CLS]` token's representation is passed to a 2-layer MLP head that outputs a 10-class softmax distribution.

---

## Improvements

To push accuracy beyond 61%, try:

- **Data augmentation** — random flips, crops, cutout, mixup
- **Larger model** — increase `PROJECTION_DIM` to 128, add more blocks
- **Warmup + cosine decay schedule** — ViTs are sensitive to learning rate scheduling
- **Pre-training** — fine-tune from an ImageNet-pretrained ViT checkpoint
- **Higher resolution** — resize CIFAR-10 to 64×64 or 96×96 before patching
- **Stochastic depth** — regularization technique from DeiT paper

---

## CIFAR-10 Classes

`airplane` · `automobile` · `bird` · `cat` · `deer` · `dog` · `frog` · `horse` · `ship` · `truck`

---
