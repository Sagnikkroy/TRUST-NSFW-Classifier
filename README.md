# TRUST Content Moderation Classifier

TRUST is a Vision Transformer (ViT)-based image classification model designed for automated content moderation and media safety applications. Unlike traditional binary classifiers, TRUST provides probability distributions across multiple content categories, enabling flexible moderation workflows and threshold-based decision making.

The model is trained to classify images into five moderation categories and is suitable for content filtering, review systems, and safety-focused applications.

---

## Demo

[🤗 Hugging Face Demo](https://huggingface.co/spaces/Sagnikroy/TRUST_NSFW_Detection)

---

## Table of Contents

* [Overview](#overview)
* [Model Summary](#model-summary)
* [Evaluation Results](#evaluation-results)
* [Model](#model)

---

## Overview

Modern content moderation systems often require more nuanced decisions than simple safe/unsafe classification. TRUST addresses this challenge by producing probability scores across multiple content categories, allowing developers to define custom moderation thresholds and workflows.

The model was trained on the `deepghs/nsfw_detect` dataset and generates probability distributions that can be used for automated filtering, content restriction, or manual review pipelines.

---

## Model Summary

The classifier is built on the Vision Transformer (ViT) architecture, leveraging self-attention mechanisms to capture global image context and improve classification performance.

| Feature          | Details                             |
| ---------------- | ----------------------------------- |
| Base Model       | `google/vit-base-patch16-224-in21k` |
| Input Resolution | 224 × 224                           |
| Output Type      | Softmax Probability Distribution    |
| Categories       | 5                                   |
| Framework        | PyTorch, Hugging Face Transformers  |

### Content Categories

* **Neutral** — Safe-for-work everyday imagery
* **Drawing** — Non-explicit artwork, illustrations, comics, and anime
* **Suggestive** — Non-explicit suggestive content
* **Explicit** — Explicit real-world adult content
* **Illustrated Explicit** — Explicit illustrated or animated content

---

## Evaluation Results

Metrics reported on the validation dataset.

| Metric               | Score |
| -------------------- | ----- |
| Accuracy             | 93.8% |
| Precision (Weighted) | 98.5% |
| F1 Score             | 0.93  |

### Confusion Matrix

The model was optimized to reduce false negatives in explicit-content categories while maintaining strong overall classification performance.

<p align="center">
  <img width="80%" src="benchmarks/confusion_matrix_professional.png">
</p>

---

## Model

The deployed model and inference API are available on Hugging Face Spaces:

👉 https://huggingface.co/spaces/Sagnikroy/TRUST_NSFW_Detection

---

## Technical Stack

* PyTorch
* Hugging Face Transformers
* Vision Transformer (ViT)
* FastAPI
* Hugging Face Spaces

---

## Key Features

* Multi-class content classification
* Vision Transformer-based architecture
* Probability-based moderation outputs
* Real-time inference API
* Cloud deployment through Hugging Face Spaces
* Configurable moderation thresholds
