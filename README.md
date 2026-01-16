# TRUST-NSFW-Classifier
<div align="center">
  <img src="assets\trustnsfwicon.png" width="60%" alt="TRUST NSFW Soft Model V1" />
</div>
<hr>
<div align="center" style="line-height: 2;">
  
  <a href="https://huggingface.co/Sagnikroy/TRUST-NSFW-Soft-V1/tree/main"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sagnik%20Roy-ffc107?style=flat&logoColor=white"/></a>
  &nbsp;
  <a href="https://x.com/sagnikkroy"><img alt="Twitter Follow"
    src="https://img.shields.io/badge/Twitter-sagnikkroy-white?style=flat&logo=x&logoColor=black"/></a>
  
  <br>

  <a href="https://github.com/Sagnikkroy"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-Sagnikkroy-181717?style=flat&logo=github&logoColor=white"/></a>
  &nbsp;
  <a href="https://www.linkedin.com/in/sagnikkroy/"><img alt="LinkedIn"
    src="https://img.shields.io/badge/LinkedIn-Sagnik%20Roy-0077B5?style=flat&logo=linkedin&logoColor=white"/></a>

</div>

<hr>

# Introduction

**TRUST-NSFW-Classifier** is a high-performance, deep-learning solution engineered to solve the "Binary Bottleneck" of modern content moderation. While traditional classifiers often fail at the nuance between "artistic" and "explicit," TRUST leverages state-of-the-art **Vision Transformer (ViT)** architecture to deliver granular, multi-dimensional safety scores.

By moving beyond simple "Safe/Unsafe" labels, this model provides a **probabilistic transparency layer**. It allows platforms to automate complex moderation workflows—distinguishing between safe-for-work(SFW) digital art, suggestive "sexy" content, and hardcore explicit material with surgical precision. 

Built for speed, accuracy, and reliability, **TRUST** sets a new standard for open-source safety tools, ensuring that communities stay safe without sacrificing the freedom of legitimate creators.

---

## Table of Contents
* [Introduction](#-introduction)
* [Model Summary](#-model-summary)
* [Dataset](#-dataset)
* [Evaluation Results](#-evaluation-results)
* [Installation & Usage](#-installation--usage)
* [Contact](#-contact)

---

## Introduction
In digital content moderation, a binary "Safe/Unsafe" toggle is often insufficient for modern platforms. **TRUST-NSFW-Classifier** addresses this by providing **probabilistic outputs**. 

By utilizing the `deepghs/nsfw_detect` dataset, this model categorizes images into five specific levels of safety. This allows developers to implement "Soft Moderation"—such as applying a blur filter only when the "Porn" or "Hentai" probability exceeds a specific threshold (e.g., 85%), or flagging "Sexy" content for manual human review.

---

## Model Summary
The classifier is built upon the **Vision Transformer (ViT)** architecture, which uses self-attention mechanisms to understand global image context rather than just local pixel patterns.

| Feature | Details |
| :--- | :--- |
| **Base Model** | `google/vit-base-patch16-224-in21k` |
| **Input Size** | 224 x 224 pixels |
| **Output Type** | Softmax Probabilities (0.0 - 1.0) |
| **Classes** | 5 (Neutral, Drawing, Sexy, Porn, Hentai) |
| **Framework** | PyTorch & Hugging Face Transformers |



---

<p align="center">
  <img width="80%" src="benchmarks/demo_sample_0.png">
</p>


**Class Definitions:**
* **Neutral:** Completely safe-for-work, everyday imagery.
* **Drawing:** Safe-for-work art, comics, and anime.
* **Sexy:** Suggestive content, lingerie, or provocative poses (Non-explicit).
* **Porn:** Explicit real-life adult content.
* **Hentai:** Explicit anime or illustrated adult content.

---

## Evaluation & Benchmarks
*Note: These metrics represent the model performance on the validation split.*

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | 94.2% |
| **Precision (Weighted)** | 0.93 |
| **Recall (Weighted)** | 0.94 |
| **F1-Score** | 0.93 |

### Confusion Matrix
Our training focuses on minimizing **False Negatives** in the "Porn" and "Hentai" categories to ensure maximum platform safety.
<p align="center">
  <img width="80%" src="benchmarks/confusion_matrix.png">
</p>




---
### Model 
The model is available on Huggingface please follow this link
 <a href="https://huggingface.co/Sagnikroy/TRUST-NSFW-Soft-V1/tree/main"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Sagnik%20Roy-ffc107?style=flat&logoColor=white"/></a>

