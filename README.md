# Detecting AI-Generated Speech Using Neural Networks

**Authors:** Dev Patel, Arnav Mejari, Onkar Shinde  
**Course:** CSC 525 –- Neural Networks and Deep Learning  

---

## Problem Statement

Recent advances in generative AI have made synthetic speech nearly indistinguishable from real human voice. This introduces risks such as:

- Voice impersonation fraud  
- Compromised identity verification  
- Misinformation through synthetic media  

This project formulates the task as:

> Given an audio clip → classify it as **Real (0)** or **AI-generated (1)**

---

## Dataset

We use the **Fake-or-Real Speech Dataset (FOR-norm subset)**:

- Balanced dataset (~17.8K samples)
- Preprocessed for:
  - Consistent sample rate
  - Normalized amplitude
  - Uniform audio format

| Split       | Real | Fake | Total |
|------------|------|------|-------|
| Training   | 6,978 | 6,978 | 13,956 |
| Validation | 1,413 | 1,413 | 2,826 |
| Test       | 544   | 544   | 1,088 |

---

## Approach Overview

This project is implemented as a **single end-to-end Jupyter Notebook pipeline** that includes:

### 1. Audio Preprocessing
- Load raw `.wav` files  
- Normalize duration (trim/pad)  

### 2. Feature Extraction
- Convert audio → **MFCC (Mel-Frequency Cepstral Coefficients)**  
- Captures spectral + temporal characteristics  

### 3. Model Training
Multiple architectures explored:
- Logistic Regression (baseline)
- CNN
- CNN + BiLSTM + Attention
- CNN + Transformer
- ResNet
- **Temporal Convolutional Network (TCN)**

### 4. Evaluation
- Accuracy  
- F1 Score  
- ROC-AUC  
- Confusion Matrix  

### 5. Threshold Optimization
- Adjust classification threshold to control:
  - False positives  
  - False negatives  

---

## Best Model: Threshold-Tuned TCN

### Performance:
- **Accuracy:** 97.9%  
- **F1 Score:** 0.979  
- **ROC-AUC:** 0.998  

---

## Threshold Trade-off

| Model Variant        | False Positives | False Negatives | Fake Detected |
|---------------------|----------------|-----------------|--------------|
| Original TCN        | 3              | 31              | 513 / 544    |
| Threshold-Tuned TCN | 13             | 10              | 534 / 544    |

### Interpretation:
- **Original TCN** → safer (fewer false accusations)  
- **Tuned TCN** → more aggressive (better fake detection)  

---

## Notebook Structure

The entire workflow is contained in a **single notebook**, which typically follows this sequence:

1. Data loading & inspection  
2. Audio preprocessing  
3. MFCC feature extraction  
4. Dataset preparation (train/val/test)  
5. Model definitions  
6. Training loops  
7. Evaluation & metrics  
8. Model comparison  
9. Threshold tuning  
10. Final results & analysis  

---

## Key Insights

- MFCC features effectively capture artifacts in synthetic speech  
- Temporal models (especially TCN) outperform static classifiers  
- Threshold tuning is critical for real-world deployment  
- Trade-offs depend on application:
  - Security → prioritize detection  
  - Verification → minimize false positives  

---
## How to Run

1. Clone the repository  
2. Open the notebook:
```
persistent_speech_pipeline_notebook.ipynb
```
3. Run cells sequentially  

> GPU recommended for faster training

---

## References

1. Mohammed Abdeldayem - *Fake-or-Real Dataset*  
2. Davis & Mermelstein (1980) - MFCC  
3. Bai et al. (2018) - Temporal Convolutional Networks  

---

## Project Artifact

Notebook and results available here

---

## Summary

This project demonstrates that:

> **AI-generated speech can be reliably detected using MFCC-based temporal modeling, with TCN achieving near state-of-the-art performance.**

---







