(((
# Lightweight Hybrid Deepfake Video Detection Using MobileNetV2 and Temporal Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![Dataset](https://img.shields.io/badge/Dataset-FaceForensics++-green)
![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey)

**Group 24 | PRT 840 Thesis | Charles Darwin University | 2026**

---

## Overview

This repository contains the complete implementation, experimental results, 
and trained model for a lightweight hybrid deepfake video detection system. 
The system combines a MobileNetV2 spatial classifier with a frame-difference 
temporal inconsistency module, evaluated on FaceForensics++ C23 using a 
controlled ablation study.

The central research question was:

> *Does adding a frame-difference temporal module to a lightweight MobileNetV2 
> spatial classifier improve deepfake detection under high-quality C23 
> compression — and if not, why?*

---

## Key Results

| Metric | CNN-Only | Hybrid System | Change |
|--------|----------|---------------|--------|
| Accuracy | 61.86% | 58.33% | ▼ 3.53 pp |
| Precision | 73.18% | 81.25% | ▲ 8.07 pp |
| **Recall (Fake)** | **37.44%** | **21.67%** | **▼ 15.77 pp** |
| F1-Score | 49.54% | 34.21% | ▼ 15.33 pp |

**Quantization Results (FP32 → INT8)**

| Property | FP32 | INT8 | Gain |
|----------|------|------|------|
| Model Size | 23.76 MB | 2.74 MB | 8.7× smaller |
| Inference Time | 173.18 ms/frame | 12.20 ms/frame | 14.2× faster |

---

## Five Key Findings

1. **Within-Dataset Overfitting** — 92% training accuracy vs 61.86% test 
   accuracy. The CNN memorises compression artefacts not universal deepfake 
   features.

2. **Naive Temporal Fusion Costs 15.77 pp Recall** — Under C23 compression, 
   pixel-difference features cannot distinguish synthetic from authentic 
   motion. The temporal module actively biased predictions toward Real, 
   missing 78% of fake videos.

3. **Quantization Exceeds Theory** — 8.7× size reduction and 14.2× speedup 
   exceed the theoretical 4× prediction due to MobileNetV2's depthwise 
   separable convolution architecture. The 2.74 MB INT8 model is directly 
   deployable on mobile and edge hardware.

4. **Precision-Recall Safety Risk** — 81% precision but only 22% recall means 
   4 in 5 deepfakes pass undetected. A dangerous false sense of security in 
   real-world deployment.

5. **Evaluation Granularity Matters** — Frame-level and video-level evaluation 
   of the same model produce different accuracy figures. First controlled 
   within-study demonstration of this effect.

## Repository Structure

```
deepfake-detection-group24/
├── deepfake_detection.ipynb
├── results/
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_hybrid.png
│   ├── training_curves.png
│   ├── cnn_vs_hybrid_comparison.png
│   ├── quantization_comparison.png
│   ├── training_history.json
│   └── final_metrics.json
├── baseline_model_quantized.tflite
└── README.md
```
## System Architecture
Input Video (FaceForensics++ C23)
↓
Frame Extraction (30 frames per video, 224×224)
↓
┌───────────────────┐    ┌──────────────────────┐
│  MobileNetV2 CNN  │    │   Temporal Module     │
│  Spatial Score    │    │   Frame-Difference    │
│  (per-frame avg)  │    │   Score (normalised)  │
└────────┬──────────┘    └──────────┬───────────┘
│                          │
└──────────┬───────────────┘
↓
Final Score = 0.7 × CNN + 0.3 × Temporal
↓
≥ 0.5 → FAKE  |  < 0.5 → REAL

---

## Setup and Usage

### Requirements

```bash
pip install tensorflow==2.x
pip install opencv-python
pip install scikit-learn
pip install matplotlib
```

### Run the Notebook

1. Open `deepfake_detection.ipynb` in Google Colab
2. Mount your Google Drive
3. Update the dataset path to your FaceForensics++ download location
4. Run all cells in order

### Run Inference on a Single Video

```python
# Load the quantized model
import tensorflow as tf
interpreter = tf.lite.Interpreter(
    model_path='baseline_model_quantized.tflite'
)
interpreter.allocate_tensors()
# See notebook Section 4.9 for full end-to-end inference code
```

---

## Dataset

This project uses **FaceForensics++** under C23 compression.

- Access requires registration at: 
  https://github.com/ondyari/FaceForensics
- Only the **Deepfakes** manipulation method was used
- 400 real + 400 fake videos (subset of the full dataset)

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | MobileNetV2 (ImageNet pretrained) |
| Unfrozen layers | Last 40 layers |
| Optimizer | Adam |
| Learning rate | 5e-5 |
| Loss function | Binary Crossentropy |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping patience | 5 |
| Best checkpoint | Epoch 6 (val_acc = 67.86%) |
| Platform | Google Colab — NVIDIA T4 GPU |

---

## Authors

| Name | Student ID |
|------|------------|
| Amanparteek Singh | 391075 |
| Harsimranjit Kaur | 390359 |
| Darshan Meti | 388441 |
| Rahul Sharma | 388446 |

---

## Supervisor

**Charles Yeo**
Senior Lecturer — Information Technology
Charles Darwin University

---

## Citation

If you use this work please cite:
Singh, A., Kaur, H., Meti, D. & Sharma, R. (2026).
Lightweight Hybrid Deepfake Video Detection Using MobileNetV2
and Temporal Analysis. PRT 840 Thesis,
Charles Darwin University.
)))
