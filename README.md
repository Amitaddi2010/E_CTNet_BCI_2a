# E-CTNet: Enhanced Convolutional Transformer Network for EEG Motor Imagery Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-BCI__Competition__IV__2a-008080?style=for-the-badge)](http://www.bbci.de/competition/iv/)
[![Accuracy](https://img.shields.io/badge/Mean__Accuracy-85.11%25-brightgreen?style=for-the-badge)](https://github.com/Amitaddi2010/E_CTNet_BCI_2a)
[![Cohen's Kappa](https://img.shields.io/badge/Mean__Kappa-0.8014-blue?style=for-the-badge)](https://github.com/Amitaddi2010/E_CTNet_BCI_2a)

---

## 📌 Project Overview

This repository provides an end-to-end deep learning framework for 4-class Motor Imagery (MI) EEG signal classification on the **BCI Competition IV Dataset 2a**. The framework integrates **Enhanced Convolutional Transformer Networks (E-CTNet)**, **Euclidean Alignment (EA)**, and **Segmentation & Reconstruction (S&R) Data Augmentation** to achieve state-of-the-art motor imagery decoding performance.

The 4-class motor imagery tasks evaluated in this project include:
1. **Left Hand** 🤚
2. **Right Hand** ✋
3. **Both Feet** 🦶
4. **Tongue** 👅

---

## 🚀 Key Achievements & Results

Our optimized **E-CTNet** pipeline achieved a milestone **85.11% Mean Accuracy** ($\mathbf{\kappa = 0.8014}$) across all 9 subjects on the official BCI Competition IV 2a Evaluation Session E test set.

### 📊 Subject-Wise Benchmark Comparison

| Subject | CTNet Baseline Acc | CTNet + S&R Aug ($N=15$) | Soft Voting Ensemble | CTNet Improved (5-Fold CV) | **E-CTNet Target Optimized** | Best Cohen's Kappa ($\kappa$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subject 1** | 66.67% | 75.69% | 71.53% | 88.57% | **91.67%** | **0.8889** |
| **Subject 2** | 54.17% | 55.56% | 49.31% | 62.49% | **71.53%** | **0.6204** |
| **Subject 3** | 74.31% | 87.50% | 82.64% | 94.09% | **95.83%** | **0.9444** |
| **Subject 4** | 42.36% | 49.31% | 38.89% | 62.17% | **74.31%** | **0.6574** |
| **Subject 5** | 53.47% | 56.25% | 60.42% | 76.09% | **82.64%** | **0.7685** |
| **Subject 6** | 40.28% | 41.67% | 43.06% | 63.23% | **75.00%** | **0.6667** |
| **Subject 7** | 55.56% | 84.03% | 84.03% | 96.19% | **97.22%** | **0.9630** |
| **Subject 8** | 76.39% | 79.86% | 87.50% | 88.54% | **91.67%** | **0.8889** |
| **Subject 9** | 52.78% | 66.67% | 73.61% | 84.74% | **86.11%** | **0.8148** |
| **Mean ± Std** | **57.33% ± 12.0%** | **66.28% ± 15.5%** | **65.66% ± 17.4%** | **79.57% ± 13.12%** | **85.11% ± 8.85%** | **0.8014 ± 0.1180** |

#### Key Highlights:
- **Top Individual Accuracy**: **97.22%** on Subject 7 ($\kappa = 0.9630$) and **95.83%** on Subject 3 ($\kappa = 0.9444$).
- **Consistency**: Standard deviation across subjects reduced from **12.0%** in baseline to **8.85%** in the optimized E-CTNet pipeline.
- **Milestone Surpassed**: Exceeded the target baseline benchmark of 85.0% mean accuracy.

---

## 🧠 Model Architectures

The project supports multiple state-of-the-art EEG neural architectures configurable via [`config.yaml`](file:///f:/Amit/BCI_Classification/config.yaml):

```
       ┌─────────────────────────────────────────────────────────┐
       │                   Raw EEG Input (22 x 1000)             │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │     1D Temporal Conv (Kernel 1x64, F1=16) + BatchNorm   │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Spatial Depthwise Conv (22x1, D=2, MaxNorm=1.0) + ELU  │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │      Average Pooling (1x8) + Pointwise Conv (1x16)      │
       └────────────────────────────┬────────────────────────────┘
                                    │
                        ┌───────────┴───────────┐
                        ▼                       ▼
            ┌───────────────────────┐ ┌────────────────────┐
            │ CNN Feature Map Path  │ │  Positional Enc.   │
            └───────────┬───────────┘ └─────────┬──────────┘
                        │                       │
                        │                       ▼
                        │             ┌────────────────────┐
                        │             │ Transformer Encoder│
                        │             │ (Multi-Head Attn)  │
                        │             └─────────┬──────────┘
                        │                       │
                        └───────────┬───────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │         Residual Skip Connection (CNN + Trans)          │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │        Linear Classification Head (4 Classes)           │
       └─────────────────────────────────────────────────────────┘
```

### 1. **CTNet (Enhanced Convolutional Transformer)**
- **Temporal Patch Extraction**: 1D temporal convolutions ($1 \times 64$ kernel at 250 Hz) capture bandpass spectral features.
- **Spatial Depthwise Convolution**: Spatial filtering across 22 EEG electrodes with weight max-norm constraint ($\max = 1.0$) to stabilize gradients.
- **Transformer Encoder**: Multi-Head Self-Attention (MHSA) with sinusoidal positional encoding models temporal context.
- **Residual Connection**: Adds direct CNN feature maps to Transformer outputs ($F_{\text{out}} = F_{\text{cnn}} + F_{\text{trans}}$), preventing information degradation in deeper layers.

### 2. **EEGConformerATCNet**
- Hybrid multi-branch network featuring multi-scale temporal convolutions ($k \in [7, 15, 31, 63]$), spatial depthwise convs, Multi-Head Attention, dilated TCN blocks, and **Dynamic Gated Feature Fusion**.

### 3. **ShallowFBCSPNet & EEGNet**
- **ShallowFBCSPNet**: Inspired by Filter Bank Common Spatial Patterns with bandpass convolutions, squared activation, log-power average pooling.
- **EEGNet**: Compact architecture using depthwise and separable convolutions for low-parameter EEG decoding.

---

## 🛠️ Data Preprocessing & Augmentation Pipeline

1. **Bandpass Filtering**: $4.0\text{ Hz} - 38.0\text{ Hz}$ Chebyshev/Butterworth zero-phase bandpass filter.
2. **Epoch Extraction**: 4.0-second time window ($t = 0.0\text{s}$ to $4.0\text{s}$ relative to cue onset), yielding 1,000 temporal samples per trial.
3. **Euclidean Alignment (EA)**: Aligning subject covariance matrices to the identity matrix $I$:
   $$R = \frac{1}{N} \sum_{i=1}^N X_i X_i^T, \quad \tilde{X}_i = R^{-1/2} X_i$$
   This significantly reduces intra-subject and session-to-session variability.
4. **Segmentation & Reconstruction (S&R)**: Sub-dividing trial samples into contiguous temporal segments and shuffling across identical class trials to augment trial diversity ($N = 15$).
5. **Mixup & Cosine Annealing**: Dynamic mixup regularization during training with warm-restart learning rate scheduling.

---

## 📁 Repository Structure

```
BCI_Classification/
├── config.yaml                    # Master YAML configuration file
├── dataset.py                     # Data loader, bandpass filtering, EA, S&R augmentation
├── models.py                      # CTNet, EEGConformerATCNet, ShallowFBCSPNet, EEGNet
├── train.py                       # Training loop, loss functions, learning rate schedulers
├── evaluate.py                    # Evaluation & metrics calculation (Accuracy, Cohen's Kappa)
├── run_subject_tuned_cv.py        # Subject-specific model training & cross-validation
├── run_session_e_evaluation.py    # Official Session E test set evaluation pipeline
├── run_optimized_cv.py            # Optimized cross-validation routine
├── run_per_subject_grid_search.py # Subject-tuned hyperparameter grid search
├── visualize.py                   # t-SNE feature visualization & confusion matrices generator
├── check_urls.py                  # Dataset download link verifier
├── download_bnci_mat.py           # BNCI2014_001 dataset fetcher script
├── download_bits.ps1              # BITS PowerShell download utility
├── requirements.txt               # Dependencies list
├── results/                       # Markdown tables, TeX tables, and saved figures
│   ├── results_table.md           # Markdown performance summary
│   ├── results_table.tex          # LaTeX formatted table
│   └── figures/                   # Confusion matrices, loss curves, t-SNE plots
└── README.md                      # Project documentation
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+ with CUDA support
- Git

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/Amitaddi2010/E_CTNet_BCI_2a.git
cd E_CTNet_BCI_2a
pip install -r requirements.txt
```

### 3. Dataset Setup
The framework reads GDF/MAT files from the `BCICIV_2a_gdf (1)` directory. If the dataset is not present, you can automatically download it using MOABB / BNCI fetcher:

```bash
python download_bnci_mat.py
```

---

## 🏃 Usage Guide

### 1. Subject-Tuned Model Training
Train subject-specific E-CTNet models with Euclidean Alignment and S&R Augmentation:

```bash
python run_subject_tuned_cv.py
```

### 2. Session E Evaluation
Evaluate trained subject models on the official unseen Session E test set:

```bash
python run_session_e_evaluation.py
```

### 3. Subject Hyperparameter Grid Search
Run per-subject grid search to find optimal learning rate, weight decay, and dropout settings:

```bash
python run_per_subject_grid_search.py
```

### 4. Feature & Performance Visualization
Generate t-SNE feature projections and per-subject confusion matrices:

```bash
python visualize.py
```

Generated plots will be saved in `results/figures/`.

---

## 📖 References & Citations

- **CTNet Architecture**: *A Convolutional Transformer Network for EEG-Based Motor Imagery Classification*, Nature Scientific Reports, 2024. [GitHub](https://github.com/snailpt/CTNet)
- **EEGNet**: Lawhern et al., *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*, Journal of Neural Engineering, 2018.
- **ShallowFBCSPNet**: Schirrmeister et al., *Deep learning with convolutional neural networks for EEG decoding and visualization*, Human Brain Mapping, 2017.
- **Euclidean Alignment**: He et al., *Transfer learning for brain-computer interfaces: A Euclidean space data alignment approach*, IEEE TNSRE, 2019.
- **BCI Competition IV Dataset 2a**: Brunner et al., *BCI Competition 2008 – Graz dataset A*, Institute for Knowledge Discovery, Graz University of Technology, 2008.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
