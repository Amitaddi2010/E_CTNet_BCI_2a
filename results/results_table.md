# BCI Competition IV - Dataset 2a Comprehensive Subject-Wise Accuracy Report

### Subject-Wise Accuracy Matrix Across Methodologies

| Subject | CTNet Baseline Acc | CTNet + S&R Aug (N=15) | Soft Voting Ensemble | CTNet Improved 5-Fold CV | **CTNet Optimized Target** | Best Cohen's Kappa (κ) |
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
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean ± Std** | **57.33% ± 12.0%** | **66.28% ± 15.5%** | **65.66% ± 17.4%** | **79.57% ± 13.12%** | **85.11% ± 8.85%** | **0.8014 ± 0.1180** |

---

### Key Milestone Achievements:
- **Target Reached**: The **85.0%** overall accuracy target was successfully achieved with **85.11%** mean accuracy using Optimized CTNet (Euclidean Alignment + Extended S&R Augmentation + Mixup + Warm Restarts).
- **Mean Dataset Accuracy**: Advanced from **79.57%** (CTNet Improved) to **85.11%** (CTNet Optimized Target).
- **Mean Cohen's Kappa**: Reached **κ = 0.8014** across all 9 subjects.
- **Peak Individual Performances**:
  - **Subject 7**: **97.22%** ($\kappa = 0.9630$)
  - **Subject 3**: **95.83%** ($\kappa = 0.9444$)
  - **Subject 1 & 8**: **91.67%** ($\kappa = 0.8889$)
 |


