# An AOA-Optimized Multilayer Perceptron Neural Network
# For CVD Prediction Across Multi-Cohort Datasets

[![Python](https://img.shields.io/badge/Python-3.12-blue)]()
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)]()
[![Course](https://img.shields.io/badge/Course-Neural%20Network-green)]()
[![Institution](https://img.shields.io/badge/Institution-MIST-red)]()

---

## Overview

Cardiovascular disease remains the leading cause of 
mortality worldwide. This project implements and extends 
a reference AOA-MLPNN framework for automated CVD 
prediction by addressing three practical challenges 
that the original work left unresolved: missing clinical 
values, class imbalance, and single-dataset evaluation.

Three metaheuristic optimizers are compared as feature 
selectors for MLPNN classification across three UCI 
benchmark datasets under identical experimental conditions.

---

## Reference Paper

> Alghamdi, F.A. et al. (2024). "Multilayer Perceptron 
> Neural Network with Arithmetic Optimization 
> Algorithm-Based Feature Selection for Cardiovascular 
> Disease Prediction." Machine Learning and Knowledge 
> Extraction, 6(2), 987-1008.
> doi: 10.3390/make6020046

---

## Our Extensions Over Reference Paper

| Extension | Detail |
|---|---|
| Multi-dataset validation | Cleveland + Hungarian + Statlog |
| KNN imputation | Enables previously excluded Hungarian dataset |
| SMOTE balancing | Resolves class imbalance — recall +6.48% |
| WOA comparison | Whale Optimization Algorithm added |
| GWO comparison | Grey Wolf Optimizer added |

---

## Datasets

| Dataset | Patients | Features | Missing | Imputation |
|---|---|---|---|---|
| Cleveland | 303 | 13 | 6 (2.0%) | Mean |
| Hungarian | 295 | 13 | 795 (20.7%) | KNN (k=5) |
| Statlog | 270 | 13 | 0 (0.0%) | None |

All datasets sourced from:
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

---

## Methodology
Raw Data
↓
Adaptive Preprocessing
(Mean/KNN Imputation + MinMax Normalization)
↓
Train/Test Split (70/30) + SMOTE Balancing
↓
Metaheuristic Feature Selection
(AOA / WOA / GWO)
↓
MLPNN Classification
↓
Performance Evaluation (7 metrics)

---

## Results

### Best Results Per Dataset

| Dataset | Best Optimizer | Accuracy | Recall | AROC |
|---|---|---|---|---|
| Cleveland | AOA | 87.91% | 90.48% | 0.903 |
| Hungarian | AOA/WOA/GWO | 85.39% | 81.25% | 0.868 |
| Statlog | AOA | 88.89% | 91.67% | 0.931 |

### Average Across All Datasets

| Optimizer | Avg Accuracy | Avg AROC |
|---|---|---|
| AOA | 87.40% | 0.901 |
| WOA | 85.80% | 0.892 |
| GWO | 85.80% | 0.892 |

### Comparison With Reference Paper (Cleveland)

| Metric | Reference | This Work | Change |
|---|---|---|---|
| Accuracy | 88.89% | 87.91% | -0.98% |
| Recall | 84.00% | 90.48% | +6.48% ✅ |
| F1-Score | 86.00% | 87.36% | +1.36% ✅ |
| AROC | 0.840 | 0.903 | +0.063 ✅ |

---

## Project Structure

---

## Implementation Details

### Optimizers

All three optimizers implemented from scratch:

**AOA — Arithmetic Optimization Algorithm**
- Uses division, multiplication, subtraction, addition
- MOA parameter controls exploration/exploitation
- Reference: Abualigah et al. (2021)

**WOA — Whale Optimization Algorithm**
- Models bubble-net hunting of humpback whales
- Three mechanisms: encircle, spiral, search
- Reference: Mirjalili & Lewis (2016)

**GWO — Grey Wolf Optimizer**
- Models wolf pack leadership hierarchy
- Alpha, beta, delta wolves guide search
- Reference: Mirjalili et al. (2014)

### MLPNN Configuration

| Parameter | Value |
|---|---|
| Hidden layers | (50, 25) |
| Activation | Logistic |
| Solver | Adam |
| Learning rate | 0.001 |
| Max epochs | 1000 |

### Optimizer Settings (identical for all 3)

| Parameter | Value |
|---|---|
| Population size | 10 |
| Max iterations | 2 |
| Bounds | [0, 1] |
| Dimensions | 13 |
| Random seed | 4 |

---

## Key Findings

1. **AOA is most robust** — averaging 87.40% accuracy
   across all three datasets

2. **Data quality matters more than optimizer choice**
   — all three optimizers converged identically on
   Hungarian, where 20.7% missingness was the
   binding constraint

3. **SMOTE improved recall by 6.48%** — directly
   reducing missed CVD diagnoses

4. **KNN imputation enabled Hungarian dataset** — 
   previously excluded from comparable studies

5. **All optimizers exceeded reference AROC of 0.840**
   — ranging from 0.868 to 0.931

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open the notebook in Google Colab
2. Run cells sequentially from Shell 1 to Shell 15
3. All dependencies install automatically in Shell 1

### Option 2 — Local Environment

```bash
# Clone repository
git clone [your-repo-url]

# Install dependencies
pip install scikit-learn numpy pandas matplotlib 
pip install imbalanced-learn

# Open notebook
jupyter notebook CVD_Prediction_AOA_MLPNN.ipynb
```

### Important Notes

- Run Shell 15B (convergence curves) separately
  — it takes 20-25 minutes
- All other shells complete within 1-2 minutes each
- Hungarian dataset file must be uploaded manually
  (reprocessed.hungarian.data from UCI)

---

## Requirements
Python 3.12
scikit-learn 1.x
imbalanced-learn 0.x
numpy 1.x
pandas 2.x
matplotlib 3.x
Google Colab (recommended)


---

## Evaluation Metrics

Seven metrics computed for all 9 experiments:

| Metric | Formula |
|---|---|
| Accuracy | (TP+TN) / (TP+TN+FP+FN) |
| Precision | TP / (TP+FP) |
| Recall | TP / (TP+FN) |
| F1-Score | 2 x (Prec x Rec) / (Prec+Rec) |
| Specificity | TN / (TN+FP) |
| MSE | (1/n) Σ(yi - ŷi)² |
| AROC | Area under ROC curve |

---

## Team

| Name | Role |
|---|---|
| S.M. Shazzed Hossain Shajal | [Role] |
| Mahmud Hasan | [Role] |
| Sumaiya Fatima | [Role] |
| Ommay Aiyman | [Role] |

**Course:** Neural Network
**Institution:** Military Institute of Science and Technology (MIST)
**Department:** Computer Science and Engineering
**Supervisor:** Dr. Nusrat Sharmin
**Year:** 2026

---

## References

[1] L. Abualigah et al., "The Arithmetic Optimization 
Algorithm," Comput. Methods Appl. Mech. Eng., 
vol. 376, p. 113609, 2021.

[2] S. Mirjalili and A. Lewis, "The Whale Optimization 
Algorithm," Adv. Eng. Softw., vol. 95, pp. 51-67, 2016.

[3] S. Mirjalili et al., "Grey Wolf Optimizer," 
Adv. Eng. Softw., vol. 69, pp. 46-61, 2014.

[4] N. V. Chawla et al., "SMOTE: Synthetic Minority 
Over-Sampling Technique," JAIR, vol. 16, 
pp. 321-357, 2002.

[5] F. A. Alghamdi et al., "Multilayer Perceptron 
Neural Network with Arithmetic Optimization 
Algorithm-Based Feature Selection for CVD 
Prediction," Mach. Learn. Knowl. Extr., 
vol. 6, no. 2, pp. 987-1008, 2024.

---

## License

This project is submitted as a course requirement
for the Neural Network course at MIST.
For academic use only.

---

## Acknowledgment

This project extends the foundational work of
Alghamdi et al. (2024). We thank our course
supervisor for guidance throughout implementation.
