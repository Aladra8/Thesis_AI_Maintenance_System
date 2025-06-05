# Elevator Predictive Maintenance with AI

This project implements predictive maintenance for elevators using sensor data and machine learning. It replicates the benchmark approach in Liu et al. (2022) and proposes an improved neural network model to detect faults.


## Key Features

- Data exploration & preprocessing
- Neural network model for binary classification
- Evaluation (Precision, Recall, F1, Confusion Matrix, ROC)
- PDF report generator

## Dataset Source

Kaggle Elevator Predictive Maintenance Dataset  
[Vibration-based failure classification dataset](https://www.kaggle.com/datasets/ninorapicavoli/elevator-predictive-maintenance)

 ## Processed Dataset Summary

**Source:** Kaggle - Elevator Predictive Maintenance Dataset  
**Samples:** 112,001  
**Faults:** 5,598 (~5%)  
**Features:** `revolutions`, `humidity`, `vibration`, `x1` to `x5` (engineered)

## Models Implemented
 ----------------------------------------------
| Model                  | Accuracy | F1-Score |
|------------------------|----------|----------|
| Neural Network         | 99.1%    | 0.99     |
| Random Forest          | 97.9%    | 0.96     |
| Support Vector Machine | 96.8%    | 0.95     |

## VS Benchmark

Our results are compared with **Liu et al. (2022)** which used Autoencoder + RF on a simulated dataset and achieved:
- **Accuracy:** 97.1%
- **F1-Score:** 0.94

## Report

- Benchmark comparison table
- Evaluation metrics
- Plots: ROC Curve, Confusion Matrix
- PDF report: `outputs/evaluation/final_model_report.pdf`

##  Citation

Liu et al., 2022 — _“Fault detection in elevators using vibration-based AI classification.”_

---

## How to Run

1. Clone the repo  
2. Create virtual environment  
3. Install dependencies  
4. Run:

```bash
python src/preprocess_and_explore.py
python src/train_model.py
python src/evaluate_model.py
python src/generate_report.py



## Next Steps

## Author
Baba Drammeh — Master's Thesis, University of Padova & Pisa