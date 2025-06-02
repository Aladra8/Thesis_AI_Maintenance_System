# Elevator Predictive Maintenance with AI

This project implements predictive maintenance for elevators using sensor data and machine learning. It replicates the benchmark approach in Liu et al. (2022) and proposes an improved neural network model to detect faults.


## Key Features

- Data exploration & preprocessing
- Neural network model for binary classification
- Evaluation (Precision, Recall, F1, Confusion Matrix, ROC)
- PDF report generator

## Dataset

Kaggle Elevator Predictive Maintenance Dataset  
[Vibration-based failure classification dataset](https://www.kaggle.com/datasets/ninorapicavoli/elevator-predictive-maintenance)

## Results Summary

- Accuracy: **99.08%**
- Recall (Faults): **1.00**
- Precision (Faults): **0.85**
- AUC: **0.9997**

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
Benchmark comparison

SVM and Random Forest experiments

Final thesis write-up integration

## Author
Baba Drammeh — Master's Thesis, University of Padova & Pisa