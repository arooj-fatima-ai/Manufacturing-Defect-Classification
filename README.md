# Manufacturing Defect Classification

## Overview
An ML-based classification project that predicts whether a manufactured unit is defective or non-defective using manufacturing process data.

## Objective
To investigate manufacturing process measurements and use machine learning to identify patterns associated with defective outcomes.

## Dataset
A simulated manufacturing-process dataset was created for this educational project. It contains 6,000 manufacturing units with process measurements and a defect-status target.

## Features
- Production Line
- Process Temperature
- Process Pressure
- Cycle Time
- Material Thickness
- Machine Vibration Level
- Operator Experience
- Inspection Score

## Target
is_defective:
- 1 = Defective
- 0 = Non-Defective

## Machine Learning Models
- Logistic Regression
- Decision Tree

The models are compared using classification evaluation metrics, and the final model is selected based on the actual results.

## Application
A Streamlit web application allows users to enter manufacturing process readings and receive a defect prediction with a risk/probability indication.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## How to Run

Install the required libraries:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

## Output
The application predicts whether the entered manufacturing observation is:

- Defective
- Non-Defective