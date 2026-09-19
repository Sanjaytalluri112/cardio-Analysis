# Cardio Risk Analysis

A machine learning-based cardiovascular risk assessment project that predicts the likelihood of heart disease from clinical patient data.

## Overview

This project uses clinical and demographic features to train and evaluate multiple machine learning models for cardiovascular risk prediction.

The project includes:

- Data preprocessing and exploratory analysis
- Feature preparation and scaling
- Multiple machine learning models
- Model evaluation and comparison
- A Flask-based web application for prediction
- Saved trained models and preprocessing components
- Visualization and evaluation plots

## Models

The project uses the following machine learning models:

- Logistic Regression
- Random Forest
- XGBoost

The trained models are stored in the `models/` directory.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- Matplotlib
- Jupyter Notebook

## Project Structure

```text
cardioRisk_assessment/
├── data/
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── heart_app/
│   └── app.py
│
├── models/
│   ├── feature_names.pkl
│   ├── imputer.pkl
│   ├── logistic_model.pkl
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   ├── scaler_cols.pkl
│   └── xgb_model.pkl
│
├── plots/
│   ├── confusion_matrices.png
│   ├── correlation_heatmap.png
│   ├── feature_dists.png
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── roc_curves.png
│   └── target_correlation.png
│
├── templates/
│   └── index.html
│
├── heart_disease_uci.csv
├── preprocessing_eda.ipynb
├── model_training.ipynb
├── model_evaluation.ipynb
└── README.md
```

## Workflow

```text
Clinical Dataset
       ↓
Data Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Preparation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Cardiovascular Risk Prediction
       ↓
Flask Web Application
```

## Running Locally

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cardio-Analysis
```

### 2. Create a Virtual Environment

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost flask matplotlib jupyter
```

### 4. Run the Flask Application

```bash
python heart_app/app.py
```

Open the application in your browser:

```text
http://127.0.0.1:5000
```

## Evaluation

The project includes visualizations for model evaluation and data analysis, including:

- ROC curves
- Confusion matrices
- Feature importance
- Model comparison
- Correlation analysis
- Feature distributions


