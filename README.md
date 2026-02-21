# ECON 626: Machine Learning for Economists (University of Waterloo)

# Prediction Competition 4 (PC4)  
## Random Forest Regression for Used-Car Price Prediction

This repository contains my solution to **Prediction Competition 4 (PC4)** for ECON 626 (Machine Learning).  

The objective was to predict the **natural log of used-car prices** using structured vehicle attributes and generate out-of-sample predictions for a held-out test set.

---

## Problem Description

Given a training dataset of used vehicles with observed prices, the goal is to predict:

\[
\log(\text{price})
\]

for each vehicle in a separate test dataset.

This is a supervised regression problem with structured tabular data including:

- Vehicle characteristics (make, model, trim)
- Engine and drivetrain specifications
- Fuel economy measures
- Physical dimensions
- Categorical descriptors

---

## Methodology

### 1. Preprocessing Pipeline

A reproducible preprocessing pipeline was implemented using the **tidymodels** framework.

Key steps:

- Conversion of mixed-format numeric fields using `parse_number()`
- Median imputation for missing numerical values
- Replacement of missing categorical values with an explicit `"Unknown"` level
- One-hot encoding of categorical predictors
- Removal of zero-variance predictors

All preprocessing steps are applied consistently to both training and test data via a unified workflow.

---

### 2. Feature Engineering

Several domain-informed features were constructed:

- **Vehicle age**: difference between listing year and model year  
- **Power density**: horsepower per liter of engine displacement  
- **Average MPG**: mean of city and highway fuel economy  
- **Vehicle footprint**: length × width  

These features aim to capture structural drivers of price variation.

---

### 3. Model Specification

A **Random Forest regression model** was trained using:

- 700 trees  
- \( m = \lfloor \sqrt{p} \rfloor \) predictors sampled per split  
- Minimum node size = 10  

The model was implemented using the `ranger` engine within a `tidymodels` workflow.

---

## Evaluation

Model performance on the training data was assessed using:

- **R²**
- Diagnostic plot of actual vs. predicted values (y vs ŷ)

A 45-degree reference line is included in the diagnostic plot to assess calibration.

---

## Reproducibility

The script:

- Implements preprocessing and modeling within a single workflow object
- Generates the required submission file format
- Produces a diagnostic plot
- Avoids hard-coded data transformations outside the pipeline

Datasets are not included in this repository.

---

## Project Structure

```
├── .gitignore
├── AIEcon_W2026_Prediction_Competition_4_v1.pdf
├── README.md
├── pc4.csv
├── pc4.pdf
└── pc4_code.R
```
