

# Early Churn Prediction During Onboarding

## Overview

This repository contains the code developed for the MSc dissertation **“Understanding and Predicting Customer Churn During Onboarding Using Behavioural Data and Interpretable Machine Learning.”**
The project focuses on predicting **early churn during the onboarding phase** of a fintech mobile application using behavioural interaction data.

Non-completion of onboarding is treated as an early form of churn, allowing the study to focus on **early user decision-making and friction**.


## Project Structure

dataset/          # Raw and cleaned datasets
notebooks/     # data cleaning, feature engineering and data modelling
src/       # preprocessedand modelled files
README.md


## Methodology Summary

* **Exploratory Data Analysis (EDA):**
  Analysed onboarding behaviour, class imbalance, and interaction patterns.

* **Preprocessing & Feature Engineering:**
  Cleaned data and engineered behavioural features such as time spent, repetition, delays, and progression speed to capture user hesitation and disengagement.

* **Modeling:**

  * Logistic Regression (baseline, interpretable)
  * Random Forest (non-linear interactions)
  * XGBoost (advanced behavioural patterns)

* **Evaluation:**
  Models were evaluated using ROC-AUC, precision, recall, F1-score, and probability calibration.

* **Explainability:**
  SHAP was used to provide global and local explanations, enabling transparent interpretation of churn drivers and supporting actionable insights.


## Ethical Focus

Explainability was prioritised to ensure responsible use of predictions, supporting friction reduction rather than coercive conversion strategies.


## Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Jupyter Notebook


## Academic Context

MSc in Artificial Intelligence and Data Science
CITY College – University of York Europe Campus


