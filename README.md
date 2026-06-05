# Early Breast Cancer Detection Using Ensemble Machine Learning :

## Overview

This project develops a machine learning system to classify breast tumors as benign or malignant using the Breast Cancer Wisconsin dataset. Multiple models are trained and combined using an ensemble approach to improve performance.

## Dataset

* Source: UCI Machine Learning Repository
* Samples: 569
* Features: 30
* Classes:

  * 0 → Malignant
  * 1 → Benign


## Models Used

* Support Vector Machine
* Random Forest
* Gradient Boosting
* Voting Classifier (Soft Voting Ensemble)

## Workflow

Load Data → EDA → Preprocessing → Model Training → Ensemble → Evaluation → Prediction


## Results

* Ensemble Accuracy: ~98%
* Cross Validation Accuracy: ~97–98%
* ROC-AUC Score: 0.995
  <img width="681" height="548" alt="image" src="https://github.com/user-attachments/assets/01df7e7d-5354-4a04-b3ee-b5395c1e0cb2" />


## Evaluation Techniques

* Confusion Matrix
  <img width="597" height="517" alt="image" src="https://github.com/user-attachments/assets/7cc41d3f-b7bd-46c8-b245-cdd2ef8a25ce" />

* Classification Report
* ROC Curve
* Learning Curve
* Feature Importance

## Key Features

* Comparison of multiple models
* Ensemble learning (hard vs soft voting)
* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* Model saving using pickle

## Project Structure

├── notebook.ipynb
├── cancer_model.pkl
├── scaler.pkl
├── README.md

## Installation

pip install numpy pandas matplotlib seaborn scikit-learn

## Usage

Run the notebook to:

* Train and evaluate models
* Visualize performance
* Generate predictions

Sample predictions are included using dataset entries.


## Results Preview


* ROC Curve
 <img width="742" height="527" alt="image" src="https://github.com/user-attachments/assets/d8dc8f18-d9f5-4be0-b799-3028fa1359e0" />

* Learning Curve
  <img width="736" height="544" alt="image" src="https://github.com/user-attachments/assets/e925f583-c3c9-4e9e-829b-6f991f9308b4" />

* Feature Importance Graph
  <img width="1268" height="648" alt="image" src="https://github.com/user-attachments/assets/4bc1715e-a0ea-441e-bdfd-609e646ff0b6" />


## Recent Updates

* Deployed as a web application with a React frontend and Flask backend.
* Added support for Multi-Cancer detection including Cervical Cancer.

## Future Work

* Integrate medical imaging data
* Add explainability methods

## Author

Ankan Kumar Panja

ROC Curve
<img width="742" height="527" alt="image" src="https://github.com/user-attachments/assets/d8dc8f18-d9f5-4be0-b799-3028fa1359e0" />

Learning Curve
<img width="736" height="544" alt="image" src="https://github.com/user-attachments/assets/e925f583-c3c9-4e9e-829b-6f991f9308b4" />

Feature Importance
<img width="1268" height="648" alt="image" src="https://github.com/user-attachments/assets/4bc1715e-a0ea-441e-bdfd-609e646ff0b6" />


