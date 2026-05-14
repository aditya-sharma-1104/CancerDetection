import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import urllib.request

def load_and_preprocess_breast_cancer():
    """Loads and preprocesses the Wisconsin Breast Cancer dataset."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    # Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, data.feature_names, data.target_names

def load_and_preprocess_cervical_cancer():
    """Loads and preprocesses the UCI Cervical Cancer dataset."""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'
    dataset_path = 'cervical_cancer.csv'
    
    if not os.path.exists(dataset_path):
        print("Downloading Cervical Cancer dataset...")
        urllib.request.urlretrieve(url, dataset_path)
        
    df = pd.read_csv(dataset_path, na_values='?')
    
    # The columns 'STDs: Time since first diagnosis' and 'STDs: Time since last diagnosis'
    # have >90% missing values, drop them
    df = df.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
    
    # Target variables are Hinselmann, Schiller, Citology, Biopsy
    # We will use 'Biopsy' as the primary target for cancer classification
    target_col = 'Biopsy'
    
    # Drop other target columns to prevent leakage
    df = df.drop(columns=['Hinselmann', 'Schiller', 'Citology'])
    
    # Impute missing values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # Preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    target_names = np.array(['Healthy', 'Cancer'])
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns, target_names

def get_dataset(cancer_type="breast"):
    if cancer_type == "breast":
        return load_and_preprocess_breast_cancer()
    elif cancer_type == "cervical":
        return load_and_preprocess_cervical_cancer()
    else:
        raise ValueError(f"Unknown cancer type: {cancer_type}")
