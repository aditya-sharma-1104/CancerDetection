"""
Retrain and save all models for the current environment.
Run this once from the project root: python backend/train_models.py
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. BREAST CANCER (Wisconsin dataset via sklearn)
# ─────────────────────────────────────────────
print("Training Breast Cancer model...")
data = load_breast_cancer()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

rf  = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)
gb  = GradientBoostingClassifier(random_state=42)
ensemble = VotingClassifier([("rf", rf), ("svm", svm), ("gb", gb)], voting="soft")
ensemble.fit(X_train_s, y_train)

from sklearn.metrics import accuracy_score
print(f"  Breast accuracy: {accuracy_score(y_test, ensemble.predict(X_test_s)):.4f}")

with open(os.path.join(MODELS_DIR, "cancer_model.pkl"), "wb") as f: pickle.dump(ensemble, f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f: pickle.dump(scaler, f)
breast_columns = list(data.feature_names)
with open(os.path.join(MODELS_DIR, "breast_columns.pkl"), "wb") as f: pickle.dump(breast_columns, f)
print("  Saved: cancer_model.pkl, scaler.pkl, breast_columns.pkl")

# ─────────────────────────────────────────────
# 2. LUNG / CERVICAL CANCER (UCI Cervical dataset)
# ─────────────────────────────────────────────
print("Training Lung/Cervical Cancer model...")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cervical_cancer.csv")
df = pd.read_csv(DATA_PATH, na_values="?")

# Drop high-NaN columns and target alternatives
drop_cols = [
    "STDs: Time since first diagnosis", "STDs: Time since last diagnosis",
    "Hinselmann", "Schiller", "Citology"
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
df.dropna(inplace=True)

X2 = df.drop(columns=["Biopsy"])
y2 = df["Biopsy"].astype(int)

lung_columns = list(X2.columns)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s  = scaler2.transform(X2_test)

rf2  = RandomForestClassifier(n_estimators=100, random_state=42)
svm2 = SVC(probability=True, random_state=42)
gb2  = GradientBoostingClassifier(random_state=42)
ensemble2 = VotingClassifier([("rf", rf2), ("svm", svm2), ("gb", gb2)], voting="soft")
ensemble2.fit(X2_train_s, y2_train)

print(f"  Lung accuracy: {accuracy_score(y2_test, ensemble2.predict(X2_test_s)):.4f}")

with open(os.path.join(MODELS_DIR, "lung_model.pkl"), "wb") as f: pickle.dump(ensemble2, f)
with open(os.path.join(MODELS_DIR, "lung_scaler.pkl"), "wb") as f: pickle.dump(scaler2, f)
with open(os.path.join(MODELS_DIR, "lung_columns.pkl"), "wb") as f: pickle.dump(lung_columns, f)
print("  Saved: lung_model.pkl, lung_scaler.pkl, lung_columns.pkl")

print("\n✅ All models trained and saved successfully!")
