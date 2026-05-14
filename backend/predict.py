import pickle
import pandas as pd
import os

# Get the absolute path to the models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load models
breast_model = pickle.load(open(os.path.join(MODELS_DIR, "cancer_model.pkl"), "rb"))
breast_scaler = pickle.load(open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb"))
breast_columns = pickle.load(open(os.path.join(MODELS_DIR, "breast_columns.pkl"), "rb"))

lung_model = pickle.load(open(os.path.join(MODELS_DIR, "lung_model.pkl"), "rb"))
lung_scaler = pickle.load(open(os.path.join(MODELS_DIR, "lung_scaler.pkl"), "rb"))
lung_columns = pickle.load(open(os.path.join(MODELS_DIR, "lung_columns.pkl"), "rb"))


def predict_cancer(cancer_type, data):

    if cancer_type == "breast":

        df = pd.DataFrame([data], columns=breast_columns)
        df_scaled = breast_scaler.transform(df)

        pred = breast_model.predict(df_scaled)[0]
        prob = breast_model.predict_proba(df_scaled)[0][1]

        return {
            "type": "breast",
            "result": "Benign" if pred == 1 else "Malignant",
            "probability": float(prob)
        }

    elif cancer_type == "lung":

        df = pd.DataFrame([data], columns=lung_columns)
        df_scaled = lung_scaler.transform(df)

        pred = lung_model.predict(df_scaled)[0]
        prob = lung_model.predict_proba(df_scaled)[0][1]

        return {
            "type": "lung",
            "result": "Cancer" if pred == 1 else "No Cancer",
            "probability": float(prob)
        }

    return {"error": "Invalid cancer type"}
