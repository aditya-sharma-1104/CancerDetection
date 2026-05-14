import pickle
import os
import numpy as np

class CancerPredictionRouter:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.target_names = {
            "breast": ["Malignant", "Benign"], # Based on original Wisconsin breast cancer (0: Malignant, 1: Benign)
            "cervical": ["Healthy", "Cancer"]  # 0: Healthy, 1: Cancer
        }
        self._load_models()

    def _load_models(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        cancer_types = {
            "breast": ("cancer_model.pkl", "scaler.pkl"),
            "cervical": ("cervical_model.pkl", "cervical_scaler.pkl")
        }
        
        for ctype, (model_file, scaler_file) in cancer_types.items():
            model_path = os.path.join(root_dir, model_file)
            scaler_path = os.path.join(root_dir, scaler_file)
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, "rb") as f:
                    self.models[ctype] = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scalers[ctype] = pickle.load(f)
            else:
                print(f"Warning: Models for {ctype} cancer not found. Please run model.py to train them.")

    def predict(self, cancer_type, features):
        """
        Routes the prediction to the correct model.
        
        Args:
            cancer_type (str): The type of cancer ('breast', 'cervical').
            features (list or np.array): The feature values for the patient.
            
        Returns:
            dict: standardized output with prediction class and risk score.
        """
        if cancer_type not in self.models:
            raise ValueError(f"Model for {cancer_type} cancer is not loaded or unsupported.")
            
        scaler = self.scalers[cancer_type]
        model = self.models[cancer_type]
        
        # Ensure features is a 2D array
        features_array = np.array(features).reshape(1, -1)
        
        # Scale features
        scaled_features = scaler.transform(features_array)
        
        # Predict class
        prediction = model.predict(scaled_features)[0]
        predicted_class_name = self.target_names[cancer_type][prediction]
        
        # Predict probability (Risk Score)
        # Soft voting ensemble returns probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        
        # Risk score is typically the probability of the positive/cancer class
        # For breast cancer, 0 is Malignant (Cancer). For cervical, 1 is Cancer.
        if cancer_type == "breast":
            risk_score = probabilities[0] # Probability of Malignant
        else:
            risk_score = probabilities[1] # Probability of Cancer
            
        return {
            "cancer_type": cancer_type,
            "prediction": int(prediction),
            "diagnosis": predicted_class_name,
            "risk_score": float(risk_score),
            "probabilities": probabilities.tolist()
        }

if __name__ == "__main__":
    # Test the router locally
    router = CancerPredictionRouter()
    
    # Dummy features for testing (just zeros with correct lengths)
    breast_dummy = np.zeros(30) # Breast cancer has 30 features
    cervical_dummy = np.zeros(30) # Cervical cancer also has 30 features after preprocessing
    
    print("\n--- Testing Routing Logic ---")
    try:
        print("Breast Cancer Prediction:")
        print(router.predict("breast", breast_dummy))
        
        print("\nCervical Cancer Prediction:")
        print(router.predict("cervical", cervical_dummy))
    except Exception as e:
        print(f"Error during testing: {e}")
