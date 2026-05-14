import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data import get_dataset

def train_ensemble(X_train, y_train, X_test, y_test, cancer_type):
    print(f"--- Training models for {cancer_type} cancer ---")
    
    rf = RandomForestClassifier(random_state=42)
    svm = SVC(probability=True, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    
    rf.fit(X_train, y_train)
    svm.fit(X_train, y_train)
    gb.fit(X_train, y_train)
    
    print(f"RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")
    print(f"SVM Accuracy: {accuracy_score(y_test, svm.predict(X_test)):.4f}")
    print(f"GB Accuracy: {accuracy_score(y_test, gb.predict(X_test)):.4f}")
    
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("gb", gb)],
        voting="soft"
    )
    ensemble.fit(X_train, y_train)
    
    ens_pred = ensemble.predict(X_test)
    print(f"Ensemble Accuracy: {accuracy_score(y_test, ens_pred):.4f}\n")
    
    return ensemble

def train_and_save_all_models():
    # We will save the models in the root directory.
    # The existing files are cancer_model.pkl and scaler.pkl. We will preserve those names for breast cancer 
    # to maintain backward compatibility with existing code.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    cancer_types = {
        "breast": ("cancer_model.pkl", "scaler.pkl"),
        "cervical": ("cervical_model.pkl", "cervical_scaler.pkl")
    }
    
    for ctype, (model_filename, scaler_filename) in cancer_types.items():
        X_train, X_test, y_train, y_test, scaler, _, _ = get_dataset(ctype)
        ensemble_model = train_ensemble(X_train, y_train, X_test, y_test, ctype)
        
        # Save model
        model_path = os.path.join(root_dir, model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(ensemble_model, f)
            
        # Save scaler
        scaler_path = os.path.join(root_dir, scaler_filename)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
            
        print(f"Saved {ctype} model to {model_filename} and scaler to {scaler_filename}")

if __name__ == "__main__":
    train_and_save_all_models()
