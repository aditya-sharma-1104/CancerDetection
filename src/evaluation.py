import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from data import get_dataset
from router import CancerPredictionRouter

def evaluate_models():
    """Evaluates the models and generates a comprehensive report."""
    
    cancer_types = ["breast", "cervical"]
    router = CancerPredictionRouter()
    
    evaluation_results = []
    
    for ctype in cancer_types:
        print(f"\n{'='*40}")
        print(f"Evaluating {ctype.capitalize()} Cancer Model")
        print(f"{'='*40}")
        
        # Load dataset
        _, X_test_scaled, _, y_test, _, _, target_names = get_dataset(ctype)
        
        # We will use the models loaded in the router to predict
        model = router.models[ctype]
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1] # Probability of positive class
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan # In case of single class in y_true, though unlikely
            
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if not np.isnan(auc):
            print(f"ROC-AUC:   {auc:.4f}")
            
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        evaluation_results.append({
            "Cancer Type": ctype.capitalize(),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })
        
    print(f"\n{'='*40}")
    print("Summary of All Models")
    print(f"{'='*40}")
    summary_df = pd.DataFrame(evaluation_results)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_models()
