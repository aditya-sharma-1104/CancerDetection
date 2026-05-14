import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, roc_curve, auc

# Loading the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(probability=True, random_state=42)
gb = GradientBoostingClassifier(random_state=42)

rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
gb.fit(X_train_scaled, y_train)

ensemble = VotingClassifier(estimators=[("rf", rf), ("svm", svm), ("gb", gb)], voting="soft")
ensemble.fit(X_train_scaled, y_train)

# Accuracy
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
svm_acc = accuracy_score(y_test, svm.predict(X_test_scaled))
gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))
ens_acc = accuracy_score(y_test, ensemble.predict(X_test_scaled))

print(f"RF Accuracy: {rf_acc:.4f}")
print(f"SVM Accuracy: {svm_acc:.4f}")
print(f"GB Accuracy: {gb_acc:.4f}")
print(f"Ensemble Accuracy: {ens_acc:.4f}")

# Save Confusion Matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(ensemble, X_test_scaled, y_test, display_labels=data.target_names, cmap='Blues')
plt.title("Confusion Matrix - Ensemble Model")
plt.savefig("confusion.png")
plt.close()

# Save ROC Curve
rf_probs = rf.predict_proba(X_test_scaled)[:, 1]
svm_probs = svm.predict_proba(X_test_scaled)[:, 1]
gb_probs = gb.predict_proba(X_test_scaled)[:, 1]
ensemble_probs = ensemble.predict_proba(X_test_scaled)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
ens_fpr, ens_tpr, _ = roc_curve(y_test, ensemble_probs)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {auc(rf_fpr, rf_tpr):.2f})")
plt.plot(svm_fpr, svm_tpr, label=f"SVM (AUC = {auc(svm_fpr, svm_tpr):.2f})")
plt.plot(gb_fpr, gb_tpr, label=f"Gradient Boosting (AUC = {auc(gb_fpr, gb_tpr):.2f})")
plt.plot(ens_fpr, ens_tpr, label=f"Ensemble (AUC = {auc(ens_fpr, ens_tpr):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

# Save Feature Importance
importance = rf.feature_importances_
features = X.columns
imp_df = pd.DataFrame({"Feature": features, "Importance": importance})
imp_df = imp_df.sort_values(by="Importance", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=imp_df, palette="viridis")
plt.title("Top 10 Important Features (Random Forest)")
plt.tight_layout()
plt.savefig("features.png")
plt.close()
