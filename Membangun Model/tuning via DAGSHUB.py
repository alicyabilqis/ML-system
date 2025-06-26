# modelling_tuning.py via DAGSHUB {FIXX}

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
from dagshub import init
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# === Setup MLflow credentials ===
os.environ["MLFLOW_TRACKING_USERNAME"] = "alicyabilqis"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "a7ea6e35ec95b8929a1bf5a4a49d07b448b53b6d"
init(repo_owner='alicyabilqis', repo_name='MSML', mlflow=True)
mlflow.set_experiment("RF DagsHub Hyperparameter Tuning")

# === Load dataset ===
data = pd.read_csv("Forest cover_preprocessing_dataset.csv")
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Hyperparameter ranges ===
n_estimators_range = [50, 100, 150]
max_depth_range = [10, 20, 30]

best_accuracy = 0
best_params = {}
best_model = None
best_y_pred = None

with mlflow.start_run(run_name="Hyperparameter Tuning"):
    for n in n_estimators_range:
        for d in max_depth_range:
            with mlflow.start_run(nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model = model
                    best_y_pred = y_pred

    # === Log best model manually ===
    joblib.dump(best_model, "best_rf_model.pkl")
    mlflow.log_artifact("best_rf_model.pkl")

    # === Log best metrics ===
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_params(best_params)

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, best_y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig("confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # === Classification Report ===
    cls_report = classification_report(y_test, best_y_pred, output_dict=True)
    cls_df = pd.DataFrame(cls_report).iloc[:-1, :-1].T
    plt.figure(figsize=(10, 6))
    sns.heatmap(cls_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title("Classification Report")
    plt.savefig("classification_report.png")
    plt.close()
    mlflow.log_artifact("classification_report.png")

    print(f"\n✅ Model terbaik hasil tuning: {best_params} - Accuracy: {best_accuracy:.4f}")
    print("✅ Best model, metrics, and visualizations logged to DagsHub via MLflow.")
