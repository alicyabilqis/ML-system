# modelling_tuning.py {FIXX}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set MLflow Tracking ke lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Forest_Cover_Classification")

# Load data
data = pd.read_csv("Forest cover_preprocessing_dataset.csv")
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Hyperparameter ranges
n_estimators_range = [40, 50, 60]
max_depth_range = [12, 14, 16]

best_accuracy = 0
best_params = {}
best_model = None

with mlflow.start_run(run_name="Hyperparameter Tuning"):
    for n in n_estimators_range:
        for d in max_depth_range:
            with mlflow.start_run(nested=True):
                model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
                model.fit(X_train, y_train)
                acc = model.score(X_test, y_test)

                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model = model
                    best_y_pred = model.predict(X_test)

    input_example = X_test.iloc[:1]
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model_tuned",
        input_example=input_example
    )

    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_params(best_params)

    # Log confusion matrix
    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Training Confusion Matrix")
    plt.savefig("training_confusion_matrix.png")
    plt.close()
    mlflow.log_artifact("training_confusion_matrix.png")

    # Log metric_info.json
    with open("metric_info.json", "w") as f:
        json.dump({"best_accuracy": best_accuracy}, f)
    mlflow.log_artifact("metric_info.json")

    # Log estimator.html
    with open("estimator.html", "w") as f:
        f.write(f"<html><body><h2>Best Estimator</h2><p>{best_model}</p></body></html>")
    mlflow.log_artifact("estimator.html")

    print(f"\n✅ Model terbaik hasil tuning: {best_params} - Accuracy: {best_accuracy:.4f}")
