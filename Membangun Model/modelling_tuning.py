# modelling_tuning.py {FIXX}

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set MLflow Tracking ke lokal
#mlflow.set_tracking_uri("http://127.0.0.1:5000")
#mlflow.set_experiment("Forest_Cover_Classification")

# Load dataset
#!gdown 1HHv8WwNGGksU2IwY2vIJlsD8xr5tBsiV
data = pd.read_csv("Forest cover_preprocessing_dataset.csv")
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

                # Logging manual
                mlflow.log_param("n_estimators", n)
                mlflow.log_param("max_depth", d)
                mlflow.log_metric("accuracy", acc)

                print(f"Tuning - n_estimators={n}, max_depth={d}, accuracy={acc:.4f}")

                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params = {"n_estimators": n, "max_depth": d}
                    best_model = model

    input_example = X_test.iloc[:1]
    
    # Log model terbaik
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model_tuned",
        input_example=input_example
    )
    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_params(best_params)

    print(f"\n✅ Model terbaik hasil tuning: {best_params} - Accuracy: {best_accuracy:.4f}")

