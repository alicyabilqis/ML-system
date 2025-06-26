# modelling.py {FIXX}

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set MLflow Tracking ke lokal
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Forest_Cover_Classification")

# Load data
data = pd.read_csv("Forest cover_preprocessing_dataset.csv")
X = data.drop("Cover_Type", axis=1)
y = data["Cover_Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Mulai experiment MLflow
with mlflow.start_run(run_name="No Tuning"):
    mlflow.autolog()  # Aktifkan autolog
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    print("Model trained and logged using autolog.")
