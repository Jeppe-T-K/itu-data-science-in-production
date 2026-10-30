"""A possible solution for Exercise 4: evaluate the registered thresholded
classifier on drifted data - detect data drift per feature with a KS test,
measure accuracy/F1 on both drift variants, and log each evaluation as an
MLflow run."""

import pandas as pd
import scipy
import mlflow

from sklearn.metrics import accuracy_score, f1_score

from data_util import (
    get_breast_cancer_data,
    get_breast_cancer_drift_data,
    get_breast_cancer_concept_data,
)

def detect_data_drift(X_old, X_new):
    drifted_features = []
    for i in range(X_old.shape[1]):
        res = scipy.stats.kstest(X_old[:, i], X_new[:, i])
        if res.pvalue < 0.05:  # significance level
            drifted_features.append(i)
    return drifted_features

# The test split from training time is the drift baseline: it is the data
# the model was approved on (in the course project, the training job stores
# a baseline statistics artifact for exactly this purpose)
_, _, X_test, _, _, y_test = get_breast_cancer_data()

model_name = "thresholded_logistic_model"
model_version = 1  # Or higher if you ran Exercise 2 multiple times

# Load the registered wrapper (the threshold is stored with it)
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# The new data the model is meeting "in production"
datasets = {
    "data_drift": get_breast_cancer_drift_data,
    "concept_drift": get_breast_cancer_concept_data,
}

mlflow.set_experiment("Thresholded Logistic Regression")
for name, loader in datasets.items():
    X_new, y_new = loader()

    with mlflow.start_run(run_name=name):
        drifted = detect_data_drift(X_test, X_new)
        mlflow.log_metric("data_drifted_features", len(drifted))

        # The wrapper returns predictions together with a timestamp
        y_pred = model.predict(pd.DataFrame(X_new))["predictions"]
        accuracy = accuracy_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(
            f"{name}: drifted features={len(drifted)}, "
            f"accuracy={accuracy:.4f}, f1={f1:.4f}"
        )
