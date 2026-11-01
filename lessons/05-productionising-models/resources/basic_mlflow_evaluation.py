"""A possible solution for Exercise 4: evaluate the registered thresholded
classifier on drifted data - detect data drift per feature with a KS test,
measure accuracy/F1 on both drift variants, and log each evaluation as an
MLflow run."""

import numpy as np
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

def detect_label_drift(y_old, y_new):
    """Chi-square test: does the class distribution of the new data
    still match the baseline from training?"""
    classes = np.unique(np.concatenate([y_old, y_new]))
    old_counts = np.array([(y_old == c).sum() for c in classes])
    new_counts = np.array([(y_new == c).sum() for c in classes])
    expected = new_counts.sum() * old_counts / old_counts.sum()
    res = scipy.stats.chisquare(f_obs=new_counts, f_exp=expected)
    return res.pvalue

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

        label_pvalue = detect_label_drift(y_test, y_new)
        mlflow.log_metric("label_drift_pvalue", label_pvalue)

        # The wrapper returns predictions together with the threshold that
        # produced them
        y_pred = model.predict(pd.DataFrame(X_new))["predictions"]
        accuracy = accuracy_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        print(
            f"{name}: drifted features={len(drifted)}, "
            f"label_drift_p={label_pvalue:.4f}, "
            f"accuracy={accuracy:.4f}, f1={f1:.4f}"
        )
