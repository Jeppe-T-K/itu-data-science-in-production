import pandas as pd
import scipy
import mlflow

from data_util import (
    get_classification_data,
    get_classification_drift_data,
    get_classification_concept_data,
)

def detect_data_drift(X_old, X_new):
    drifted_features = []
    for i in range(X_old.shape[1]):
        res = scipy.stats.kstest(X_old[:, i], X_new[:, i])
        if res.pvalue < 0.05:  # significance level
            drifted_features.append(i)
    return drifted_features

# The training data, to compare new data against
X_train, _, y_train, _ = get_classification_data()

model_name = "thresholded_logistic_model"
model_version = 1  # Or higher if you ran it multiple times

# Load logged model (the threshold is stored with it)
model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/{model_version}"
)

# The new data the model is meeting "in production".
# Swap between the two to compare data drift vs. concept drift.
X_new, X_eval, y_new, y_eval = get_classification_drift_data()
# X_new, X_eval, y_new, y_eval = get_classification_concept_data()

# Create evaluation dataset
eval_data = pd.DataFrame(X_eval)
eval_data["target"] = y_eval

mlflow.set_experiment("Thresholded Logistic Regression")
with mlflow.start_run():
    data_drift_columns = detect_data_drift(X_train, X_eval)
    mlflow.log_metric("data_drifted_features", len(data_drift_columns))

    result = mlflow.evaluate(
        model,             # Function to evaluate
        eval_data,         # Evaluation data
        targets="target",  # Target column
        model_type="classifier",  # Task type
    )
