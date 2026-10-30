import pandas as pd
import scipy
import mlflow

from sklearn.metrics import accuracy_score, f1_score

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

# The test split from training time is the baseline the new data is
# compared against (in the course project, the training job stores a
# baseline statistics artifact for exactly this purpose)
_, _, X_test, _, _, y_test = get_classification_data()

model_name = "thresholded_logistic_model"
model_version = 1  # Or higher if you ran it multiple times

# Load logged model (the threshold is stored with it)
model = mlflow.pyfunc.load_model(
    f"models:/{model_name}/{model_version}"
)

# The new data the model is meeting "in production".
# Swap between the two to compare data drift vs. concept drift.
_, X_new, _, y_new = get_classification_drift_data()
# _, X_new, _, y_new = get_classification_concept_data()

mlflow.set_experiment("Thresholded Logistic Regression")
with mlflow.start_run(run_name="drift_evaluation"):
    data_drift_columns = detect_data_drift(X_test, X_new)
    mlflow.log_metric("data_drifted_features", len(data_drift_columns))

    # The wrapper returns predictions together with a timestamp
    output = model.predict(pd.DataFrame(X_new))
    y_pred = output["predictions"]
    mlflow.log_metric("accuracy", accuracy_score(y_new, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_new, y_pred))
