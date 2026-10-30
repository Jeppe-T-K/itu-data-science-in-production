from datetime import datetime, timezone

import numpy as np
import pandas as pd
import mlflow

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mlflow.pyfunc import PythonModel


class ThresholdedLogisticRegression(PythonModel):
    """A logistic regression wrapped with a decision threshold, returning
    the predictions together with a timestamp of the prediction call."""

    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input) -> dict:
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values
        proba = self.model.predict_proba(model_input)[:, 1]
        return {
            "predictions": (proba >= self.threshold).astype(int).tolist(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Load data (stratified 70/15/15 train/validation/test split)
X, y = load_breast_cancer(return_X_y=True)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, random_state=42, stratify=y_rest
)

# Define the model hyperparameters
params = {"C": 0.5, "max_iter": 5000, "random_state": 8}


# Sweep thresholds on the validation split
def sweep_thresholds(lr, X_val, y_val):
    proba = lr.predict_proba(X_val)[:, 1]
    threshold_sweep = {}
    for threshold in np.arange(0.1, 0.91, 0.05):
        preds = (proba >= threshold).astype(int)
        threshold_sweep[round(threshold, 2)] = {
            "f1": f1_score(y_val, preds),
            "precision": precision_score(y_val, preds, zero_division=0),
            "recall": recall_score(y_val, preds, zero_division=0),
        }
    best_threshold = max(threshold_sweep, key=lambda t: threshold_sweep[t]["f1"])
    return best_threshold, threshold_sweep


mlflow.set_experiment("Thresholded Logistic Regression")
with mlflow.start_run() as run:
    # Train inside the run: the whole experiment belongs to the run
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)

    # Sweep thresholds and pick the best one
    best_threshold, threshold_sweep = sweep_thresholds(lr, X_val, y_val)

    # Wrap the model and sanity-check the wrapped predictions
    model = ThresholdedLogisticRegression(lr, threshold=best_threshold)
    proba = lr.predict_proba(X_val)[:, 1]
    expected = (proba >= best_threshold).astype(int).tolist()
    # Outside MLflow the context is unused, so pass it explicitly
    assert model.predict(None, pd.DataFrame(X_val))["predictions"] == expected

    # Log parameters (including the chosen threshold) and the sweep artifact
    mlflow.log_params(params)
    mlflow.log_param("threshold", best_threshold)
    mlflow.log_dict(threshold_sweep, "threshold_sweep.json")

    # Evaluate at the chosen threshold on the validation and test splits
    for split, X, y in [("validation", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(None, pd.DataFrame(X))["predictions"]
        mlflow.log_metric(f"{split}_accuracy", accuracy_score(y, y_pred))
        mlflow.log_metric(f"{split}_f1_score", f1_score(y, y_pred))

    # Log the wrapped model as the artifact of the run
    mlflow.pyfunc.log_model(
        name="thresholded_logistic_model",
        python_model=model,
    )

# Register the logged model in the model registry (after the run)
mlflow.register_model(
    f"runs:/{run.info.run_id}/thresholded_logistic_model",
    name="thresholded_logistic_model",
)

# Load the model back from the registry and verify it predicts the same
loaded = mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")
loaded_pred = loaded.predict(pd.DataFrame(X_test))["predictions"]
assert loaded_pred == model.predict(None, pd.DataFrame(X_test))["predictions"]
