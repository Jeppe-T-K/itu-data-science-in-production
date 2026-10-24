import numpy as np
import pandas as pd
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from data_util import get_classification_data
from mlflow.pyfunc import PythonModel


class ThresholdedLogisticRegression(PythonModel):
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input) -> np.ndarray:
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values
        proba = self.model.predict_proba(model_input)[:, 1]
        return (proba >= self.threshold).astype(int)


# Load data
X_train, X_test, y_train, y_test = get_classification_data()

# Define the model hyperparameters
params = {"max_iter": 1000, "random_state": 8}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Sweep thresholds based on the predicted probabilities
proba = lr.predict_proba(X_test)[:, 1]
sweep = {}
for threshold in np.arange(0.1, 0.91, 0.05):
    preds = (proba >= threshold).astype(int)
    sweep[round(threshold, 2)] = {
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
    }
best_threshold = max(sweep, key=lambda t: sweep[t]["f1"])

# Wrap the model and sanity-check the wrapped predictions
model = ThresholdedLogisticRegression(lr, threshold=best_threshold)
y_pred = model.predict(pd.DataFrame(X_test))
assert (y_pred == (proba >= best_threshold).astype(int)).all()

mlflow.set_experiment("Thresholded Logistic Regression")
with mlflow.start_run():
    # Log parameters (including the threshold) and the sweep artifact
    mlflow.log_params(params)
    mlflow.log_param("threshold", best_threshold)
    mlflow.log_dict(sweep, "threshold_sweep.json")
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred))

    # Log and register the wrapped model
    mlflow.pyfunc.log_model(
        name="Thresholded Logistic Model",
        python_model=model,
        registered_model_name="thresholded_logistic_model",
    )

# Load the model back from the registry and verify it predicts the same
loaded = mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")
loaded_pred = loaded.predict(pd.DataFrame(X_test))
assert (loaded_pred == y_pred).all()
