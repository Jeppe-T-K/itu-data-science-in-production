import numpy as np
import pandas as pd
import mlflow
import os

from sklearn.metrics import mean_squared_error

from data_util import get_data, get_data_drift_data, get_concept_drift_data
from mlflow.pyfunc import PythonModel


class BasicModel(PythonModel):
    def linear(self, numbers):
        return np.array([x[0] for x in numbers])  # implement a simple linear function

    def predict(self, context, model_input) -> np.ndarray[float]:
        if isinstance(model_input, pd.DataFrame):
            model_input = list(model_input.values)
        return self.linear(model_input)


# Load data
X_train, X_test, y_train, y_test = get_data()

# Define the model hyperparameters
params = {}

# Train the model
lr = BasicModel()

# Predict on the test set
y_pred = lr.predict(params, X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)

mlflow.set_experiment("Basic Linear Model")
with mlflow.start_run():
    # Create a Dataset object for later
    os.makedirs("mlflow_artifacts", exist_ok=True)
    csv_path = "mlflow_artifacts/X_train.csv"
    pd.DataFrame(X_test).to_csv(csv_path, index=False)
    dataset = mlflow.data.from_pandas(pd.DataFrame(X_test), source=csv_path)
    mlflow.log_input(dataset, context="training")

    # Log other parameters
    mlflow.log_params(params)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.pyfunc.log_model(
        name="Custom Linear Model", 
        python_model=lr,
        registered_model_name="custom_linear_model"
    )