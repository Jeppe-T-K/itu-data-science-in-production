import mlflow

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from data_util import get_data

# Load data
X_train, X_test, y_train, y_test = get_data()

# Define the model hyperparameters
params = {
    "alpha": 0.5,
    "max_iter": 1000,
    "random_state": 8,
}

# Train the model
lr = Lasso(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

mlflow.set_experiment("Basic Linear Model")
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(params)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.sklearn.log_model(
        lr,
        "lasso_model",
        registered_model_name="lasso_linear_model"
    )