import mlflow

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

from data_util import get_data


def train_and_log(params, experiment_name="Basic Linear Model"):
    """Train a Lasso on the example data and log everything to MLflow."""
    X_train, X_test, y_train, y_test = get_data()

    lr = Lasso(**params)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Params: {params} -> Mean Squared Error: {mse}")

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.sklearn.log_model(lr, name="lasso_model")
    return run.info.run_id


# 1. Run the experiment with two different parameter settings
run_id_1 = train_and_log({"alpha": 0.5, "max_iter": 1000, "random_state": 8})
run_id_2 = train_and_log({"alpha": 1.0, "max_iter": 1000, "random_state": 8})

# Register the logged model of the first run in the model registry
model_details = mlflow.register_model(
    f"runs:/{run_id_1}/lasso_model", name="lasso_regression_model"
)
print(f"Registered {model_details.name} version {model_details.version}")

# 2. Compare the runs and select the best one programmatically
runs = mlflow.search_runs(
    experiment_names=["Basic Linear Model"],
    order_by=["metrics.mean_squared_error ASC"],
)
best_run = runs.iloc[0]
print(f"Best run: {best_run['run_id']} (MSE {best_run['metrics.mean_squared_error']:.4f})")

# 3. Autologging variant: let MLflow figure out what to record. Replace the
#    explicit logging in train_and_log with this and compare the runs.
# mlflow.autolog()
# with mlflow.start_run():
#     lr = Lasso(alpha=0.5, max_iter=1000, random_state=8)
#     lr.fit(X_train, y_train)
#     # metrics are logged automatically around fit/predict
