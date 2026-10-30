import mlflow

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.50, random_state=42, stratify=y_rest
)


def train_and_log(params, experiment_name="Basic Logistic Model"):
    """Train a LogisticRegression on the breast cancer data and log everything to MLflow."""
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run() as run:
        # Train inside the run: the whole experiment (fit, predict, metric)
        # belongs to the run, and autologging relies on it
        lr = LogisticRegression(**params)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f"Params: {params} -> F1 Score: {f1}")

        mlflow.log_params(params)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(lr, name="logistic_regression_model")
    return run.info.run_id


# 1. Run the experiment with two different parameter settings
run_id_1 = train_and_log({"C": 0.5, "max_iter": 5000, "random_state": 8})
run_id_2 = train_and_log({"C": 0.1, "max_iter": 5000, "random_state": 8})

# Register the logged model of the first run in the model registry
model_details = mlflow.register_model(
    f"runs:/{run_id_1}/logistic_regression_model", name="breast_cancer_classifier"
)
print(f"Registered {model_details.name} version {model_details.version}")

# 2. Compare the runs and select the best one programmatically
runs = mlflow.search_runs(
    experiment_names=["Basic Logistic Model"],
    order_by=["metrics.f1_score DESC"],
)
best_run = runs.iloc[0]
print(f"Best run: {best_run['run_id']} (F1 {best_run['metrics.f1_score']:.4f})")

# 3. Autologging variant: let MLflow figure out what to record. Replace the
#    explicit logging in train_and_log with this and compare the runs.
# mlflow.autolog()
# with mlflow.start_run():
#     lr = LogisticRegression(C=0.5, max_iter=5000, random_state=8)
#     lr.fit(X_train, y_train)
#     # metrics are logged automatically around fit/predict
