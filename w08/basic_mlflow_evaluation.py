import pandas as pd
import scipy
import mlflow

from sklearn.metrics import mean_squared_error

from data_util import get_data, get_data_drift_data, get_concept_drift_data

def detect_data_drift(X_old, X_new):
    drifted_features = []
    for i in range(X_old.shape[1]):
        res = scipy.stats.kstest(X_old[:, i], X_new[:, i])
        if res.pvalue < 0.05:  # significance level
            drifted_features.append(i)
    return drifted_features

# Load data
X_train, X_test, y_train, y_test = get_data_drift_data()

model_name = "custom_linear_model"
model_version = 1

# Load logged model
lr = mlflow.pyfunc.load_model(
    f"models:/{model_name}/{model_version}"
)
# Get latest training data from runs
run = mlflow.get_run(lr.metadata.run_id)

dataset_info = run.inputs.dataset_inputs[0].dataset
X_train_old = pd.read_csv(mlflow.data.get_source(dataset_info).load()).values

# Create evaluation dataset
eval_data = pd.DataFrame(X_test)
eval_data["target"] = y_test

mlflow.set_experiment("Basic Linear Model")
with mlflow.start_run():
    print(X_train_old, X_test)
    print(detect_data_drift(X_train_old, X_test))
    #mlflow.log_metric("data_drifted_features", len(detect_data_drift(X_train_old, X_test)))
    result = mlflow.evaluate(
    lr,  # Function to evaluate
    eval_data,  # Evaluation data
    targets="target",  # Target column
    model_type="regressor",  # Task type
)