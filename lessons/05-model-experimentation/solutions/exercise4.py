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

# The training data to compare new data against
X_train, _, y_train, _ = get_classification_data()

# Load the registered wrapper (the threshold is stored with it)
model = mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")

# Evaluate on both drifted datasets and log each as a run
datasets = {
    "data_drift": get_classification_drift_data,
    "concept_drift": get_classification_concept_data,
}

mlflow.set_experiment("Thresholded Logistic Regression")
for name, loader in datasets.items():
    _, X_eval, _, y_eval = loader()

    eval_data = pd.DataFrame(X_eval)
    eval_data["target"] = y_eval

    with mlflow.start_run(run_name=name):
        drifted = detect_data_drift(X_train, X_eval)
        mlflow.log_metric("data_drifted_features", len(drifted))

        result = mlflow.evaluate(
            model,             # the loaded pyfunc wrapper
            eval_data,         # evaluation data
            targets="target",  # target column
            model_type="classifier",
        )

        print(
            f"{name}: drifted features={drifted}, "
            f"accuracy={result.metrics.get('accuracy')}"
        )
