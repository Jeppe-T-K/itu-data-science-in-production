---
title: MLflow Basics
---

# Exercise 1: MLflow Catch-Up

MLflow should be familiar to you at this point, so this first exercise is a quick catch-up to make sure everyone has a working setup and the core concepts (experiments, runs, parameters, metrics, artifacts) are fresh. If you have never used MLflow before, don't panic — the exercise is a self-contained introduction, and everything you need is under the [MLflow tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/).

MLflow has a lot of built-in "flavours" that make it easy to track models from well-known ML libraries such as scikit-learn or Tensorflow. That is what we use here — the _custom_ models come in the next exercise.

### Exercise 0: Installation

Make sure MLflow and friends are installed:
`pip install mlflow`
`pip install scipy`
`pip install scikit-learn`

### Exercise 1.1: Run an ML experiment

1. <details> <summary> Edit the <code>resources/example_model.py</code> script to log the model run with MLflow</summary>
   First <code>import mlflow</code>

   Then set the MLflow experiment via <code>mlflow.set_experiment("My experiment name")</code>

   Lastly start an MLflow run and log relevant things:
   ```python
   with mlflow.start_run():
       mlflow.log_params(params)
       mlflow.log_metric("mean_squared_error", mse)
       mlflow.sklearn.log_model(
         lr, registered_model_name="lasso_regression_model"
       )
   ```
   </details>

2. <details> <summary> Run training with different parameters</summary>
   For example set alpha in params to 0.5.
   </details>

3. <details> <summary> Compare the two models </summary>
   This can be done via the printed output or via the mlflow ui.

   In the terminal run <code>mlflow ui</code>
   </details>

4. <details><summary>What exactly is logged for each run?</summary>
   Check the mlruns directory or through the UI under "artifacts"
   </details>

5. <details><summary>Try to use autologging</summary>
   Add <code>mlflow.autolog()</code> and remove other MLflow logging
   </details>

If all of that went smoothly: great, you are caught up. The interesting part starts in the next exercise, where a model is _more_ than what a standard flavour can express.

[Next: Exercise 2 - A Custom Model Wrapper](custom-models)
