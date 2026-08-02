---
title: MLflow Basics
---

# Exercise 1: Run an ML Experiment

For this exercise, the goal is to use MLflow to create an experiment and log relevant artifacts and metrics for different ML models. Much of what you need for this is under the https://mlflow.org/docs/latest/ml/tracking/quickstart/ page.

MLflow has a lot of in-built "flavours" that make it easy to run experiments for well-known ML libraries such as SKlearn or Tensorflow. For this exercise we can use the SKlearn model in 

### Exercise 0: Installation

Make sure MLflow and scipy is installed:
`pip install mlflow`
`pip install scipy`
`pip install scikit-learn`

### Exercise 1: Run an ML experiment

1. <details> <summary> Edit the example_model.py script to log model run with MLflow</summary>
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

[Next: Exercise 2 - Running a Custom Model](C-custom-models.md)
