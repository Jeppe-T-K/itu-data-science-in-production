---
title: MLflow Basics
---

# Exercise 1: MLflow Catch-Up

MLflow should be familiar to you at this point, so this first exercise is a quick catch-up to make sure everyone has a working setup and the core concepts (experiments, runs, parameters, metrics, artifacts, the model registry) are fresh. If you have never used MLflow before, don't panic - the exercise is a self-contained introduction, and everything you need is under the [MLflow tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/).

MLflow has a lot of built-in "flavours" that make it easy to track models from well-known ML libraries such as scikit-learn or Tensorflow. That is what we use here - the _custom_ models come in the next exercise.

Make sure MLflow and friends are installed (see Exercise 0 on the [exercise intro](exercise-intro) page):
`pip install mlflow`
`pip install scipy`
`pip install scikit-learn`

### Exercise 1.1: Run an ML experiment

1. <details> <summary> Edit the <code>resources/example_model.py</code> script to log the model run with MLflow</summary>
   First <code>import mlflow</code>

   Then set the MLflow experiment via <code>mlflow.set_experiment("My experiment name")</code>

   Then start an MLflow run, log the parameters and the metric, and log the model as an artifact _of the run_:

   ```python
   with mlflow.start_run() as run:
       mlflow.log_params(params)
       mlflow.log_metric("mean_squared_error", mse)
       mlflow.sklearn.log_model(lr, name="lasso_model")
   ```

   Finally, register the model in the model registry, pointing at the artifact you just logged. This two-step flow (log to the run, then register from the run) is exactly how the course project's training notebook does it:

   ```python
   mlflow.register_model(
       f"runs:/{run.info.run_id}/lasso_model",
       name="lasso_regression_model",
   )
   ```
   </details>

2. <details> <summary> Run training with different parameters</summary>
   For example set alpha in params to 0.5.
   </details>

3. <details> <summary> Compare the runs and select the best model</summary>
   You can compare the two runs via the printed output or via the MLflow UI:

   In the terminal run <code>mlflow ui</code>

   You can also select the best run _programmatically_, which is what the project's deployment notebook does when it picks the model to promote (there it orders by the test F1-score):

   ```python
   runs = mlflow.search_runs(
       experiment_names=["My experiment name"],
       order_by=["metrics.mean_squared_error ASC"],
   )
   best_run = runs.iloc[0]
   ```

   <pre><i><u>Why is selecting "the best run" a decision you should make in code rather than by eyeballing the UI?</u></i></pre>
   </details>

4. <details><summary>What exactly is logged for each run?</summary>
   Check the mlruns directory or through the UI under "artifacts". For each run you should be able to find: the parameters you logged, the metric, and the model artifact together with its metadata (conda environment, the model's input signature if MLflow could infer one, ...).

   By default everything is stored locally under <code>./mlruns</code>. MLflow can also log to a central place with <code>mlflow.set_tracking_uri(...)</code> - a database, or a tracking server your whole team shares. The course project, for instance, points MLflow at a local SQLite database so the run history lives in one file.
   </details>

5. <details><summary>Try to use autologging</summary>
   Add <code>mlflow.autolog()</code> and remove other MLflow logging

   <pre><i><u>Look at what the autologged run contains compared to your explicit logging. Which approach gives you more control? Why might a production training job (like the project's) prefer explicit logging of exactly the parameters and metrics it cares about?</u></i></pre>
   </details>

If all of that went smoothly: great, you are caught up. The interesting part starts in the next exercise, where a model is _more_ than what a standard flavour can express.
