---
title: Custom Models
---

# Exercise 2: A Custom Model Wrapper

Sometimes you can't use a standard scikit-learn (or otherwise supported) model, so you have to define your own.

A very common case: a **classification threshold**. `LogisticRegression.predict()` classifies based on whether the predicted probability is above 0.5 - but 0.5 is not sacred. If one class is much rarer than the other, or a false positive is much costlier than a false negative, the _useful_ threshold might be 0.3, 0.7, or anywhere else.

The threshold is a decision you make _with data_ during experimentation, and it belongs _with the model_ when you promote it to production - not hardcoded in some serving script by whoever deploys it. So in this exercise you will wrap the logistic regression from your <code>example_model.py</code> of Exercise 1 in a custom MLflow model that also stores the threshold, using MLflow's [`PythonModel`](https://mlflow.org/docs/latest/api/python_model/):

### Exercise 2.1: Question the default threshold

You already have a trained classifier: <code>example_model.py</code> from Exercise 1, with its train/validation/test split. Now start editing that script instead of starting over.

1. <details><summary>Warm-up: look at the probabilities</summary>
   Look at <code>predict_proba</code> for a few validation samples. What exactly does the model output? How does <code>predict</code> turn those probabilities into classes?

   <details><summary>Hints:</summary>

   - <code>predict_proba</code> returns one row per sample and one column per class, in the order of <code>lr.classes_</code>. For our two classes that is <code>P(y = 0)</code> (malignant) and <code>P(y = 1)</code> (benign). Each row sums to 1, and for binary classification the interesting part is column 1, i.e. <code>predict_proba(X)[:, 1]</code>.
   - <code>predict</code> does not compute anything new - it picks the class with the highest probability (an argmax over the columns). For two classes that is exactly "class 1 if <code>P(y = 1) &gt;= 0.5</code>, else class 0". So <code>predict</code> is <code>predict_proba</code> with a hardcoded 0.5 threshold - that default is what we are about to question.
   </details>
   </details>

2. <details><summary>Sweep thresholds</summary>
   Using the <i>validation</i> split, compute precision, recall and F1 for thresholds from e.g. 0.1 to 0.9 in steps of 0.05, based on <code>predict_proba</code>. Then check how the chosen threshold performs on the <i>test</i> set.

   <details><summary>A possible implementation:</summary>

   ```python
   def sweep_thresholds(lr, X_val, y_val):
       proba = lr.predict_proba(X_val)[:, 1]
       threshold_sweep = {}
       for threshold in np.arange(0.1, 0.91, 0.05):
           preds = (proba >= threshold).astype(int)
           threshold_sweep[round(threshold, 2)] = {
               "f1": f1_score(y_val, preds),
               "precision": precision_score(y_val, preds, zero_division=0),
               "recall": recall_score(y_val, preds, zero_division=0),
           }
       best_threshold = max(threshold_sweep, key=lambda t: threshold_sweep[t]["f1"])
       return best_threshold, threshold_sweep
   ```
   </details>

   <i><u>Pick a "best" threshold for your metric of choice. Why is F1 at threshold 0.5 not necessarily the best choice? When would you favor precision over recall (or vice versa)?</u></i>

   <i><u>Why do we sweep on the validation split rather than the test split? (Hint: the test set is our stand-in for "unseen production data" - once we optimize against it, it stops being unseen.) The course project tunes its threshold on the validation set in exactly this way, and additionally uses macro F1 because its classes are imbalanced.</u></i>
   </details>

3. <details><summary>Log the sweep to MLflow</summary>
   Log the whole threshold sweep as an artifact and the chosen threshold as a parameter, together with the usual training run: parameters, metrics (validation and test), and the (un-wrapped) model.

   Building on the MLflow code from Exercise 1, that could look like:

   ```python
   mlflow.set_experiment("My experiment name")
   with mlflow.start_run() as run:
       best_threshold, threshold_sweep = sweep_thresholds(lr, X_val, y_val)

       mlflow.log_params(params)
       mlflow.log_param("threshold", best_threshold)
       mlflow.log_dict(threshold_sweep, "threshold_sweep.json")

       mlflow.log_metric("validation_f1_score", threshold_sweep[best_threshold]["f1"])
       test_preds = (lr.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)
       mlflow.log_metric("test_f1_score", f1_score(y_test, test_preds))

       mlflow.sklearn.log_model(lr, name="logistic_regression_model")
   ```
   </details>

### Exercise 2.2: Wrap it in a `PythonModel`

1. <details><summary>Define the wrapper class</summary>
   Subclass <code>mlflow.pyfunc.PythonModel</code>. The wrapper holds both the fitted model <i>and</i> the threshold, and returns the predictions together with a timestamp of the prediction call - a small taste of how the course project's wrapper returns predictions plus extra outputs (there: confidences, and a log of every request):

   ```python
   from datetime import datetime, timezone

   class ThresholdedLogisticRegression(PythonModel):
       def __init__(self, model, threshold=0.5):
           self.model = model
           self.threshold = threshold

       def predict(self, context, model_input):
           if isinstance(model_input, pd.DataFrame):
               model_input = model_input.values
           proba = self.model.predict_proba(model_input)[:, 1]
           return {
               "predictions": (proba >= self.threshold).astype(int).tolist(),
               "timestamp": datetime.now(timezone.utc).isoformat(),
           }
   ```

   <i><u>Why must the threshold live in the model object (or an artifact) instead of in the caller's code? And why does the wrapper return plain Python lists instead of numpy arrays? (Hint: the output has to survive being sent over HTTP as JSON.)</u></i>
   </details>

2. <details><summary>Log and register the wrapper</summary>
   Log the wrapper as the model artifact of the run with the pyfunc flavour:

   ```python
   with mlflow.start_run() as run:
       ...
       mlflow.pyfunc.log_model(
           name="thresholded_logistic_model",
           python_model=ThresholdedLogisticRegression(lr, threshold=best_threshold),
       )

   mlflow.register_model(
       f"runs:/{run.info.run_id}/thresholded_logistic_model",
       name="thresholded_logistic_model",
   )
   ```

   Note that <code>python_model</code> is an <i>instance</i> - everything you store on it is serialized with the model. This is the same log-then-register flow you used in Exercise 1 and that the course project uses.
   </details>

3. <details><summary>Load it back and verify</summary>
   Load the model from the registry (<code>mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")</code>) and predict on the test set.

   Hardcoding the version number is fine for this exercise. The course project avoids the hardcoding by referring to registry stages and aliases instead (e.g. <code>models:/cnn_model@Production</code>) - you will meet those ideas later in the course.

   <i><u>Verify that the loaded model's <code>predictions</code> equal what you get by manually thresholding <code>predict_proba</code> at your chosen threshold. What happens if you load the scikit-learn flavour instead - does it know anything about your threshold?</u></i>
   </details>

4. <details><summary>Experiment with the threshold</summary>
   Log a couple of runs with different thresholds (and the same underlying model). Look at the runs in the MLflow UI.

   <i><u>You just ran several "experiments" that share the same trained model but represent <b>different products</b>. Why does it make sense to track the threshold as a first-class parameter of a run?</u></i>
   </details>

### Exercise 2.3: Reflect

Discuss:

- The lecture talked about training-serving skew. Where could a threshold reintroduce it if it is NOT stored with the model?
- Which other "extras" could you imagine bundling with a model this way? (hint: feature scaling, label encoders, calibration...)
