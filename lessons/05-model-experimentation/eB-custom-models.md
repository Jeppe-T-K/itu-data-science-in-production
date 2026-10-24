---
title: Custom Models
---

# Exercise 2: A Custom Model Wrapper

Sometimes you can't use a standard scikit-learn (or otherwise supported) model, so you have to define your own.

A very common case: a **classification threshold**. `LogisticRegression.predict()` classifies based on whether the predicted probability is above 0.5 - but 0.5 is not sacred. If one class is much rarer than the other, or a false positive is much costlier than a false negative, the _useful_ threshold might be 0.3, 0.7, or anywhere else.

The threshold is a decision you make _with data_ during experimentation, and it belongs _with the model_ when you promote it to production - not hardcoded in some serving script by whoever deploys it. So in this exercise you will wrap a logistic regression in a custom MLflow model that also stores the threshold, using MLflow's [`PythonModel`](https://mlflow.org/docs/latest/api/python_model/):

### Exercise 2.1: Train a classifier (and question the default threshold)

1. <details><summary>Generate classification data</summary>
   Use <code>get_classification_data()</code> from <code>resources/data_util.py</code> and train a <code>LogisticRegression</code> on it.

   <pre><i><u>Look at <code>predict_proba</code> for a few test samples: what exactly does the model output? How does <code>predict</code> turn those probabilities into classes?</u></i></pre>
   </details>

2. <details><summary>Sweep thresholds</summary>
   Using the test split, compute precision, recall and F1 for thresholds from e.g. 0.1 to 0.9 in steps of 0.05, based on <code>predict_proba</code>.

   <pre><i><u>Pick a "best" threshold for your metric of choice. Why is F1 at threshold 0.5 not necessarily the best choice? When would you favor precision over recall (or vice versa)?</u></i></pre>
   </details>

3. <details><summary>Log the sweep to MLflow</summary>
   Log the whole sweep as an artifact (e.g. <code>mlflow.log_dict(sweep, "threshold_sweep.json")</code>) and the chosen threshold as a parameter, together with the usual training run: parameters, metrics, and the (un-wrapped) model.
   </details>

### Exercise 2.2: Wrap it in a `PythonModel`

1. <details><summary>Define the wrapper class</summary>
   Subclass <code>mlflow.pyfunc.PythonModel</code>. The wrapper holds both the fitted model <i>and</i> the threshold:

   ```python
   class ThresholdedLogisticRegression(PythonModel):
       def __init__(self, model, threshold=0.5):
           self.model = model
           self.threshold = threshold

       def predict(self, context, model_input):
           if isinstance(model_input, pd.DataFrame):
               model_input = model_input.values
           proba = self.model.predict_proba(model_input)[:, 1]
           return (proba >= self.threshold).astype(int)
   ```

   <pre><i><u>Why must the threshold live in the model object (or an artifact) instead of in the caller's code?</u></i></pre>
   </details>

2. <details><summary>Log and register the wrapper</summary>
   Instead of <code>mlflow.sklearn.log_model</code>, use the pyfunc flavour:

   ```python
   mlflow.pyfunc.log_model(
       name="thresholded_logistic_model",
       python_model=ThresholdedLogisticRegression(lr, threshold=best_threshold),
       registered_model_name="thresholded_logistic_model",
   )
   ```

   Note that <code>python_model</code> is an <i>instance</i> - everything you store on it is serialized with the model.
   </details>

3. <details><summary>Load it back and verify</summary>
   Load the model from the registry (<code>mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")</code>) and predict on the test set.

   <pre><i><u>Verify that the loaded model's predictions equal what you get by manually thresholding <code>predict_proba</code> at your chosen threshold. What happens if you load the scikit-learn flavour instead - does it know anything about your threshold?</u></i></pre>
   </details>

4. <details><summary>Experiment with the threshold</summary>
   Log a couple of runs with different thresholds (and the same underlying model). Look at the runs in the MLflow UI.

   <pre><i><u>You just ran several "experiments" that share the same trained model but represent <b>different products</b>. Why does it make sense to track the threshold as a first-class parameter of a run?</u></i></pre>
   </details>

### Exercise 2.3: Reflect

<pre><i><u>Discuss:

<li>The lecture talked about training-serving skew. Where could a threshold reintroduce it if it is NOT stored with the model?</li>
<li>Which other "extras" could you imagine bundling with a model this way? (hint: feature scaling, label encoders, calibration...)</li></u></i></pre>
