---
title: Model Monitoring
---

# Exercise 4: Check Performance of Deployed Model

Once a model is deployed, you need to monitor its performance. Here we test how we can log two things: **data drift**, which is when your input variables drift, and **concept drift**, which is when the fundamental relationship between your dependent and independent variables changes.

We continue with the thresholded classifier from the previous exercises. `resources/data_util.py` provides two new datasets for this: `get_classification_drift_data()` and `get_classification_concept_data()`.

In these exercises the test set plays the role of "the data the model was approved on": it is the baseline we compare new data against, and it is where we evaluate performance. The course project does the same thing more seriously: its training notebook stores a set of baseline statistics, and the serving endpoint compares live traffic against that baseline.

### Exercise 4.1: Evaluate on drifted data

1. <details> <summary> Load your registered model </summary>
   Load the wrapper with <code>mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")</code> - the threshold comes with it, as it should.

   <pre><i><u>You logged the model in Exercise 2 - check in the MLflow UI what version your model has ended up on.</u></i></pre>
   </details>

2. <details> <summary> Detect data drift </summary>
   Compare the feature distributions of the test set (your baseline) and the new data with a statistical test, e.g. a Kolmogorov–Smirnov test per feature (see <code>resources/basic_mlflow_evaluation.py</code> for inspiration):

   ```python
   def detect_data_drift(X_old, X_new):
       drifted_features = []
       for i in range(X_old.shape[1]):
           res = scipy.stats.kstest(X_old[:, i], X_new[:, i])
           if res.pvalue < 0.05:
               drifted_features.append(i)
       return drifted_features
   ```

   Run it with the data drift dataset. Do you expect to see a difference in the distribution of the X variable? Do you expect the model's error to change?

   <details><summary>Expectation</summary>
   Yes, the distribution of the X variable has changed - that is essentially the definition of data drift, and the KS test should fire. Whether the accuracy changes is less obvious: our simple model might handle the new range of inputs fine. In realistic scenarios the model might not extrapolate properly, so we always need to keep an eye on performance - and on the <i>prediction distribution</i>: with a fixed threshold, a shifted input distribution shifts the share of positive predictions.
   </details>
   </details>

3. <details> <summary> Evaluate the model on both new datasets </summary>
   The new datasets come with labels, so we can measure performance directly. Get predictions from the loaded wrapper and compute the metrics yourself, logging the evaluation as an MLflow run so it can be compared against the training-time run:

   ```python
   with mlflow.start_run(run_name="data_drift"):
       mlflow.log_metric("data_drifted_features", len(detect_data_drift(X_test, X_new)))
       output = model.predict(pd.DataFrame(X_new))
       y_pred = output["predictions"]
       mlflow.log_metric("accuracy", accuracy_score(y_new, y_pred))
       mlflow.log_metric("f1_score", f1_score(y_new, y_pred))
   ```

   Then repeat with the concept drift dataset: do you expect the model error to change? Do you expect a difference in the distribution of the X variable?

   <details><summary>Expectation</summary>
   Now we expect the model's error to increase sharply, since the fundamental relationship between X and y has been reversed. The distribution of X is the same as during training, so the KS test does <i>not</i> fire. That is exactly why monitoring inputs alone is not enough: you also need to track outcomes and metrics over time.
   </details>

   <pre><i><u>For a model that returns plain predictions (a standard flavour), <code>mlflow.evaluate</code> can compute and log all of this for you in one call. Why do we extract <code>predictions</code> and compute the metrics ourselves here? (Hint: look at what your wrapper returns.)</u></i></pre>
   </details>

4. <details> <summary> Discuss: monitoring a served model </summary>
   <pre><i><u><li>Your service from Exercise 3 could log every request. How would you turn those logs into drift metrics? What would you alert on?</li>
<li>For the concept drift case, what do you need that a serving service alone cannot provide? (hint: where do the true labels come from?)</li>
<li>How does the threshold from Exercise 2 interact with drift - can a "fine" model start misbehaving just because the class balance shifts?</li></u></i></pre>
   </details>

And that is it! You can do a lot more with MLflow and FastAPI, and you might see some more complex commands later in the course, but remember: start simple and then add complexity later.

[Back to Introduction](exercise-intro)
