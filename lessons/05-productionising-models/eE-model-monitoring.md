---
title: Model Monitoring
---

# Exercise 4: Check Performance of Deployed Model

Once a model is deployed, you need to monitor its performance. Here we test how we can log two things: **data drift**, which is when your input variables drift, and **concept drift**, which is when the fundamental relationship between your dependent and independent variables changes.

We continue with the thresholded classifier from the previous exercises. <code>resources/data_util.py</code> provides two new datasets for this, built from the same breast cancer data: <code>get_breast_cancer_drift_data()</code>, where the measurement device has drifted so that every reading comes out 25% higher, and <code>get_breast_cancer_concept_data()</code>, where a new labeling guideline has flipped every diagnosis.

In these exercises the test set (from <code>get_breast_cancer_data()</code>) plays the role of "the data the model was approved on": it is the baseline we compare new data against, and it is where we evaluate performance. The course project does the same thing more seriously: its training notebook stores a set of baseline statistics, and the serving endpoint compares live traffic against that baseline.

### Exercise 4.1: Evaluate on drifted data

1. <details> <summary> Load your registered model </summary>
   Load the wrapper with <code>mlflow.pyfunc.load_model("models:/thresholded_logistic_model/1")</code> - the threshold comes with it, as it should.

   <i><u>You logged the model in Exercise 2 - check in the MLflow UI what version your model has ended up on.</u></i>
   </details>

2. <details> <summary> Evaluate the model on both new datasets </summary>
   Both new datasets come with labels, so you can measure performance directly. Log one run per dataset:

   ```python
   datasets = {
       "data_drift": get_breast_cancer_drift_data,
       "concept_drift": get_breast_cancer_concept_data,
   }

   for name, loader in datasets.items():
       with mlflow.start_run(run_name=name):
           X_new, y_new = loader()
           output = model.predict(pd.DataFrame(X_new))
           y_pred = output["predictions"]
           mlflow.log_metric("accuracy", accuracy_score(y_new, y_pred))
           mlflow.log_metric("f1_score", f1_score(y_new, y_pred))
   ```

   <i><u>For a model that returns plain predictions (a standard flavour), <code>mlflow.evaluate</code> can compute and log all of this for you in one call. Why do we extract <code>predictions</code> and compute the metrics ourselves here? (Hint: look at what your wrapper returns.) Why is this useful to wrap this in MLflow?</u></i>
   </details>

3. <details> <summary> Which test detects which drift? </summary>
   Here are two drift detectors. Look at what each of them compares, and figure out which one detects the data drift and which one can detect the concept drift. Then add the matching test call to each of your two runs and log the result as a metric.

   ```python
   def detect_foo_drift(X_old, X_new):
       """Kolmogorov-Smirnov test"""
       drifted_features = []
       for i in range(X_old.shape[1]):
           res = scipy.stats.kstest(X_old[:, i], X_new[:, i])
           if res.pvalue < 0.05:
               drifted_features.append(i)
       return drifted_features

   def detect_bar_drift(y_old, y_new):
       """Chi-square test"""
       classes = np.unique(np.concatenate([y_old, y_new]))
       old_counts = np.array([(y_old == c).sum() for c in classes])
       new_counts = np.array([(y_new == c).sum() for c in classes])
       expected = new_counts.sum() * old_counts / old_counts.sum()
       res = scipy.stats.chisquare(f_obs=new_counts, f_exp=expected)
       return res.pvalue
   ```

   <i><u>Which dataset does each test fire on? Why does the feature test stay quiet for the concept drift data even though the model's error explodes?</u></i>

   <details><summary>What you should see</summary>

   On the data drift dataset: the feature test fires on about 26 of the 30 features, the label test stays quiet, and accuracy drops from about 0.97 to about 0.74. On the concept drift dataset: the feature test stays quiet, the label test fires (p-value near zero), and accuracy collapses to about 0.05 - the inputs are unchanged, but the outcomes flipped.
   </details>
   </details>

4. <details> <summary> Discuss: measuring drift in production </summary>
   In these exercises the new data arrived as a ready-made file, labels included, and each drift check was a handful of lines of code. In production both drifts are harder to measure - not because the statistics get harder, but because the data to run them on has to be collected somewhere.

   - <b>Data drift</b> only involves inputs, so in principle your serving service sees everything it needs on every request. What would you have to log to keep the KS tests running on live traffic?
   - <b>Concept drift</b> only showed up through the labels: the feature distributions were fine, the class balance flipped. The test above needs true labels for new data - where do they come from?

   <details><summary>What it takes in practice</summary>

   For <b>data drift</b>:

   - Log every request: the feature values, the prediction, a timestamp, and which model version served it - without the features there is nothing to test.
   - Store the baseline (the test set, or summary statistics saved by the training job) so live data has something to be compared against.
   - Aggregate requests into time windows and run the tests per window, alerting on drift counts or p-values - never on individual requests.

   For <b>concept drift</b>:

   - True labels arrive late (a confirmed diagnosis, a returned order, ...): join the delayed ground truth back to the logged predictions and track windowed accuracy/F1 over time.
   - Watch the true label distribution itself: if inputs and predictions are stable but the outcome share moves, the relationship underneath has changed.
   - Monitor calibration: predicted probabilities that no longer match observed frequencies are an early concept-drift symptom.
   - Sample predictions for human review where labels never materialize - and expect the remedy to be relabeling or retraining, not a "fix" of the model.
   </details>
   </details>

And that is it! Hopefully these exercises gave you an idea how you can deploy and serve a model via MLflow/FastAPI. You might see more complex setups in production systems (including the project), but now all the exercises should cover the main parts you need for the project!

[Back to Introduction](exercise-intro)
