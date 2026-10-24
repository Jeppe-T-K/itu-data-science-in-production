---
title: Local Pipeline
---

# Exercise 3: Build and Run the Pipeline Locally

So far you have a scaffold, not a project. In this exercise you will implement the steps you mapped out in Exercise 2 as small, composable scripts, and orchestrate them with the Makefile so the whole pipeline runs with a single command.

We will keep the ML deliberately boring — a scikit-learn classifier on a toy dataset (e.g. `sklearn.datasets.load_digits`). The point is the _plumbing_: how a notebook's linear flow becomes discrete steps with clear inputs and outputs that can be run and re-run independently.

### Exercise 3.1: One step at a time

Implement each step as a function (or small script) in your package, each reading from and writing to the directory you decided in Exercise 2. Suggested split:

1. <details><summary>Data ingestion: <code>&lt;module_name&gt;/dataset.py</code></summary>
   Download or generate the raw data and save it to <code>data/raw/</code> in a sensible format (e.g. parquet or csv).

   <pre><i><u>Why is it important that the raw data is immutable? Where would "real" data come from in a production setting?</u></i></pre>
   </details>

2. <details><summary>Data preparation: <code>&lt;module_name&gt;/features.py</code></summary>
   Read the raw data, do any cleaning/feature engineering, and write the processed training data to <code>data/processed/</code>. Include a train/test split so evaluation is honest.

   <pre><i><u>What did the lecture say belongs in the "silver" layer, and why should the split be deterministic (hint: <code>random_state</code>)?</u></i></pre>
   </details>

3. <details><summary>Model training: <code>&lt;module_name&gt;/modeling/train.py</code></summary>
   Read the processed data, fit a model, and save the serialized model to <code>models/</code>.

   <pre><i><u>The lecture mentioned several serialization/packaging formats. Which one does scikit-learn naturally give you? What are the risks of that format?</u></i></pre>
   </details>

4. <details><summary>Model evaluation: extend <code>train.py</code> or add an <code>evaluate</code> step</summary>
   Load the trained model plus the test split, compute at least two metrics, and write them to a file (e.g. <code>reports/metrics.json</code>) instead of just printing them.

   <pre><i><u>Why write metrics to a file? Consider: who or what (hint: CI) will need to read them later?</u></i></pre>
   </details>

5. <details><summary>Inference: <code>&lt;module_name&gt;/modeling/predict.py</code></summary>
   Load the saved model and a few rows of processed data, and write predictions to <code>models/predictions.csv</code>.

   <pre><i><u>This is "batch" serving. What other mode of serving did the lecture mention, and when would you use it?</u></i></pre>
   </details>

<pre><i><u>Notice the pattern: every step loads its input, does one thing, and writes its output to a predictable location. No step reads anything that an earlier step didn't write.</u></i></pre>

### Exercise 3.2: Orchestrate with the Makefile

Now wire the steps together in the `Makefile`. CCDS already gives you a starting point — extend it so that:

1. Each step has its own target, e.g.:

   ```makefile
   data:  ## Generate and process the raw data
       python -m <module_name>.dataset
       python -m <module_name>.features

   train:  ## Train and evaluate the model
       python -m <module_name>.modeling.train

   predict:  ## Run batch inference
       python -m <module_name>.modeling.predict
   ```

2. There is a target that runs the whole pipeline:

   ```makefile
   pipeline: data train predict  ## Run the entire pipeline end-to-end
   ```

3. Run it:

   ```bash
   make pipeline
   ```

   <pre><i><u>Make only re-runs a target when you ask it to. Try touching (or deleting) <code>data/raw/</code> and re-running: which steps do you need to re-run, and in which order? Does your Makefile let you run e.g. only "train" without "data"?</u></i></pre>

4. Delete everything under `data/` and `models/`, then re-run `make pipeline` from scratch.

   <pre><i><u>Would this work on a colleague's machine? What about on a fresh CI machine (hint: remember this for Exercise 5)?</u></i></pre>

### Exercise 3.3: Reflect

<pre><i><u>Discuss in pairs, comparing with the "stage 0" (notebook) workflow from the lecture:

<li>Which single file would you have to change to swap the model for another one? To swap the dataset?</li>
<li>Where is the boundary between exploration (notebooks) and repetition (source code) in your project?</li>
<li>What is still missing before you would call this MLOps level 1? (hint: automation, storage, CI)</li></u></i></pre>

[Next: Exercise 4 - Cloud Storage](cloud-storage)
