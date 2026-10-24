---
title: CI/CD
---

# Exercise 5: Automate with GitHub Actions (CI/CD)

Right now, your pipeline runs when _you_ remember to run it. In the lecture's maturity levels, that is stage 1 at best. The defining trait of stage 2 is that every change to the code is automatically built, tested, and delivered - continuous integration and continuous delivery.

GitHub Actions gives you CI/CD for free: a workflow file in your repo describes what to do on every push or pull request, and GitHub runs it on a fresh virtual machine. In this exercise you will add tests to your project and a workflow that runs them - and produces your model artifact automatically.

### Exercise 5.1: Write tests first

CI is only as good as what it checks, so you need something for it to run. Look in the `tests/` directory of your CCDS scaffold.

1. <details><summary>Add a test for one pipeline step</summary>
   Write a pytest test for one of your pure functions from Exercise 3 - e.g. the data preparation function. Give it a tiny, fixed input and assert on the output. Run it locally:

   <pre><code>pytest tests/</code></pre>

   <pre><i><u>Why do we test the individual steps rather than "the pipeline" as one big test? Think about how long a full pipeline takes and how hard a failure is to debug.</u></i></pre>
   </details>

2. <details><summary>Add a smoke test for the model</summary>
   Add a test that loads the trained model (or trains it on a tiny dataset) and checks that it performs better than a trivial baseline on a few samples.

   <pre><i><u>This mirrors the lecture's point that an evaluation "must be better than baseline". How could a test like this stop a broken model from ever reaching production?</u></i></pre>
   </details>

### Exercise 5.2: Your first workflow

1. <details><summary>Create the workflow file</summary>
   Create <code>.github/workflows/ci.yml</code>:

   <pre><code>name: CI

   on:
     push:
       branches: [main]
     pull_request:

   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-python@v5
           with:
             python-version: "3.12"
         - name: Install dependencies
           run: pip install -e ".[dev]"
         - name: Run tests
           run: pytest tests/ -v</code></pre>

   Commit it, push, and watch the run under the <b>Actions</b> tab on GitHub.

   <pre><i><u>Look at the workflow syntax: what do <code>on</code>, <code>jobs</code>, and <code>steps</code> each mean?</u></i></pre>
   </details>

2. <details><summary>Break it on purpose</summary>
   Push a commit where a test fails (or just rename a function your test imports). Watch CI catch it.

   <pre><i><u>What happened on the pull request page? Would you want GitHub to <i>prevent</i> merging when checks fail? Find out what branch protection rules are.</u></i></pre>
   </details>

3. <details><summary>Speed it up</summary>
   Configure dependency caching (e.g. <code>cache: "pip"</code> on <code>setup-python</code>) and time the difference.

   <pre><i><u>Why does installation dominate the runtime? What else could you cache?</u></i></pre>
   </details>

### Exercise 5.3: Produce the model artifact in CI

Continuous _delivery_ means the pipeline's output is produced automatically, not just its checks.

1. <details><summary>Run the pipeline as a job</summary>
   Add a job (or extend the existing one) that runs <code>make pipeline</code> after the tests pass. Remember: the runner is a fresh machine - nothing in <code>data/</code> or <code>models/</code> exists until your pipeline creates it. That is the point of Exercise 4's question about fresh clones.
   </details>

2. <details><summary>Publish the artifact</summary>
   Upload the trained model and metrics with <a href="https://github.com/actions/upload-artifact"><code>actions/upload-artifact</code></a>:

   <pre><code>- uses: actions/upload-artifact@v4
   with:
     name: model
     path: models/</code></pre>

   Once the run finishes, download the artifact from the run page. This is exactly how trained models are handed from a training workflow to whatever comes next: a validator, a deployment job, or a colleague.
   </details>

3. <details><summary>Optional: use your bucket from CI</summary>
   If you did Exercise 4 with a real cloud provider (or are feeling ambitious): GitHub can hold secrets too. Add your S3 credentials under <b>Settings → Secrets and variables → Actions</b>, map them into the job with <code>env:</code>, and let the CI pipeline upload the model artifact to the bucket instead of the runner.

   <pre><i><u>Why is it OK for a secret to live in GitHub's secret store, but not in the repository?</u></i></pre>
   </details>

### Exercise 5.4: Reflect

<pre><i><u>Look back at the whole session, and place your project on Google's MLOps maturity ladder:

<li>Stage 0 (notebooks, manual): where exactly did you leave that behind?</li>
<li>Stage 1 (pipeline, re-runnable): which parts of your project give you that today?</li>
<li>Stage 2 (CI/CD automation): which parts do you have, and which are missing?</li>
<li>What remains before your model actually <i>serves</i> predictions to someone - and which of the coming lessons covers that?</li></u></i></pre>

[Back to Introduction](exercise-intro)
