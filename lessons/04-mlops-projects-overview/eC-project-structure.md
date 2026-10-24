---
title: Project Structure
---

# Exercise 2: Inspect Project Structure

### Exercise 2.1: The scaffold

Inspect the structure of the resulting repo:

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── {{ cookiecutter.module_name }}   <- Source code for use in this project.
   │
   ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
   │
   ├── config.py               <- Store useful variables and configuration
   │
   ├── dataset.py              <- Scripts to download or generate data
   │
   ├── features.py             <- Code to create features for modeling
   │
   ├── modeling
   │   ├── __init__.py
   │   ├── predict.py          <- Code to run model inference with trained models
   │   └── train.py           <- Code to train models
   │
   └── plots.py                <- Code to create visualizations
```

<pre><i><u>Discuss in pairs how this structure makes sense to you</u>

<li>Do you understand where the ML code goes?</li>
<li>What does each of the pre-generated files do?</li>
<li>Any terms that you don't know what they mean?</li>
<li>Other confusing things?</li></i></pre>

### Exercise 2.2: Where do the MLOps steps live?

In the lecture we walked through the typical steps of an ML pipeline, mirroring Google's MLOps maturity levels. For the data component: extraction/ingestion, exploration and validation, and preparation/wrangling. For the model component: training, evaluation/testing, and validation/packaging. Finally the code component: serving, logging/monitoring, and maintenance.

Now map those steps onto your freshly scaffolded project:

1. <details><summary>Map each step to a directory</summary>
   For each step below, note _where_ its input and output should live, and _which file_ in <code>&lt;module_name&gt;/</code> should implement it:

   <table>
   <tr><th>Step</th><th>Input (from)</th><th>Output (to)</th><th>Implemented in</th></tr>
   <tr><td>Data extraction/ingestion</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Data validation</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Data preparation</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Model training</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Model evaluation</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Model packaging</td><td>?</td><td>?</td><td>?</td></tr>
   <tr><td>Inference/serving</td><td>?</td><td>?</td><td>?</td></tr>
   </table>

   <pre><i><u>Notice how the <code>data/</code> directory (raw → interim → processed) already encodes the data steps, while <code>modeling/</code> encodes the model steps. The mapping is not accidental!</u></i></pre>
   </details>

2. <details><summary>Discuss: what makes this a pipeline?</summary>
   <pre><i><u>In pairs: each step consumes the output of a previous step and produces something the next step consumes. What does that imply about...

<li>how you should design the inputs/outputs of each script?</li>
<li>what happens if you change one step — which downstream steps must be re-run?</li>
<li>why you should <b>not</b> commit data and models to git? (check .gitignore)</li></u></i></pre>
   </details>

3. <details><summary>Look at the Makefile</summary>
   Open the generated <code>Makefile</code>. CCDS pre-populates it with targets such as <code>data</code>, <code>train</code>, and <code>test</code>.

   <pre><i><u>What is a Make target, and why is a Makefile a convenient way to orchestrate the steps of your pipeline?</u></i></pre>
   </details>

You now have a mental map of the project. In the next exercise, we make it real: implementing each step so the whole pipeline runs with a single command.

[Next: Exercise 3 - Local Pipeline](local-pipeline)
