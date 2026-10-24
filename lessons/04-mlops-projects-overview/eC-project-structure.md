---
title: Project Structure
---

# Exercise 2: Inspect Project Structure

## Exercise 2.1: The scaffold

Inspect the structure of the resulting repo. It may look something like this:

- **`LICENSE`** — Open-source license if one is chosen
- **`Makefile`** — Makefile with convenience commands like `make data` or `make train`
- **`README.md`** — The top-level README for developers using this project.
- **`data/`** — Directory for storing data locally
  - **`external/`** — Data from third party sources.
  - **`interim/`** — Intermediate data that has been transformed.
  - **`processed/`** — The final, canonical data sets for modeling.
  - **`raw/`** — The original, immutable data dump.
- **`docs/`** — A default mkdocs project; see www.mkdocs.org for details
- **`models/`** — Trained and serialized models, model predictions, or model summaries
- **`notebooks/`** — Jupyter notebooks. Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`.
- **`pyproject.toml`** — Project configuration file with package metadata for `{{ cookiecutter.module_name }}` and configuration for tools like black
- **`references/`** — Data dictionaries, manuals, and all other explanatory materials.
- **`reports/`** — Generated analysis as HTML, PDF, LaTeX, etc.
  - **`figures/`** — Generated graphics and figures to be used in reporting
- **`requirements.txt`** — The requirements file for reproducing the analysis environment, e.g. generated with `pip freeze > requirements.txt`
- **`setup.cfg`** — Configuration file for flake8
- **`{{ cookiecutter.module_name }}/`** — Source code for use in this project.
  - **`__init__.py`** — Makes `{{ cookiecutter.module_name }}` a Python module
  - **`config.py`** — Store useful variables and configuration
  - **`dataset.py`** — Scripts to download or generate data
  - **`features.py`** — Code to create features for modeling
  - **`modeling/`**
    - **`__init__.py`**
    - **`predict.py`** — Code to run model inference with trained models
    - **`train.py`** — Code to train models
  - **`plots.py`** — Code to create visualizations


How does this structure make sense to you?

- Do you understand where the ML code goes?
- What does each of the pre-generated files do?
- Any terms that you don't know what they mean?
- Other confusing things?

## Exercise 2.2: Where do the MLOps steps live?

In the lecture we walked through the typical steps of an ML pipeline that consisted of a **data component**, a **model component** and a **code component**. Now it's time to put this into code.

1. Where does each step live in this CCDS structure?
   - Data extraction/ingestion
   - Data validation
   - Data preparation
   - Model training
   - Model evaluation
   - Model packaging
   - Inference/serving

2. Copy the files in https://github.com/Jeppe-T-K/ccds-dsip-exercise-repo/tree/main/pipeline_steps into your project and validate that you can run the steps.

3. Where would you put the other code parts that we've covered so far?
   - Dockerfile(s)
   - Tests
   - Workflows

You now have a mental map of the project. In the next exercise, you will try to build a pipeline for this using a single command.
