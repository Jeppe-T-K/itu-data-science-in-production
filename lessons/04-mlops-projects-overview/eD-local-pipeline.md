---
title: Local Pipeline
---

# Exercise 3: Run the Pipeline Locally

In Exercise 2 you mapped the MLOps steps onto your project structure. In this exercise you will run that pipeline the MLOps level 1 way: with a single command.

## Exercise 3.1: Get set up

Carry your project forward from the previous exercises. Work in the repo you scaffolded in Exercise 1 and added different scripts to in Exercise 2.

<details><summary>If you haven't set up a project yet, you can use this one instead</summary>

It is a CCDS scaffold with example implementations of each pipeline stage in [`pipeline_steps/`](https://github.com/Jeppe-T-K/ccds-dsip-exercise-repo) filled into `src/`:

```bash
git clone https://github.com/Jeppe-T-K/ccds-dsip-exercise-repo.git my-ds-project
cd my-ds-project
```

</details>

Either way you should recognize the CCDS layout from Exercise 2:
- `src/dataset.py` fetches or generates the raw data
- `src/features.py` prepares it
- `src/modeling/train.py` trains a model
- `src/modeling/predict.py` runs batch inference

### Questions

- Can you run each component by itself?
    - Ideally yes. We aim for modularity, so each step should be possible to run without the other. However, the training step might require that some data has been fetched/processed.
- How do you handle python packages?
    - Depending on what you installed:
        - uv: `uv add ...`
        - requirements.txt: add e.g. `sklearn==1.3` in file
        - conda: `conda add` # AI double check this
        - etc...

## Exercise 3.2: Run the pipeline with the Makefile

For this exercise, we keep it simple and make use of the Makefile CCDS uses.

> For the project we expect you to use Go/Dagger for the pipelines. 

In the Makefile you can define how to run each step. CCDS already has a few pre-filled ones, and so you can glimpse the pattern directly:

```
.PHONY: requirements  # runs as recipe, doesn't look for file
requirements:         # make <command>, e.g. `make requirements`
    uv sync           # Command(s) to run. Can be multiline
```


You can modify the `Makefile` so each `make <command>` is easy. Create a command for each of the following:
- `data`
- `features`
- `train`
- `predict` 
- plus a target that chains them all in order (e.g. `pipeline`)


Run it, then inspect what appeared under `data/` and `models/`. Try running the individual targets too, e.g. `make train` alone. If you cloned the example repo, make sure [uv](https://docs.astral.sh/uv/) is installed - the Makefile uses it to sync dependencies.

### Questions

- How does this differ from using Dagger for creating the pipeline
    - No specification of Docker image
    - AI suggest 2 other things
- What is still missing before you would call this MLOps level 1? 
    - hint: automation and CI
