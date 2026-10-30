---
title: Exercise Intro
---

# Background

The lecture used the DORA metrics as the red thread for why model experimentation, selection and monitoring matter. Now we make it concrete: you will take a small model through the whole journey - track it, wrap it, serve it, and monitor it.

An ML experiment produces more than a model: it produces parameters, metrics, artifacts and (crucially) the *decision* about which configuration was best. Without a tool to record all of that, experiments live in ad-hoc scripts and spreadsheets: results are not comparable, not reproducible, and the model that ends up in production is "the one Jeppe ran last Tuesday". The tool we use for that is [MLflow](https://mlflow.org/), which covers experiment tracking as well as a model registry and model serving; the lecture also mentioned alternatives such as Weights & Biases, Neptune and DVC.

Throughout the course you are building a real project (the [itu-sdse-project](https://github.com/lasselundstenjensen/itu-sdse-project) repository) whose [notebooks](https://github.com/lasselundstenjensen/itu-sdse-project/tree/main/notebooks) train a CNN on images of glass vials, register it in MLflow, and serve it behind an API. These exercises deliberately use a small tabular logistic regression instead - so you can see clearly what happens at each step - but the flow is the same one you will meet in the project: train, tune the classification threshold, wrap the model so the threshold travels with it, register it, serve it, and watch for drift.

## Learning Goals

> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Organise and compare ML experiments using MLflow (tracking and the model registry)</li>
> <li>Wrap a model with custom inference logic (such as a classification threshold) and log it as a first-class model</li>
> <li>Serve a registered model, both with MLflow's scoring server and behind your own FastAPI API</li>
> <li>Explain how to detect data drift and concept drift in a deployed model</li>
> </ul>

## Exercises

The following exercises will guide you through model experimentation:

- [Exercise 1: MLflow Catch-Up](mlflow-basics) - Track ML experiments, register models, and select the best run
- [Exercise 2: A Custom Model Wrapper](custom-models) - Wrap a logistic regression with a classification threshold
- [Exercise 3: Serving the Model](fastapi-serving) - Serve the registered model directly with MLflow, then wrap it in your own FastAPI app
- [Exercise 4: Check Performance of Deployed Model](model-monitoring) - Monitor for data and concept drift

There are 5 python files in the [resources](resources) folder for these exercises:
* [data_util.py](resources/data_util.py), which you can use for generating the data (imagine it is the output of the data processing pipelines), including classification data with a train/validation/test split and drift variants
* [example_model.py](resources/example_model.py), which you can use as the basic SKLearn model
* [basic_mlflow_training.py](resources/basic_mlflow_training.py), which you can use as a guide for wrapping a model with a threshold and setting up MLflow
* [basic_fastapi_app.py](resources/basic_fastapi_app.py), which you can use as a guide for serving the model with FastAPI
* [basic_mlflow_evaluation.py](resources/basic_mlflow_evaluation.py), which you can use for evaluating the (created) models using new data and check for drifts.

As always for the most effective learning, try to use the MLflow documentation and figure out how to solve the exercises yourself first before consulting the basic MLflow scripts.

---

# Exercise 0: Setup

Make sure MLflow and friends are installed:
`pip install mlflow`
`pip install scipy`
`pip install scikit-learn`

Try running it with:
`mlflow --version`

Does this command run and show a version number of 2.18 or newer? Great! (FastAPI and uvicorn arrive with Exercise 3, so don't worry about them yet.)
