---
title: Introduction
---

# Model Experimentation, Selection and Monitoring

## Overview

### Agenda

 * [08:15 – 09:00] – Lecture: DORA metrics and model experimentation, selection and monitoring
 * [09:15 – 12:00] – Exercises: MLflow catch-up, custom model wrapper + FastAPI serving

### Preparation

For the exercises:

* https://MLflow.org/docs/latest/ml/ (use as reference material, don't do exercises)
* https://fastapi.tiangolo.com/ (use as reference material, don't do the tutorial)

For the lecture:

* https://www.mlebook.com/wiki/doku.php (Chapter 9)
* https://ml-ops.org/content/mlops-principles (Monitoring in particular)
* https://www.datadoghq.com/knowledge-center/dora-metrics/ (or other DORA metric google searches is fine)
* https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide (skimming it is fine)
* https://dvc.org/doc/use-cases/experiment-tracking (skimming it is fine)
* [Overview of different MLOps tools](https://arxiv.org/html/2601.20415v1) (skimming it is fine)

### Notes


## Lecture

The lecture uses the DORA metrics as the red thread for explaining why model experimentation, selection and monitoring is important:

- [Lecture Intro](lecture-intro) - Outline and the DORA metrics for MLOps
- [Working in a Team](working-in-a-team) - Working on a technical product together
- [DORA Metrics](dora-metrics) - The four delivery metrics and their shortcomings
- [DORA for MLOps](dora-for-mlops) - What the metrics mean for ML systems
- [Deployment Frequency](deployment-frequency) - What a deployment is for ML
- [Model Issues](model-issues) - Model decay, quantifying changes and drift
- [Data Issues](data-issues) - Data quality, drift and outliers
- [Logging](logging) - Log, monitor and alert for everything
- [Lead Time for Changes](lead-time-for-changes) - What affects the speed of changes
- [Experimentation](experimentation) - Tracking ML experiments with MLflow and DVC
- [In Production](putting-it-in-production) - Selecting and promoting models
- [Mean Time to Restore](mean-time-to-restore) - Restoring ML systems
- [MTTR: Causes](mttr-causes) - Troubleshooting and rolling back models
- [Change Failure Rate](change-failure-rate) - What a failure is for ML
- [Managing Failures](managing-failures) - Pilots, A/B tests and shadow testing
- [Summary](summary) - The DORA metrics revisited

## Exercises

The following exercises will guide you through model experimentation:

- [Exercise 1: MLflow Catch-Up](mlflow-basics) - Refresher on tracking ML experiments with MLflow
- [Exercise 2: A Custom Model Wrapper](custom-models) - Wrap a logistic regression with a classification threshold
- [Exercise 3: Serving with FastAPI](fastapi-serving) - Put the model behind an HTTP API
- [Exercise 4: Check Performance of Deployed Model](model-monitoring) - Monitor for data and concept drift

> **Note - Learning outcomes**
> <i> By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Organise ML experiments using common tools</li>
> <li>Wrap a model with custom inference logic (such as a classification threshold) and log it as a first-class model</li>
> <li>Motivate how this can be used to deploy models, e.g. behind a web API</li>
> <li>Explain how to detect concept and model drift</li>
> </ul>

There are 5 python files in the [resources](resources) folder for these exercises:
* [data_util.py](resources/data_util.py), which you can use for generating the data (imagine it is the output of the data processing pipelines), including classification data with drift variants
* [example_model.py](resources/example_model.py), which you can use as the basic SKLearn model
* [basic_mlflow_training.py](resources/basic_mlflow_training.py), which you can use as a guide for wrapping a model with a threshold and setting up MLflow
* [basic_fastapi_app.py](resources/basic_fastapi_app.py), which you can use as a guide for serving the model with FastAPI
* [basic_mlflow_evaluation.py](resources/basic_mlflow_evaluation.py), which you can use for evaluating the (created) models using new data and check for drifts.

As always for the most effective learning, try to use the MLflow documentation and figure out how to solve the exercises yourself first before consulting the basic MLflow scripts.
