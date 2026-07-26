# Model Experimentation, Selection and Monitoring

## Overview

### Agenda

 * [08:15 – 10:00] – Exercises: MLflow + warnings
 * [10:15 – 12:00] – Lecture: DORA metrics and model experimentation, selection and monitoring

### Preparation

For the exercises:

* https://MLflow.org/docs/latest/ml/ (use as reference material, don't do exercises)

For the lecture:

* https://www.mlebook.com/wiki/doku.php (Chapter 9.3 + 9.4 + 9.5)
* https://ml-ops.org/content/mlops-principles (Monitoring in particular)
* https://www.datadoghq.com/knowledge-center/dora-metrics/ (or other DORA metric google searches is fine)
* https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide (skimming it is fine)
* https://dvc.org/doc/use-cases/experiment-tracking (skimming it is fine)

### Notes

* Have fun!

## Slides

Slides are available on [LearnIT](https://learnit.itu.dk/).

## Exercises

The following exercises will guide you through model experimentation:

- [Exercise 0: Installation](B-mlflow-basics.md#exercise-0-installation) - Install MLflow and dependencies
- [Exercise 1: Run an ML Experiment](B-mlflow-basics.md) - Create and log ML experiments with MLflow
- [Exercise 2: Running a Custom Model](C-custom-models.md) - Define and use custom models
- [Exercise 3: Check Performance of Deployed Model](D-model-monitoring.md) - Monitor for data and concept drift

## Learning Goals

> [!NOTE]
> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Organise ML experiments using common tools</li>
> <li>Motivate how this can be used to deploy models</li>
> <li>Explain how to detect concept and model drift</li>
> </ul>

There are 4 python files in this directory for these exercises:
* [data_util.py](data_util.py), which you can use for generating the data (imagine it is the output of the data processing pipelines)
* [example_model.py](example_model.py), which you can use as the basic SKLearn model
* [basic_mlflow_training.py](basic_mlflow_training.py), which you can use to guide you on how to create a custom model/function and set up MLflow
* [basic_mlflow_evaluation.py](basic_mlflow_evaluation.py), which you can use for evaluating the (created) models using new data and check for drifts.

As always for the most effective learning, try to use the MLflow documentation and figure out how to solve the exercises yourself first before consulting the basic MLflow scripts.
