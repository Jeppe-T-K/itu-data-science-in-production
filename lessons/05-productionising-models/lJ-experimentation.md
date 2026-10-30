---
title: "Lead Time for Changes: Experimentation"
---

# Lead Time for Changes: Experimentation

The exploration phase is where data scientists try out ideas: new features, other model types, different hyperparameters. Without structure, this becomes a folder of untracked notebooks called `final_v2_REALLY_FINAL.ipynb`.

---

## Hierarchy of experiments

Tracking your experiments in a hierarchy keeps the exploration phase manageable:

- **Experiment** – the overall question you are investigating
- **Run** – one training attempt with specific parameters
- **Artifacts** – the model, metrics, plots, and configuration logged for each run

![MLflow hierarchy of experiments](/images/model_experimentation/mlflow-experiments-hierarchy.png)

From https://mlflow.org/docs/latest/ml/getting-started/logging-first-model/step3-create-experiment/

![MLflow experiments view](/images/model_experimentation/mlflow-experiments-view.png)

---

## Exploration

What do we actually explore?

- New features/columns?
- Different output?
- Other kind of model?

→ Each of these becomes an **ML experiment**

![Exploring experiments](/images/model_experimentation/mlflow-exploration.png)

---

## Running an ML experiment – in DVC + VSCode

Experiment tracking is not unique to MLflow – DVC has its own experiment management, for example through the VSCode extension:

![DVC experiments in VSCode](/images/model_experimentation/dvc-experiments-vscode.png)

From https://dvc.org/doc/user-guide/experiment-management/comparing-experiments?tab=VSCode-Extension

---

## MLflow? Demo time

We saw the basics of MLflow in the exercises this morning. In the lecture we do a quick demo of comparing experiments and models – and you will get to try it yourself in [the exercises](mlflow-basics).
