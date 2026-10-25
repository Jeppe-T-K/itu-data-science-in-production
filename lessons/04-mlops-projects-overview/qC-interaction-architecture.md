---
title: "Architecture and Stages"
---

# Architecture and Stages

Recap of the maturity stage lectures (Level 0, 1, and 2).

---

## Discussion

![google-stage-0-manual-ml.png](/images/mlops-projects-overview/google-stage-0-manual-ml.png)
From https://docs.cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

This is the full ML architecture of a real company. What problems will they run into?

<details><summary>Discussion points</summary>

- Interactive process (e.g., notebooks) - YOU deploy the model

- DS - engineer disconnect → training-serving skew

- No CI/CD, no monitoring

→ This is the Level 0 "manual process" architecture

</details>

---

## Questions

![google-mlops-architecture.png](/images/mlops-projects-overview/google-mlops-architecture.png)
From https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning

What maturity level is the organisation behind this architecture?

- A: Level 2 - Fully automated
- B: Level 0 - Manual process
- C: Impossible to tell from an architecture diagram
- D: Level 1 - ML pipeline automation

<details><summary>Answer</summary>

**D.** An automated ML pipeline enables rapid experimentation, continuous model delivery and testing, and modularized, re-used code → no training-serving skew.

B is wrong because the training pipeline clearly exists as a deployable artifact. A is wrong because there is no CI/CD that automatically builds, tests, and deploys new pipelines in response to new data or code. C is wrong - these diagrams map directly onto the maturity levels.

</details>

---

![pulse-cicd.png](/images/mlops-projects-overview/pulse-cicd.png)
From https://www.linkedin.com/pulse/cicd-what-why-important-rob-larter/

At Level 2, what do "continuous integration" and "continuous delivery" apply to?

- A: Only the application code, like classic DevOps
- B: The pipelines: continuous integration of pipelines, continuous delivery of pipelines and models
- C: The code and the models, but not the pipelines
- D: Nothing in particular; CI/CD is not a maturity question

<details><summary>Answer</summary>

**B.** In Level 2, the pipelines are the artifact under CI/CD: new data or new code automatically triggers building, testing, and deployment of new pipeline versions and models.

A is classic DevOps for application code; it is not what distinguishes Level 2. C is close, but it misses that the *pipelines* are what gets integrated and delivered. D is wrong - CI/CD for pipelines is exactly what distinguishes Level 2.

</details>

---

Which of the following are true of a Level 0 organisation? Select all that apply.

- A: A person manually deploys the model
- B: Data validation automatically triggers retraining
- C: Model training happens interactively, e.g. in notebooks
- D: Modularized pipeline components are shared between training and serving
- E: No CI/CD
- F: No monitoring

<details><summary>Answer</summary>

**A, C, E, and F.** That is the manual-process level: notebook-driven training, human deployment, no CI, no monitoring.

B is a Level 2 trait - production data automatically triggers retraining. D is a Level 1 trait - it is exactly what removes the training-serving skew.

</details>

---

What is "training-serving skew"?

- A: The model trains slower than it serves, so training lags behind production
- B: Serving the model skews the training data towards recent examples
- C: The model behaves differently in production than during training, because the training pipeline and the serving pipeline were built separately
- D: A skew statistic above 1 between the train and the test loss, used to detect overfitting

<details><summary>Answer</summary>

**C.** When the people who train and the people who serve build two different implementations of the same logic, behaviour drifts apart.

A is about speed, not behaviour. B reverses the causality - serving does not touch training data. D invents a statistic; no such metric is part of the term.

</details>

---

A model in production keeps degrading and breaking. Which mechanisms would you put in place to catch and fix it? Select all that apply.

- A: Retrain manually whenever a customer complains
- B: Data validation - data schema skews and data value skews
- C: Nothing - models are finished once deployed
- D: Model validation - offline vs. online metrics
- E: Pipeline triggers for automated retraining

<details><summary>Answer</summary>

**B, D, and E.** Validate incoming data, validate the model offline and online, and let the pipeline trigger retraining (next lecture).

A is reactive - by the time a customer complains, the damage is done. C is the Level 0 attitude, and decay compounds over time.

![breaking-models.png](/images/mlops-projects-overview/breaking-models.png)

</details>
