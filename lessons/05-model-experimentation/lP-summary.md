---
title: Summary
---

# Summary

We used the **DORA metrics for MLOps** as the red thread for why model experimentation, selection and monitoring matters:

<details><summary style="font-size: 1.2em"> Deployment frequency</summary>

For ML, a deployment involves data, model and code – and the frequency of each is affected by automation, validation speed and data cadence. Model decay, data quality issues, drift and outliers all push you to deploy new models; logging and monitoring tell you when.

</details>

<details><summary style="font-size: 1.2em"> Lead time for changes</summary>

The exploration phase is the ML-specific bottleneck. Experiment tracking (MLflow, DVC, …) turns scattered notebooks into comparable experiments with saved artifacts, and staging lets you promote the best model – remembering that recent is not the same as better.

</details>

<details><summary style="font-size: 1.2em"> Mean time to restore</summary>

Failures in ML are subtle: a new model can have lower loss but worse business KPIs. Domain knowledge plus logging is how you troubleshoot, and saving all model artifacts is what makes a rollback possible.

</details>

<details><summary style="font-size: 1.2em"> Change failure rate</summary>

Any change carries a risk. Pilot studies, A/B tests and shadow testing let you roll out partly and contain the blast radius of a bad release.

</details>

---

## MLOps?

![MLOps principles](/images/model_experimentation/ml-ops-org-circles.png)

From https://ml-ops.org/content/mlops-principles

Experimentation, selection and monitoring are not separate from the rest of MLOps – they are the practices that keep the four metrics healthy in an ML system.
