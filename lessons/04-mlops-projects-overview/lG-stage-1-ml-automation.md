---
title: "Organisation Maturity - Stage 1: Data/Model/Code Automation"
---

# Data/model/code semi-automation

<div style="text-align: center;">
  <img src="/images/mlops-projects-overview/coco-jacket.png" alt="Coco in a jacket" style="width: 50%;" />
</div>

---

<details><summary style="font-size:1.5em">Overview</summary>

![google-mlops-architecture.png](/images/mlops-projects-overview/google-mlops-architecture.png)
From https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning


- Rapid experiment
- Continuous model delivery/testing
- Modularized and re-used code
    - No training-serving skew
- Whole ML pipeline deployments

</details>

---

<details><summary style="font-size:1.5em">Dealing with breaking models</summary>

![breaking-models.png](/images/mlops-projects-overview/breaking-models.png)

- Data validation
    - Data schema skews
    - Data values skews
- Model validation
- Offline vs. online
- Evaluation metric
- “DevOps”

→ Pipeline triggers next lecture

</details>