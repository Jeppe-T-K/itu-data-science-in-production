---
title: "Organisation Maturity"
---

# Organisation Maturity

Recap of "Organisation Maturity - Overview" and the maturity stages.

---

## Discussion

<details><summary>How would you assess the MLOps maturity of a company? Can you mention any specific examples?</summary>

Typical things to look for, mapping to the generic breakdown:

- Is training a manual, interactive process or an automated pipeline?
- Who deploys the model - a person or a pipeline?
- Is there CI/CD, versioning, monitoring?

- **Level 0**: No MLOps/manual process

- **Level 1**: Semi-automated data/model/code pipelines

- **Level 2**: Fully automated

</details>

---

## Questions

CocoAnalytics is a four-person fintech startup. One data scientist trains a fraud-detection model in a Jupyter notebook on a company laptop, using data exported manually from the bank. The trained model is a pickle file that is emailed to the backend developer, who wires it into the API by hand. Nobody looks at the model again after release - the first sign of trouble is a customer complaint.

What maturity level is CocoAnalytics at?

- A: Level 1 - Semi-automated data/model/code pipelines
- B: Level 2 - Fully automated
- C: Level 0 - No MLOps/manual process

<details><summary>Answer</summary>

**C.** Interactive (notebook) training, a person deploys the model, DS-engineer disconnect → risk of training-serving skew, no CI/CD, no monitoring.

A is wrong because nothing at all is automated - no pipeline exists. B is wrong because there is no automation and no monitoring loop.

</details>

---

StreamLine Logistics, a mid-size courier company, versions data with DVC. A scheduled pipeline runs weekly: it pulls fresh data, validates it, retrains the model, and logs everything to MLflow. When a new model beats the current one, the team reviews the run together, and a developer manually deploys the new model container on Monday morning. There is a Grafana dashboard, but deciding _whether_ to retrain is always a human decision.

What maturity level is StreamLine Logistics at?

- A: Level 1 - Semi-automated data/model/code pipelines
- B: Level 0 - No MLOps/manual process
- C: Level 2 - Fully automated

<details><summary>Answer</summary>

**A.** The automated ML pipeline gives rapid experimentation, continuous training and testing, and modularized code → no training-serving skew.

B is wrong because the training pipeline is clearly automated. C is wrong because humans deploy the model and decide on retraining.

</details>

---

Nordbank, a large bank, runs its credit-risk models on a fully automated loop: drift detected in production automatically triggers data validation, retraining, and model validation; if the new model passes, it is deployed as a canary without anyone touching it. Everything is monitored in Grafana, and there is an end-to-end audit trail for every model.

What maturity level is Nordbank at?

- A: Level 1 - Semi-automated data/model/code pipelines
- B: Level 0 - No MLOps/manual process
- C: Level 2 - Fully automated

<details><summary>Answer</summary>

**C.** CI/CD for the ML pipelines themselves: production data and model changes automatically trigger pipeline building, testing, and deployment.

A and B are wrong because there is no human in the loop - retraining, validation, and deployment all happen automatically.

</details>

---

TechStyle, an e-commerce company, has excellent DevOps: GitHub Actions, Docker, and Kubernetes run the webshop with smooth CI/CD. Machine learning is handled by one data scientist who trains a churn model in a notebook twice a year; the model is manually plugged into a nightly batch job.

What maturity level is TechStyle at?

- A: Level 0, no matter which framework you use
- B: It depends on the framework: Level 0 by the generic breakdown, but "DevOps but no MLOps" (their Level 1) in Microsoft's
- C: Level 2, because their DevOps is excellent
- D: Level 1, no matter which framework you use

<details><summary>Answer</summary>

**B.** The generic breakdown only looks at the ML pipelines, which are manual → Level 0. Microsoft explicitly has a level for this situation: "Level 1: DevOps but No MLOps".

A and D are each too absolute - the frameworks genuinely disagree about where "some DevOps, no MLOps" lands. C is wrong because DevOps maturity does not transfer to ML.

<div style="overflow-x: auto; margin: 0 auto; width: fit-content;">

| Company/Paper | Maturity Level Proposed(s) |
|---------------|----------------------------|
| Google | Level 0: Manual Process<br>Level 1: ML Pipeline Automation<br>Level 2: Automated Training |
| Microsoft | Level 0: No MLOps<br>Level 1: DevOps but No MLOps<br>Level 2: Automated Training<br>Level 3: Automated Model Deployment<br>Level 4: Full MLOps Automated Operations |
| Amazon | Level 1: Initial Phase<br>Level 2: Repeatable Phase<br>Level 3: Reliable Phase<br>Level 4: Scalable Phase |
| IBM | Level 0: No MLOps<br>Level 1: ML Pipeline Automation<br>Level 2: CI/CD Pipeline Integration<br>Level 3: Advanced MLOps |
| Meenu Mary et al. | Level A: Automated Data Collection<br>Level B: Automated Model Deployment<br>Level C: Semi-automated Model Monitoring<br>Level D: Fully-automated Model Monitoring |

</div>

From _Navigating MLOps: Insights into Maturity, Lifecycle, Tools, and Careers_ (https://arxiv.org/html/2503.15577v1)

</details>
