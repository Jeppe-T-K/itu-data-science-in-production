---
title: "Code and Deployment"
---

# Code and Deployment

Recap of the "Code Component" and "Deployment Strategies" lectures.

---

## Discussion

<details><summary>Think of a machine learning model you have met in real life - as a user, at work, anywhere. What did it do, and how do you think it was served?</summary>

Examples that usually come up, and what they reveal:

- Spam filters, credit scoring, churn campaigns → often **batch**
- Recommendations, search ranking, autocomplete → usually **on demand**
- Phone keyboards, face unlock → on-device serving

- Serving to humans or machines? Sanity checks? Fallback? Interpretation outside the model?

</details>

---

## Questions

Your model is served from a serverless function. What happens on a cold start?

- A: The first prediction returns instantly, same as any other prediction
- B: The function refuses to start until a container is provisioned manually
- C: The runtime and model must boot and load first, typically taking hundreds of milliseconds up to several seconds
- D: Nothing - serverless functions are always warm once created

<details><summary>Answer</summary>

**C.** With no running instance, the first call pays for booting the runtime and loading the model - hundreds of milliseconds to seconds depending on the model size.

A and D ignore that "no running instance" is the whole point of serverless. B is wrong - provisioning is automatic, it is just *slow* the first time.

→ The trade-off: serverless does not require a running instance, but costs more per inference

</details>

---

What are the two modes of model serving?

- A: In Docker and in Kubernetes
- B: In batch and on demand
- C: Online and offline learning
- D: Training and testing

<details><summary>Answer</summary>

**B.** Predictions are either computed in bulk ahead of time (batch) or computed when requested (on demand/online/real-time).

A lists deployment technologies, not serving modes. C is about *learning* modes, not serving. D are model development phases.

![model-serving.png](/images/mlops-projects-overview/batch-vs-realtime-serving.svg)

</details>

---

Why log predictions and monitor a model in production? Select all that apply.

- A: Performance deteriorates over time
- B: To increase the parameter count
- C: Data changes
- D: Because the notebook told you to
- E: Abuse/adversarial inputs

<details><summary>Answer</summary>

**A, C, and E.** Models decay, the world moves under them, and people probe them - logging predictions is how you see any of it happening.

B is meaningless - logging does not change the model. D is not a reason.

![grafana-monitoring.gif](/images/mlops-projects-overview/grafana-monitoring.gif)

</details>

---

What goes into model maintenance decisions? Select all that apply.

- A: Newer frameworks
- B: Model freshness
- C: Personal preference
- D: Retraining time
- E: Deployment cost

<details><summary>Answer</summary>

**B, D, and E.** How stale the model is, how long a retrain takes, and what a redeployment costs together determine when and how often to retrain.

C may happen in practice, but it is not a criterion to teach. If you want to be structured about it, you could run A/B tests and check if it improves your primary metric (such as clickthrough rate or like-ratio).

A is not a maintenance decision.

</details>

---

Docker containers vs. serverless functions for serving - which statement is true?

- A: Serverless is always cheaper and faster than containers at any scale
- B: Serverless does not require a running instance, but is more expensive per inference; containers keep a lightweight instance running
- C: Containers cannot serve models; they are only for training
- D: Serverless functions require you to manage your own servers

<details><summary>Answer</summary>

**B.** That is the trade-off: serverless scales to zero and you pay per inference, containers keep the model warm with predictable latency.

A is wrong in both directions - per inference, serverless is the more expensive option. C is wrong - containers are the standard way to serve models. D contradicts the definition of serverless.

</details>

---

What pattern does the course project follow?

- A: Real-time training, offline serving
- B: Batch training, batch serving of a single model
- C: No serving; the model is only evaluated in a notebook
- D: Offline training, on-demand serving through a REST API

<details><summary>Answer</summary>

**D.** The model is trained offline in the pipeline, and served on demand through a REST API.

A flips both - training is not real-time and serving is not offline. B describes a batch-only setup without an API. C describes Level 0, and you are building a pipeline and serving.

</details>

---

Given the exercises - Cookiecutter project structure, a local pipeline, CI/CD, and cloud storage - which maturity level is the course project itself?

- A: Level 2 - Fully automated
- B: Level 0 - No MLOps/manual process
- C: Level 1 - Semi-automated data/model/code pipelines

<details><summary>Answer</summary>

**C.** The pipeline and the CI/CD for the code are automated, but a human decides when a new model is worth deploying.

A is wrong - there is no pipeline-triggered retraining; that jump comes in the next lessons. B is wrong - there is an automated pipeline, not a manual notebook-only process.

</details>
