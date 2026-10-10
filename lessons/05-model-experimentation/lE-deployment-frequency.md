---
title: Deployment Frequency
---

# Deployment Frequency

*How often do we deploy to end-users?*

---

## What deployment?

For a regular software project, a deployment ships code. But in an ML system, "a deployment" can mean several things:

- Data changes (new training data, new features)
- Model changes (new parameters, new architecture)
- Code changes (feature engineering, serving logic)

![End-to-end ML workflow](/images/model_experimentation/ml-ops-org-end-to-end.png)

From https://ml-ops.org/content/end-to-end-ml-workflow

Which of these you count as "a deployment" affects how you measure the metric at all.

---

## What affects frequency?

How often *can* you ship a new model? It depends on, among other things:

- The degree of automation in your pipeline
- How quickly you can validate that a new model is safe to ship
- How quickly the underlying data changes

We covered the degree of automation in the last lecture – the more manual steps in the pipeline, the lower your effective deployment frequency.
