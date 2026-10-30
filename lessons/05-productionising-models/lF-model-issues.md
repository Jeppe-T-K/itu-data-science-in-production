---
title: "Deployment Frequency: Model Issues"
---

# Deployment Frequency: Model Issues

---

## Model decay

A deployed model does not stay good forever – models decay over time, and the causes are many:

- Changes in dependencies
- New training data → worse model
- Different feature extraction
- Code updates
- …

![Model decay](/images/model_experimentation/model-decay.png)

From https://ml-ops.org/content/mlops-principles

---

## Quantifying changes

When the model or the world changes, how do we quantify it? Common model metrics to track:

- Training loss
- Accuracy / F1-score / ROC AUC
- Prediction bias
- Threshold ↔ business value (where you set the threshold determines the business trade-off)

![Quantifying changes](/images/model_experimentation/quantifying-changes.png)

---

## Concept/model drift

Drift is when the relationship between features and output changes – e.g., age ↔ music preference is not a stable relationship over time.

It can be:

- **Gradual** – slowly changing over time
- **Sudden** – a step change in the relationship
- **Recurring** – drifting back and forth, e.g. seasonally

![Concept drift types](/images/model_experimentation/concept-drift-types.png)

Example: 30% week-over-week change. Fig. 3, from https://link.springer.com/article/10.1007/s11002-022-09626-7
