---
title: "Mean Time to Restore: Causes"
---

# Mean Time to Restore: Causes

---

## New model releases

A nasty failure mode: you release a new model with **lower loss** on your offline metrics, but your **business KPI goes down too**.

Why does that happen?

- Users build heuristics – they adapt to the model's behaviour, and a sudden change breaks their workflows
- Buggy implementation – the model is fine, the code around it is not
- Automation is not a guarantee – an automated pipeline happily ships a bad model

So: **retrain, rectify, or rollback?**

![Loss down, KPI down](/images/model_experimentation/new-model-release-kpi.png)

> "This churn model always overestimates the chance of freight forwarders stopping to be our customers"

---

## Troubleshooting

When something is off in production, where do you look?

- Bias in the model?
- Subgroup performance shift?
- Serving latency?

→ **Domain knowledge + logging** are what let you find the cause. Without good logging there is nothing to troubleshoot from, and without domain knowledge you don't know where to look.

![Troubleshooting](/images/model_experimentation/troubleshooting.png)

---

## Rollback of model

The fastest restore for a bad model release is often the simplest one:

- Promote the old model back to production
- This requires saving all model artifacts – every version of the model, its data and its configuration, must be retrievable

Note: rolling back gets you back to a working state, but it is **not solving the underlying problem**. You still need to figure out what went wrong before the next release.

![Model versions](/images/model_experimentation/model-versions.png)
