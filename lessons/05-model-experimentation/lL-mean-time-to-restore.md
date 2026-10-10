---
title: Mean Time to Restore
---

# Mean Time to Restore

*When something breaks in production – how long until it is fixed?*

---

## What does "restore" mean for ML?

For regular software, restoring means deploying a fix for a bug. For ML systems, an incident can be many things:

- A broken pipeline
- A model that suddenly performs poorly
- Bad or missing data reaching the model
- A serving issue

![End-to-end ML workflow](/images/model_experimentation/ml-ops-org-end-to-end.png)

From https://ml-ops.org/content/end-to-end-ml-workflow

The end-to-end workflow gives you many places to look when something is wrong – and, importantly, more ways to recover: fix the code, fix the data, or swap the model.
