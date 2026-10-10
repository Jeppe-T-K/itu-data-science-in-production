---
title: Change Failure Rate
---

# Change Failure Rate

*How often does a change to production fail?*

---

## What is a failure for ML?

For regular software, a failure is usually clear: the service is down or misbehaves. For ML systems, a failure is more subtle:

![End-to-end ML workflow](/images/model_experimentation/ml-ops-org-end-to-end.png)

From https://ml-ops.org/content/end-to-end-ml-workflow

A change can "work" (no crash, predictions are served) and still be a failure if the model behind it performs worse than what it replaced.
