---
title: Lead Time for Changes
---

# Lead Time for Changes

*How long does it take from a change is committed until it runs in production?*

---

## What changes?

Just like with deployment frequency, "a change" in an ML system can be several things:

- Changes to the data or features
- Changes to the model
- Changes to the code

![End-to-end ML workflow](/images/model_experimentation/ml-ops-org-end-to-end.png)

From https://ml-ops.org/content/end-to-end-ml-workflow

---

## What affects the speed?

What makes the lead time from idea to production long or short?

- **Manual steps** – every hand-off adds waiting time
- **Exploration phase** – finding the right model takes experiments, not just engineering
- **Training time** – the model itself can take a long time to (re)train

The exploration phase is the ML-specific one – and it is what experiment tracking tools are designed to speed up. That is the topic of the next pages.
