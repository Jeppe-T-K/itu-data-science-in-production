---
title: DORA Metrics
---

# DORA Metrics

**DORA** is the Google team **D**ev**O**ps **R**esearch and **A**ssessment, who publish the yearly "State of DevOps" reports (since ~2013).

> **Note — DORA**
> DORA is not an abbreviation of the metrics themselves – it is the name of the research team behind them.

![DevOps infinity loop](/images/model_experimentation/dynatrace-devops.png)

---

## Overview

Four metrics for measuring success/performance of software delivery:

- 2 "dev" metrics
- 2 "ops" metrics

Do you have any ideas for what they could be?

![MLOps principles](/images/model_experimentation/ml-ops-org-circles.png)

From https://ml-ops.org/content/mlops-principles

---

<details><summary style="font-size: 1.5em"> Deployment frequency</summary>

How often you deploy code to end-users.

- Smaller bits > large chunks
- Typically ~1/week up to daily

</details>

---

<details><summary style="font-size: 1.5em"> Lead time for changes</summary>

The time from commit to production.

Highlights inefficient processes:

- Slow code review?
- (Too) complex code?

From 1 week/month down to <1 day.

</details>

---

<details><summary style="font-size: 1.5em"> Mean time to restore</summary>

How quickly you recover when something breaks: fix bugs → new code → update.

You don't want bugs in production…

From months down to <1 hour.

</details>

---

<details><summary style="font-size: 1.5em"> Change failure rate</summary>

How much of the deployed code is "faulty" production code.

- Nothing is perfect
- But a high rate → low trust

From ~0% up to +60%.

</details>

---

## Shortcomings

<details><summary style="font-size: 1.2em"> Goodhart's law</summary>

*"When a measure becomes a target, it ceases to be a good measure."* Optimising for the metric is not the same as optimising for good software delivery.

</details>

<details><summary style="font-size: 1.2em"> Don't compare teams</summary>

The metrics are context specific – comparing teams against each other on DORA metrics is a misuse.

</details>

<details><summary style="font-size: 1.2em"> Simplification</summary>

Four metrics is a simplification – DORA themselves track 24 other metrics in their reports.

</details>

![xkcd 2899: outdated metrics](/images/model_experimentation/xkcd-2899-metrics.png)

> "I'm pleased to report we're now identifying and replacing hundreds of outdated metrics per hour."

From https://xkcd.com/2899/
