---
title: "Deployment Frequency: Data Issues"
---

# Deployment Frequency: Data Issues

It is not only the model that can cause trouble – the data flowing into it can too.

---

## Data issues

Three main categories:

- **Data quality issues** – the data is wrong
- **Data/feature drift** – the data distribution changes over time
- **Outliers** – individual extreme values

![Data drift](/images/model_experimentation/evidently-data-drift.png)

From https://www.evidentlyai.com/ml-in-production/data-drift

---

<details><summary style="font-size: 1.5em"> Data quality</summary>

The list of possible data quality issues is long… for example:

- Data source changes
- Source data schema changes
- Data loss
- …

Common checks to catch them:

- Schema comparison
- NaNs, min/max, etc.

![Data quality](/images/model_experimentation/data-quality.png)

</details>

---

<details><summary style="font-size: 1.5em"> Data/feature drift</summary>

The distribution of your data changes over time.

Question: do you compare against the source data distribution or the feature distribution?

Statistical tests to detect it:

- **Kolmogorov-Smirnov test** – i.e., the max distance between the cumulative distributions
- **Chi-squared** for categorical features

![Data drift distributions](/images/model_experimentation/datajello-data-drift.png)

From https://datajello.com/production-ml-data-drift-and-concept-drift/

![Kolmogorov-Smirnov example](/images/model_experimentation/kolmogorov-smirnov-example.png)

From https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test#/media/File:KS_Example.png

</details>

---

<details><summary style="font-size: 1.5em"> Outliers</summary>

Where do outliers come from?

- Bots / adversarial attacks?
- NaN / infinities?
- Faulty data entries?

→ The real question: **when and how to react?**

![Faulty sensor data leading to outliers](/images/model_experimentation/outlier-sensor-data.png)

</details>
