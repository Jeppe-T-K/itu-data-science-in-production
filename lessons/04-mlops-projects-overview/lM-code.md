---
title: Code Component
---

# Code Component

---

## Different use-cases, different patterns


![ml-ops-org-serving-matrix.png](/images/mlops-projects-overview/ml-ops-org-serving-matrix.png)
From https://ml-ops.org/content/three-levels-of-ml-software

---

## Model serving

![model-serving.png](/images/mlops-projects-overview/batch-vs-realtime-serving.svg)

Adapted from https://www.iguazio.com/glossary/model-serving-pipeline/

- First training, then "inference"

- Modes of serving
    - In batch
    - On demand

- Other consideration
    - Sanity/confidence check input/output
    - Serving to humans or machines?
    - Fallback methods
    - Interpreting output outside of model

---

## Model logging/monitoring

![grafana-monitoring.gif](/images/mlops-projects-overview/grafana-monitoring.gif)
From https://grafana.com/blog/2023/08/18/monitoring-machine-learning-models-in-production-with-grafana-and-clearml/


- Log predictions

- Performance deteriorates

- Data changes

- Abuse/adversarial

→ More next lesson


---

## Model maintenance


- Model freshness

- Retraining time

- Deployment cost

![grafana-monitoring.gif](/images/mlops-projects-overview/coco-puppy.jpg)



---

## Course project?

<details><summary style="font-style: italic;">What pattern does that follow?</summary>

- Offline training

- On-demand through REST API

</details>


