---
title: MLOps Elements
---

# MLOps Elements

Based on https://ml-ops.org/content/mlops-principles

---

## Three Components of MLOps Pipelines

<div style="text-align: center;">
  <img src="/images/mlops-projects-overview/mlops-triangle.png" alt="MLOps levels in course" style="width: 80%;" />
</div>

From https://ml-architects.ch/blog_posts/mlops_maturity_model.html


<details><summary style="font-style: italic"> What's going on in each step? </summary>

<details><summary style="font-weight: bold; color: #8a9faf; font-size: 1.5em">Data</summary>

- <details><summary>1. </summary> Extraction </details>
- <details><summary>2. </summary> Validation </details>
- <details><summary>3. </summary> Preparation </details>

</details>


<details><summary style="font-weight: bold; color: #51bdf2; font-size: 1.5em">Model</summary>

- <details><summary>1. </summary> Training </details>
- <details><summary>2. </summary> Evaluation </details>
- <details><summary>3. </summary> Validation </details>
</details>

<details><summary style="font-weight: bold; color: #25506c; font-size: 1.5em">Code</summary>

- <details><summary>1. </summary> Deployment </details>
- <details><summary>2. </summary> Serving </details>
- <details><summary>3. </summary> Monitoring </details>
</details>



<details><summary style="font-size:1.0em; font-style: italic;"> Modularizing things, diagrammatically </summary>

<div style="text-align: center;">
  <img src="/images/mlops-projects-overview/mlops-pipeline.svg" alt="MLOps pipeline" style="width: 80%;" />
</div>

</details>

</details>

---

<details><summary style="font-size: 1.8em">MLOps Principles</summary>

<div style="text-align: center;">
  <img src="/images/mlops-projects-overview/mlops-principles.svg" alt="MLOps principles in course" style="width: 80%;" />
</div>


<details><summary style="font-weight: bold; color: blue; font-size: 1.5em">Versioning</summary>

* **Data**
  1) Data preparation pipelines
  2) Features store
  3) Datasets
  4) Metadata

* **Model**
  1) ML model training pipeline
  2) ML model (object)
  3) Hyperparameters
  4) Experiment tracking

* **Code**
  1) Application code
  2) Configurations

</details>


<details><summary style="font-weight: bold; color: orange; font-size: 1.5em">Automation</summary>

* **Data**
  1) Data transformation
  2) Feature creation and manipulation

* **Model**
  1) Data engineering pipeline
  2) ML model training pipeline
  3) Hyperparameter/Parameter selection

* **Code**
  1) ML model deployment with CI/CD
  2) Application build

</details>



<details><summary style="font-weight: bold; color: orange; font-size: 1.5em">Reproducibility</summary>

* **Data**
  1) Backup data
  2) Data versioning
  3) Extract metadata
  4) Versioning of feature engineering

* **Model**
  1) Hyperparameter tuning is identical between dev and prod
  2) The order of features is the same
  3) Ensemble learning: the combination of ML models is same
  4) The model pseudo-code is documented 

* **Code**
  1) Versions of all dependencies in dev and prod are identical
  2) Same technical stack for dev and production environments
  3) Reproducing results by providing container images or virtual machines 

</details>


<details><summary style="font-weight: bold; color: orange; font-size: 1.5em">Deployment</summary>

* **Data**
  1) Feature store is used in dev and prod environments

* **Model**
  1) Containerization of the ML stack
  2) REST API
  3) On-premise, cloud, or edge 

* **Code**
  1) On-premise, cloud, or edge

</details>


<details><summary style="font-weight: bold; color: green; font-size: 1.5em">Testing</summary>

* **Data**
  1) Data Validation (error detection)
  2) Feature creation unit testing

* **Model**
  1) Model specification is unit tested
  2) ML model training pipeline is integration tested
  3) ML model is validated before being operationalized
  4) ML model staleness test (in production)
  5) Testing ML model relevance and correctness
  6) Testing non-functional requirements (security, fairness, interpretability)

* **Code**
  1) Unit testing
  2) Integration testing for the end-to-end pipeline
  
</details>


<details><summary style="font-weight: bold; color: green; font-size: 1.5em">Monitoring</summary>

* **Data**
  1) Data distribution changes (training vs. serving data)
  2) Training vs serving features

* **Model**
  1) ML model decay
  2) Numerical stability
  3) Computational performance of the ML model

* **Code**
  1) Predictive quality of the application on serving data

</details>

</details>