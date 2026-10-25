---
title: "Data and Model"
---

# Data and Model

Recap of the "Data Component" and "Model Component" lectures.

---

## Discussion

<details><summary>We keep saying the model is a "black box". Why do we treat it that way - and what follows from it for how we work with models?</summary>

- Training maps inputs to outputs without exposing internal logic - we cannot look inside, only observe behaviour

- So we evaluate from the outside: holdout test sets, multiple metrics (error, bias)

- The model must beat a baseline, or it is not worth deploying

- Don't get too attached to one model - feature and model engineering, random search for hyperparameters → robustness is important

![Model black box](/images/mlops-projects-overview/black-box.svg)

</details>

---

## Questions

Of a data scientist's working hours, how much goes to collecting and cleaning data rather than modeling?

- A: Around 5%
- B: Around 60%
- C: Practically 100%
- D: Around 20%

<details><summary>Answer</summary>

**B.** CrowdFlower's survey put ~60% on cleaning (plus ~19% on collecting data), and Anaconda's State of Data Science found ~45% on data loading and preparation.

A and D are far too optimistic - the data work dominates. C overstates it: there is still modeling, evaluation, and communication left.

→ Garbage in, garbage out - the data component deserves the attention

![garbage collect](/images/mlops-projects-overview/garbage-collection.png)

</details>

---

What is the typical order of steps in the data component?

- A: Preparation → extraction → validation
- B: Validation → training → extraction
- C: Extraction → validation → preparation
- D: There are no steps; data is data

<details><summary>Answer</summary>

**C.** First you get the data (extraction), then you understand and check it (validation), then you clean and transform it (preparation).

A and B scramble the order - you cannot validate data you have not extracted, and preparation before validation wastes effort on data that may be unusable. D ignores that the pipeline has a deliberate structure.

</details>

---

What does data exploration/validation look at? Select all that apply.

- A: Missing values
- B: The model's hyperparameters
- C: Mean and standard deviation
- D: The number of GPUs
- E: Data types

<details><summary>Answer</summary>

**A, C, and E** - the basic shape of the data.

B belongs to model training, not the data. D is a hardware question.

![Data exploration](/images/mlops-projects-overview/great-expectations-data.png)

</details>

---

Data validation is not just for training - what else is it useful for?

- A: Generating new features automatically from raw data
- B: Making the dataset smaller so training runs faster
- C: Catching problems in incoming data before they poison the model
- D: Ensuring compliance by itself, without further policies

<details><summary>Answer</summary>

**C.** Validation rules double as monitors: when incoming serving or training data breaks the expected schema, you can alert on it.

A, B, and D are not what validation does - feature generation is a preparation step, size reduction is not the goal, and compliance needs policies and processes beyond data checks.

</details>

---

What does the "silver" layer refer to?

- A: The most expensive data, e.g. data purchased from a vendor
- B: A backup of the raw data, kept in case the pipeline breaks
- C: Second prize in a Kaggle competition, after the gold layer
- D: The cleaned, transformed data produced after preparation and wrangling

<details><summary>Answer</summary>

**D.** After extraction and validation, preparation turns raw data into the cleaned "silver" layer that models actually train on.

A, B, and C play on the name - the term describes a quality/processing stage, not price, backups, or medals.

</details>

---

What must be true before a model is operationalized?

- A: It has been deployed to production
- B: It has the largest parameter count of all candidates
- C: It passes unit/integration tests and is validated to be better than the baseline on a holdout set
- D: It was trained on all available data to make sure it is as up to date as possible.

<details><summary>Answer</summary>

**C.** Unit/integration tests check the model specification, and evaluation on a holdout set must beat the baseline.

A is circular - deployment is what comes *after* operationalization is justified. B rewards size over quality and may be difficult to explain to stakeholders. D leaks the test set: the holdout must stay untouched for the evaluation to mean anything.

</details>

---

Why is model packaging tricky? Select all that apply.

- A: Not all serving tools support all formats
- B: Models cannot be saved at all, only retrained
- C: Different frameworks save models in different formats (Pickle, ONNX, TorchScript, PMML, ...)
- D: Formats differ in compression and human-readability

<details><summary>Answer</summary>

**A, C, and D.** The zoo of formats (Pickle, ONNX, TorchScript, PMML, HDF, ...) is exactly why packaging and testing matters.

B is flatly wrong - every framework has a serialization format; the difficulty is compatibility, not existence.

</details>

---

Which statement about ONNX is true?

- A: It only works with PyTorch and the PyTorch serving stack
- B: It is a proprietary Apple format used for Core ML apps
- C: It stores the model as human-readable XML
- D: It is an open format supported across many frameworks and runtimes (TF, Core ML, ML.NET, ...)

<details><summary>Answer</summary>

**D.** ONNX is an open format backed by the LF AI & Data Foundation, with support across TensorFlow, Core ML, ML.NET, and more.

A is wrong - it was designed precisely to work across frameworks. C describes PMML (XML), not ONNX, which is a compressed binary format. B confuses it with Apple's .mlmodel.

</details>
