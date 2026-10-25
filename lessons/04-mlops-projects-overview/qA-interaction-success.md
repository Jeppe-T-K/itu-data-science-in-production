---
title: "Project Success"
---

# Project Success

Recap of "Project Success Factors" and "Tech Debt".

---

## Discussion

<details><summary>What does it take for a data science project to be successful? Can you agree on a definition of "success"?</summary>

- Typically: the model actually runs in production

- Personal criterion: X users over Y years

- Academic: [DS PRO-S](https://www.mdpi.com/2076-3417/16/5/2551)

</details>

---

## Questions

How many data science projects never make it into production?

- A: Roughly 13 out of 100
- B: Roughly 87 out of 100
- C: Almost all of them
- D: Roughly 50 out of 100

<details><summary>Answer</summary>

**B.** [VentureBeat 2019: 87%](https://venturebeat.com/business/why-do-87-of-data-science-projects-never-make-it-into-production) never make it into production, and [Gartner 2018: 85%](https://www.gartner.com/en/documents/4003368) deliver erroneous outcomes. My own experience is >50%.

A is actually the *success* rate (100 - 87). C overstates it - most projects fail, but not almost all. D is closer to my personal experience but still too optimistic compared to the published numbers.

![stonks.png](/images/mlops-projects-overview/stonks.png)

</details>

---

What does "fail" typically mean in these statistics?

- A: The model crashes every time it is called in production
- B: The team behind the project was disbanded
- C: The model never runs in production
- D: The model scored below 50% accuracy

<details><summary>Answer</summary>

**C.** The statistics count everything that never reaches production - including projects cancelled before deployment.

A, B, and D are specific failure modes, but they are too narrow: a project that never shipped at all still counts as a failure here.

</details>

---

Why do data science projects fail? Select all that apply.

- A: Bad data
- B: Not enough GPUs
- C: Lack of integration into existing systems
- D: Unclear business involvement
- E: The model was not a large language model
- F: Lack of diverse expertise
- G: Changing anything changes everything

<details><summary>Answer</summary>

**A, C, D, F, and G.** These are the recurring themes: bad data, no integration path, no business ownership, one-sided teams, and tight coupling everywhere.

B and E are red herrings - the choice of model family or the amount of hardware is rarely what kills a project.

</details>

---

## Tech debt

![neurips-mlops-blocks.png](/images/mlops-projects-overview/neurips-mlops-blocks.png)
From https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

The NeurIPS paper "Hidden Technical Debt in Machine Learning Systems" names several anti-patterns. Which of the following are from the paper? Select all that apply.

- A: Glue code
- B: Too many unit tests
- C: Pipeline jungles
- D: Too many CPUs
- E: Dead experimental paths
- F: Configuration debt
- G: Hidden feedback loops

<details><summary>Answer</summary>

**A, C, E, F, and G** are all named in the paper, together with unstable data dependencies and cultural debt.

B and D are not in the paper - having too many tests or too much hardware is not a form of technical debt the authors describe.

</details>

---

What is a "pipeline jungle"?

- A: A pipeline that runs in a cloud provider's "jungle" region, i.e., an unsupported zone
- B: A pipeline assembled ad-hoc from whatever tools were at hand, often around legacy data, that nobody dares to touch
- C: A pipeline with so many unit tests that it can never fail in production
- D: A data pipeline that only processes jungle imagery, such as satellite photos

<details><summary>Answer</summary>

**B.** The paper describes pipeline jungles as scar tissue from gluing tools together ad-hoc, typically around legacy data sources.

A and D are jokes, and C confuses testing with pipeline design - the problem is the structure, not the lack of tests.

</details>
