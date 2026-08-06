---
title: Project Success Factors
---

# Project success factors

![stonks.png](/images/mlops-projects-overview/stonks.png)

---

## Questions for you

<details><summary style="font-weight: italic; font-size: 1.2em">How many DS projects fail?</summary>

- My experience: >50%
- [VentureBeat 2019: 87%](https://venturebeat.com/business/why-do-87-of-data-science-projects-never-make-it-into-production)
- [Gartner 2018: 85%](https://www.gartner.com/en/documents/4003368)

</details>

<details><summary style="font-weight: italic; font-size: 1.2em">What do you think “fail” means?</summary>

- Typically: run in production
- Personal criterion: X users over Y years
- Academic: [DS PRO-S](https://www.mdpi.com/2076-3417/16/5/2551)

</details>

<details><summary style="font-weight: italic; font-size: 1.2em">Why do you think they fail?</summary>

- Bad data
- Lack of integration
- Lack of diverse expertise
- Unclear business involvement
- Changing anything changes everything
</details>


---

## Tech debt

![neurips-mlops-blocks.png](/images/mlops-projects-overview/neurips-mlops-blocks.png)
From https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

- **Complex Models Erode Boundaries**: many various dependencies and consumers

- **Data Dependencies Cost More than Code Dependencies**: unstable, unused and/or legacy data

- **Feedback Loops**: both direct and hidden

- **ML-System Anti-Patterns**: glue code, pipeline jungles, dead experimental paths, abstractions, common smells

- **Configuration Debt**: different data types/sources may have different requirements/validity/configs, etc

- **Dealing with Changes in the External World**: ML systems that can take action affect the world

- **Other Areas of ML-related Debt**: data testing, reproducability, systems with many models, cultural debt
