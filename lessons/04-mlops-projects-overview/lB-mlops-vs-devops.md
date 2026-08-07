---
title: MLOps vs DevOps
---

# MLOps vs DevOps

---

<details><summary style="font-size: 1.5em"> First: DevOps</summary>

![Dynatrace DevOps with logos](/images/mlops-projects-overview/dynatrace-devops-with-logos.svg)

Modified, from https://www.dynatrace.com/news/blog/what-is-devops/

<details><summary style="font-size: 1.2em">Development</summary>

- **Plan**: Write down goals, tasks, and project rules, organize code with _CCDS_.
- **Code**: Write the code, version control with _Git_.
- **Build**: Package the program, build with _Docker_.
- **Test**: Check for bugs and errors, use unit/integration tests.
</details>

<details><summary style="font-size: 1.2em">Operations</summary>

- **Release**: Prepare software to go live.
- **Deploy**: Implement it in production, automate with _Github Actions_ and _Dagger_.
- **Operate**: Keep servers running, organize containers with _Kubernetes_.
- **Monitor**: Watch system health, visualize and alert with _Grafana_.
</details>

</details>

---

<details><summary style="font-size: 1.5em"> And MLOps?</summary>

![Dynatrace MLOps with logos](/images/mlops-projects-overview/dynatrace-mlops-with-logos.svg)

### ML
- **Data**: External info (vs internal code), version with _DVC_.
- **Model**: Machine learning model, track with _MLflow_.

</details>

---

<details><summary style="font-size: 1.5em"> Different graphics</summary>

<details><summary style="font-size: 1.2em"> Slightly less ugly</summary>

![Ubuntu MLOps](/images/mlops-projects-overview/ubuntu-mlops.png)

From https://ubuntu.com/blog/what-is-mlops 

![ml-ops.org MLOps](/images/mlops-projects-overview/ml-ops-org-circles.png)

From https://ml-ops.org/content/mlops-principles 

</details>

<details><summary style="font-size: 1.2em"> Slightly less abstract</summary>

![ml-ops.org MLOps](/images/mlops-projects-overview/ml-ops-org-pipeline.png)

From https://ml-ops.org/content/mlops-principles 

</details>

</details>

---

<details><summary style="font-size: 1.5em"> Data and models add complexity</summary>


![conference talk tweet.png](/images/mlops-projects-overview/conference-talk-tweet.png)
Source for original graphic: https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

![Page view evolution](/images/mlops-projects-overview/page-view-evolution.png)
Page views over time

- Element of unpredictability
- Data + artifact versioning
- Changes over time

→ So much to keep track of

</details>