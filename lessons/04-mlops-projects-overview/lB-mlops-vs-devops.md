---
title: MLOps vs DevOps
---

# MLOps vs DevOps

---

## First: DevOps

![Dynatrace DevOps with logos](/images/mlops-projects-overview/dynatrace-devops-with-logos.svg)

Modified, from https://www.dynatrace.com/news/blog/what-is-devops/

### Development
- **Plan**: Write down goals, tasks, and project rules, organize code with _CCDS_.
- **Code**: Write the code, version control with _Git_.
- **Build**: Package the program, build with _Docker_.
- **Test**: Check for bugs and errors, use unit/integration tests.

### Operations
- **Release**: Prepare software to go live.
- **Deploy**: Implement it in production, automate with _Github Actions_ and _Dagger_.
- **Operate**: Keep servers running, organize containers with _Kubernetes_.
- **Monitor**: Watch system health, visualize and alert with _Grafana_.

---

## And MLOps?

![Dynatrace MLOps with logos](/images/mlops-projects-overview/dynatrace-mlops-with-logos.svg)

### ML
- **Data**: External info (vs internal code), version with _DVC_.
- **Model**: Machine learning model, track with _MLflow_.

---

## Different graphics

### Slightly less ugly

![Ubuntu MLOps](/images/mlops-projects-overview/ubuntu-mlops.png)

From https://ubuntu.com/blog/what-is-mlops 

![ml-ops.org MLOps](/images/mlops-projects-overview/ml-ops-org-circles.png)

From https://ml-ops.org/content/mlops-principles 

### Slightly less abstract

![ml-ops.org MLOps](/images/mlops-projects-overview/ml-ops-org-pipeline.png)

From https://ml-ops.org/content/mlops-principles 


---

## Data and models add complexity

![conference talk tweet.png](/images/mlops-projects-overview/conference-talk-tweet.png)
Source for original graphic: https://proceedings.neurips.cc/paper_files/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf

![Page view evolution](/images/mlops-projects-overview/page-view-evolution.png)
Page views over time

- Element of unpredictability
- Data + artifact versioning
- Changes over time

→ So much to keep track of
