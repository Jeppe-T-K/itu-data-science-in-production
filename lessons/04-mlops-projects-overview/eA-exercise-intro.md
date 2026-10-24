---
title: Exercise Intro
---

# Background

Whenever you start a coding project, there's always the question of how you will structure the code.

Where is the main entry point into the code? What util functions do you need to add? Which components does your project require?

There is obviously not _one_ correct way of doing it, and it very much depends on what kind of project you're building. A good starting point for that is [_Cookiecutter_](https://cookiecutter.readthedocs.io/en/stable/README.html), which is a tool that can help you set up various projects using whichever [template](https://www.cookiecutter.io/templates) that suits your purpose.

![Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/ccds.png "Cookiecutter Data Science AI overlord logo")

For these exercises, we will use a specific template for Data Science, called [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). This is not by any means the only way to structure a data science project, but the template has [well-reasoned arguments](https://cookiecutter-data-science.drivendata.org/opinions/) as to why.

## Learning Goals

> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Start a data science project with version control that follows the CCDS format</li>
> <li>Explain the motivation for the structure of a data science project</li>
> <li>Map the typical MLOps steps (data, models, code) onto a concrete project structure</li>
> <li>Run an ML pipeline locally from end to end as a set of composable steps</li>
> <li>Set up a CI/CD workflow that tests your project and produces artifacts on every change</li>
> <li>Use S3-compatible object storage for data and model artifacts, with credentials kept out of version control</li>
> </ul>

## Exercises

The following exercises take you from an empty directory to a working MLOps project:

- [Exercise 1: Cookiecutter Setup](cookiecutter-setup) - Scaffold the repo and push it to GitHub
- [Exercise 2: Project Structure](project-structure) - Map the MLOps steps from the lecture onto the CCDS structure
- [Exercise 3: Local Pipeline](local-pipeline) - Run the whole ML pipeline locally with a single command
- [Exercise 4: CI/CD](cicd) - Test the project and produce artifacts automatically with GitHub Actions
- [Exercise 5 (extra): Cloud Storage](cloud-storage) - Store data and model artifacts in S3-compatible object storage

---

# Exercise 0: Setup

Make sure Cookiecutter Data Science is installed:
`ccds --version`
Does this command run and show a version number above 2.something? Great!
