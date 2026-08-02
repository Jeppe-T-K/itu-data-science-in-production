---
title: Introduction
---

# MLOps Projects Overview and Pipelines

## Overview

### Agenda

 * [08:15 – 09:00] – Follow-along exercises: Cookiecutter Data Science + project teaser
 * [09:15 – 10:00] – Follow-along exercises: CCDS
 * [10:15 – 11:00] – Lecture: MLOps in organisations
 * [11:15 – 12:00] – Lecture: MLOps stages

### Preparation

For the exercises:

* [Cookiecutter Data Science: Install](https://cookiecutter-data-science.drivendata.org/)
* [Cookiecutter Data Science: Opinions on structure](https://cookiecutter-data-science.drivendata.org/opinions/)
* [Cookiecutter Data Science: Using the template](https://cookiecutter-data-science.drivendata.org/using-the-template/)
* [Cookiecutter Data Science: Github repo](https://github.com/drivendataorg/cookiecutter-data-science)

For the lecture:

* [ML-Ops.org: Motivation](https://ml-ops.org/content/motivation)
* [ML-Ops.org: MLOps principles](https://ml-ops.org/content/mlops-principles)
* [ML-Ops.org: Three levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software)
* [ML-Ops.org: End-to-End ML](https://ml-ops.org/content/end-to-end-ml-workflow)
* [Google & MLOps levels](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

### Notes

* The exercises will be a follow-along coding session, mainly using terminal/CLI
* Make sure to install Cookiecutter Data Science and test that the `ccds` command works for you.
* Slides uploaded to [learnit](https://learnit.itu.dk/course/view.php?id=3025404#section-7).

## Slides

Slides are available on [LearnIT](https://learnit.itu.dk/course/view.php?id=3025404#section-7).

## Exercises

The following exercises will guide you through setting up MLOps projects:

- [Exercise 0-1: Cookiecutter Setup](eB-cookiecutter-setup.md) - Install and initialize Cookiecutter Data Science
- [Exercise 1: Project Structure](eC-project-structure.md) - Inspect and understand the CCDS structure
- [Exercise 2: Project Teaser](eD-project-teaser.md) - Preview of MLOps project structure
- [Exercise 3: Collaboration](eE-collaboration.md) - Setting up GitHub and working together

## Learning Goals

> [!NOTE]
> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Start a data science project with version control that follows the CCDS format</li>
> <li>Explain the motivation for the structure of a data science project</li>
> <li>Share your repo with others and work on the same code</li>
> </ul>

Whenever you start a coding project, there's always the question of how you will structure the code.

Where is the main entry point into the code? What util functions do you need to add? Which components does your project require?

There is obviously not _one_ correct way of doing it, and it very much depends on what kind of project you're building. A good starting point for that is [_Cookiecutter_](https://cookiecutter.readthedocs.io/en/stable/README.html), which is a tool that can help you set up various projects using whichever [template](https://www.cookiecutter.io/templates) that suits your purpose.

![Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/ccds.png "Cookiecutter Data Science AI overlord logo")

For these exercises, we will use a specific template for Data Science, called [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). This is not by any means the only way to structure a data science project, and we will in the subsequent lessons work on adding extra pieces to our project, but the template has [well-reasoned arguments](https://cookiecutter-data-science.drivendata.org/opinions/) as to why.
