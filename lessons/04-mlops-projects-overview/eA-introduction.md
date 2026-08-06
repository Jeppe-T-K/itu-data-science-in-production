---
title: Exercise Intro
---

# Background

Whenever you start a coding project, there's always the question of how you will structure the code.

Where is the main entry point into the code? What util functions do you need to add? Which components does your project require?

There is obviously not _one_ correct way of doing it, and it very much depends on what kind of project you're building. A good starting point for that is [_Cookiecutter_](https://cookiecutter.readthedocs.io/en/stable/README.html), which is a tool that can help you set up various projects using whichever [template](https://www.cookiecutter.io/templates) that suits your purpose.

![Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/ccds.png "Cookiecutter Data Science AI overlord logo")

For these exercises, we will use a specific template for Data Science, called [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/). This is not by any means the only way to structure a data science project, and we will in the subsequent lessons work on adding extra pieces to our project, but the template has [well-reasoned arguments](https://cookiecutter-data-science.drivendata.org/opinions/) as to why.

## Learning Goals

> [!NOTE]
> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Start a data science project with version control that follows the CCDS format</li>
> <li>Explain the motivation for the structure of a data science project</li>
> <li>Share your repo with others and work on the same code</li>
> </ul>

## Exercises

The following exercises will guide you through setting up MLOps projects:

- [Exercise 1: Cookiecutter Setup](eB-cookiecutter-setup.md) - Initialize Cookiecutter Data Science
- [Exercise 1: Project Structure](eC-project-structure.md) - Inspect and understand the CCDS structure
- [Exercise 2: Project Teaser](eD-project-teaser.md) - Preview of MLOps project structure
- [Exercise 3: Collaboration](eE-collaboration.md) - Setting up GitHub and working together

# Exercise 0: Setup

Make sure Cookiecutter Data Science is installed:
`ccds --version`
Does this command run and show a version number above 2.something? Great!