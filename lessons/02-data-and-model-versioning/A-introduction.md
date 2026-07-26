# Data and Model Versioning

## Overview

### Agenda

 * [08:15 – 09:00] – Lecture: General data versioning strategies
 * [09:15 – 10:00] – Lecture: Continued
 * [10:15 – 11:00] – Exercises: DVC
 * [11:15 – 12:00] – Exercises: Continued

### Preparation

For the exercises:

* [DVC: Install](https://dvc.org/doc/install)
* [DVC: Getting started](https://dvc.org/doc/start)
* [DVC: Tutorial](https://dvc.org/doc/use-cases/versioning-data-and-models/tutorial)

For the lecture:

* [DVC](https://dvc.org/doc/)
* [ML Engineering, Chapter 3.11](http://www.mlebook.com/wiki/doku.php) (http://bit.ly/MLEbook-Chapter3)
* [ML-Ops.org: Data Engineering Pipelines](https://ml-ops.org/content/three-levels-of-ml-software)

### Notes

* Make sure to install DVC and test that the `dvc` command works for you.
* Slides uploaded to [learnit](https://learnit.itu.dk/course/view.php?id=3023571#section-6).

## Slides

Slides are available on [LearnIT](https://learnit.itu.dk/course/view.php?id=3023571#section-6).

## Exercises

The following exercises will guide you through using DVC for data and model versioning:

- [Exercise 0-1: Setup and Initialization](B-exercise-setup.md) - Install DVC and initialize in your repository
- [Exercise 2: Start Tracking Files](C-exercise-tracking.md) - Add files to DVC tracking
- [Exercise 3: Using a Remote](D-exercise-remote.md) - Configure and use remote storage
- [Exercise 4: Switching Between Versions](E-exercise-versions.md) - Manage and switch between data versions
- [Exercise 5: Import from URL](F-exercise-import.md) - Import external data

## Theory

Data versioning has plenty of similarities but also differences to normal version control that you have just worked with. 

The main thing that we want to accomplish with data version control is to be able to always go back to one version of the data and see how it looked like at a given time -- similar to normal version control. If you want to evaluate your model's performance at time $t_1$, it does not make sense to use data from time $t_2$.

One big difference is that it is simply not possible to meaningfully look at the differences between data versions and review it through a PR. Good PRs tend to only touch a few lines of code at a time, but data can be millions of rows/lines. Not only would it take _you_ ages, but it can also greatly slow down any version control software that has to calculate all the differences. And the data can be enormous, so checking out the whole repo would be impossible on your "small" laptop.

![PR Lines of code](../resources/code%20review%20strategy.jpeg "True story")

So what's a good strategy to keep track of changes to your data? Briefly explained, you store the data in a remote location and version control _references_ to the data instead.

For that purpose, we will use the tool _Data Version Control_ (DVC). It can also be used for model versioning and experiment tracking, but we'll touch on that in later lectures. Similar tools include Git Large File Storage (GitLFS) or Pachyderm. Each of these tools have their pros and cons, and DVC might struggle with performance if you have a lot of files, but it serves our purpose of refreshing our git skills and motivate the thinking of storing data remotely well enough.

![Project version control](../resources/project-versions.png "From https://dvc.org/doc/use-cases/versioning-data-and-models")

The goal of the following exercises is to take you through the journey of linking data to your project.
