# Data and Model Versioning

## Lecture Introduction

This lecture covers data and model versioning strategies for Data Science in Production: MLOps and Software Engineering.

## Why Data Versioning?

Data versioning provides several critical benefits:

- **Unique reference**: Each version of your data has a unique identifier
- **Versioned changes**: Track changes to your data over time
- **Revert/rollback**: Ability to revert to previous versions when needed
- **Check previous performance**: Reproduce and verify model performance at any point in time

## Complexity of Data Science Projects

Without proper planning, data science projects can become overly complex and unmanageable. Version control helps prevent this.

![Complexity of DS projects](/images/data-and-model-versioning/dvc-versioning.png)

From https://dvc.org/doc/use-cases/versioning-data-and-models

## Outline

The lecture covers the following topics:

1. Why data versioning?
2. Data storage
3. Data stages (medallion)
4. Data version operations
5. Metadata for versions
6. Data stages in ML
