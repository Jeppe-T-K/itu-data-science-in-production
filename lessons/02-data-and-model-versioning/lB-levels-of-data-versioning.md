# Levels of Data Versioning

## Why Data Versioning?

Data versioning is essential for maintaining reproducibility and traceability in machine learning projects. Without it, you cannot reliably:
- Reproduce previous experiments
- Track which data version produced which model
- Roll back to a working state

## Multiple Levels of Data Versioning

There are several levels of data versioning, each with increasing complexity and capabilities.

### Level 0: Unversioned Data

The simplest (and least recommended) approach is to have no versioning at all. Data is simply stored and overwritten as needed.

**Pros:**
- Simple to implement
- No overhead

**Cons:**
- No history
- No ability to revert
- No reproducibility

### Level 1: Snapshots

Create snapshots of your data at different points in time. This allows you to restore previous versions.

![LakeFS Snapshots](/images/data-and-model-versioning/lakefs-snapshots.png)

From https://lakefs.io/blog/data-versioning/

**Tools:**
- LakeFS
- Delta Lake
- Iceberg

### Level 2: Data and Code as One Asset

Version data alongside code using Git with large file support.

![Git LFS](/images/data-and-model-versioning/git-lfs.png)

From https://git-lfs.com/

**Pros:**
- Unified version control
- Familiar Git workflow

**Cons:**
- Not efficient for large datasets
- Can slow down repositories

### Level 3: Specialized Versioning

Use specialized tools designed for data versioning that handle large files efficiently.

![DVC Versioning](/images/data-and-model-versioning/dvc-versioning.png)

From https://dvc.org/doc/use-cases/versioning-data-and-models

![DVC Experiment Tracking](/images/data-and-model-versioning/dvc-experiment-tracking.png)

From https://dvc.org/doc/use-cases/experiment-tracking

**Tools:**
- DVC (Data Version Control)
- Pachyderm
- MLflow

**Pros:**
- Optimized for large datasets
- Handles data lineage
- Integrates with ML workflows

## Package Solutions

DVC provides a comprehensive solution for data versioning in ML projects.

![DVC Package Solution](/images/data-and-model-versioning/dvc-versioning.png)

From https://dvc.org/doc/use-cases/versioning-data-and-models

Key features:
- Track data files alongside code
- Store data in remote storage (S3, GCS, Azure, SSH, etc.)
- Version data with Git
- Share and collaborate on datasets
