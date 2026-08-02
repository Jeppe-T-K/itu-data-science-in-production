---
title: Levels of Data Versioning
---

# Levels of Data Versioning

---

## Multiple Levels of Data Versioning

4 levels according to the [MLE book, chapter 3.11](https://www.mlebook.com/wiki/doku.php)

| Level 0 | Level 1 | Level 2 | Level 3 |
|---------|---------|---------|---------|
| Data is unversioned | Data is versioned as a snapshot | Data and code are versioned | Data and code are versioned through specialised tools |

---

<details>
<summary style="font-size: 1.5em;">Level 0: Unversioned Data</summary>

The simplest (and least recommended) approach is to have no versioning at all. Data is simply stored and overwritten as needed.

![Manual version control](/images/data-and-model-versioning/manual-version-control.png)

- Simple to implement
    - No overhead
- No history
    - Not possible to revert
- Note: many tools have _something_ built in

</details>

---

<details>
<summary style="font-size: 1.5em;">Level 1: Snapshots</summary>

Create snapshots of your data at different points in time. This allows you to restore previous versions.

![LakeFS Snapshots](/images/data-and-model-versioning/lakefs-snapshots.png)

From https://lakefs.io/blog/data-versioning/


- Duplicate all relevant data
    - Storage intensive → cloud
    - Manual tracking of versions

- Viable if:
    - Infrequent updates
    - Few models 


</details>

---

<details>
<summary style="font-size: 1.5em;">Level 2: Data and Code as One Asset</summary>

Version data alongside code using Git with large file support.

![Git LFS](/images/data-and-model-versioning/git-lfs.png)

From https://git-lfs.com/

- Version control with Git
    - Metadata saved for large files

- Large files stored in cloud
    - Efficient formats

- Tools:
    - DVC
    - Git Large Files Storage 

</details>

---

<details>
<summary style="font-size: 1.5em;">Level 3: Specialized Versioning</summary>

Use specialized tools designed for data versioning that handle large files efficiently.

![DVC Experiment Tracking](/images/data-and-model-versioning/dvc-experiment-tracking.png)

From https://dvc.org/doc/use-cases/experiment-tracking

- Tool-specific solutions

- Comes with a lot extra functionality
    - Pipelines
    - Experiment tracking
    - Etc etc

</details>
