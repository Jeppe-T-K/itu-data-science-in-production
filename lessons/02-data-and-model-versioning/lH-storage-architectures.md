---
title: Storage Architectures
---

# Data Storage Architectures

![Databricks Data Warehouse](/images/data-and-model-versioning/data-architecture-history.svg)

Based on https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

- **Lakes**: raw, unfiltered
- **Warehouses**: processed, vetted
- **Lakehouses**: Both?
- **ELT** vs **ETL**
    - Extract, Load/Transform
    - Steps before saving/storing data

---

<details><summary style="font-size: 1.5em;">Data Warehouses</summary>

Optimized for structured data and analytical queries.

<img src="/images/data-and-model-versioning/databricks-data-warehouse.png" alt="Databricks Data Warehouse" height="400" style="display: block; margin: 0 auto;">

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics**
- Schema-on-write
- Optimized for SQL queries
- Good for reporting and BI

**Limitations**
- Lack of customizability for AI/ML
- Images and unstructured data
- Higher cost for large volumes

</details>

---

<details>
<summary style="font-size: 1.5em;">Data Lakes</summary>

Centralized repositories that hold raw data in its native format, kicked off with Hadoop.

<img src="/images/data-and-model-versioning/databricks-data-lake.png" alt="Databricks Data Lake" height="400" style="display: block; margin: 0 auto;">

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics**
- Schema-on-read
- Supports various data types
- Cost-effective for large volumes
- Good for exploration and discovery

**Limitations**
- Requires additional, complex ETL loops
- Delayed data for BI/analytics
- Cost of ownership (duplicate data, harder dvc, no [ACID transactions](https://www.databricks.com/blog/what-are-acid-transactions))
</details>

---

<details>
<summary style="font-size: 1.5em;">Data Lakehouses</summary>

Combination of data lakes and data warehouses.

<img src="/images/data-and-model-versioning/databricks-data-lakehouse.png" alt="Databricks Data Lakehouse" height="400" style="display: block; margin: 0 auto;">

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

- ACID transactions on data lake storage
- Support for both structured and unstructured data
- Optimized for both analytics and AI/ML
- Open formats (Parquet, Delta, etc.)
</details>