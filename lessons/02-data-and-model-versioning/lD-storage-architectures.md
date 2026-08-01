# Data Storage Architectures

![Databricks Data Warehouse](/images/data-and-model-versioning/data-architecture-history.jpg)

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

- Lakes: raw, unfiltered
- Warehouses: processed, vetted
- Lakehouses: Both?
- ELT vs ETL
    - Extract, Load/Transform
    - Steps before saving/storing data

---
## Data Warehouses

Optimized for structured data and analytical queries.

<img src="/images/data-and-model-versioning/databricks-data-warehouse.png" alt="Databricks Data Warehouse" height="400" style="display: block; margin: 0 auto;">

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics:**
- Schema-on-write
- Optimized for SQL queries
- Higher cost for large volumes
- Good for reporting and BI

---

## Data Lakes

Centralized repositories that hold raw data in its native format.

![Databricks Data Lake](/images/data-and-model-versioning/databricks-data-lake.png)

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics:**
- Schema-on-read
- Supports various data types
- Cost-effective for large volumes
- Good for exploration and discovery


---

## Data Lakehouses

Combination of data lakes and data warehouses.

![Databricks Lakehouse](/images/data-and-model-versioning/databricks-data-lakehouse.png)

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics:**
- ACID transactions on data lake storage
- Support for both structured and unstructured data
- Optimized for both analytics and AI/ML
- Open formats (Parquet, Delta, etc.)
