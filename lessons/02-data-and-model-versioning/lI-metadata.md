# Metadata

Metadata is crucial for understanding and managing data versions.

![Metadata Management](/images/data-and-model-versioning/metadata-layer.png)

From https://www.acceldata.io/blog/what-is-metadata-definition-types-and-importance

<details><summary style="font-size: 1.2em;">Descriptive</summary>

- **Description**: Human-readable description of the data
- **Keywords**: Tags and subjects for categorization
- **Titles**: Dataset or table names
- **Authors**: Who created the data

</details>

<details><summary style="font-size: 1.2em;">Structural</summary>

- **Schema**: Data structure, column names, types
- **Partitioning**: Physical layout rules
- **Relationships**: How tables and datasets are connected
- **Statistics**: Min, max, mean, null counts, etc.
- **Data Quality**: Quality scores, validation results

</details>

<details><summary style="font-size: 1.2em;">Administrative</summary>

- **Author**: Who created or modified the data
- **Timestamp**: When was it created or modified
- **Version**: Version identifier (tag, commit hash, etc.)
- **Changelog**: What changed between versions
- **Ownership**: Who is responsible for the data
- **Policy**: Access management, sensitivity labels, permissions

</details>

<details><summary style="font-size: 1.2em;">Technical</summary>

- **Format**: File format (Parquet, CSV, JSON, etc.)
- **Size**: Dataset size and dimensions
- **Compression**: Compression method used

- **Resolution**: Data granularity or precision

</details>

<details><summary style="font-size: 1.2em;">Provenance</summary>

- **Lineage**: Where did this data come from (data lineage)
- **Origin**: Source systems and extraction methods
- **History**: Complete transformation history

</details>

<details><summary style="font-size: 1.2em;">Usage</summary>

- **Usage**: Monitoring usage and cost estimates
- **Access patterns**: Who accesses the data and when
- **Query history**: Track how data is queried

</details>

