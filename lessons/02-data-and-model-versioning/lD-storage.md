# Data Storage

## Storage Costs Money

One of the fundamental realities of data versioning is that storage has a cost. This affects how we approach data management.

![Backblaze Storage Costs](/images/data-and-model-versioning/backblaze-storage.png)

From https://www.backblaze.com/blog/whats-the-diff-hot-and-cold-data-storage/

## Cost Strategies

### Cloud vs On-Premise

**Cloud Storage:**
- Scalable
- Pay-as-you-go
- Managed by provider
- Easy to access from anywhere

**On-Premise Storage:**
- Capital expenditure upfront
- Full control
- Potential security benefits
- Maintenance overhead

### Hot vs Cold Storage

**Hot Storage:**
- Frequently accessed data
- Faster access times
- Higher cost per GB
- Examples: SSD, fast HDD, in-memory

**Cold Storage:**
- Rarely accessed data
- Slower access times
- Lower cost per GB
- Examples: Glacier, archive storage, tape

### Retention Policies

Define how long different types of data should be kept:
- Raw data: May need to be kept indefinitely for reproducibility
- Processed data: Can often be regenerated
- Temporary files: Can be deleted after use

### Partitioning

Split data into logical partitions:
- By date (year/month/day)
- By project
- By data type
- By version

## Data Types

### Tabular/Relational Data

Structured data organized in tables with rows and columns.

![Tabular Data](/images/data-and-model-versioning/statology-tabular.png)

From https://www.statology.org/tabular-data/

**Examples:**
- CSV files
- SQL databases
- Parquet files
- Spreadsheets

### Images (RAW files, compressed)

Image data comes in various formats with different storage requirements.

![Nikon RAW File Size Options](/images/data-and-model-versioning/nikon-raw.png)

From https://mcpactions.com/nikon-raw-s-file-size-option/

**Considerations:**
- RAW files are much larger than compressed formats
- Compression may lose information
- Different formats for different use cases

## Structured vs Unstructured Data

**Structured Data:**
- Defined schema
- Organized format
- Easy to query and analyze
- Examples: SQL tables, CSV files, JSON with schema

**Unstructured Data:**
- No defined schema
- Variable format
- Harder to query
- Examples: Text documents, images, audio, video

## Data Storage Types

### Data Lakes

Centralized repositories that hold raw data in its native format.

![Qlik Data Lake](/images/data-and-model-versioning/databricks-data-lake.png)

From https://www.qlik.com/us/data-lake

**Characteristics:**
- Schema-on-read
- Supports various data types
- Cost-effective for large volumes
- Good for exploration and discovery

### Data Warehouses

Optimized for structured data and analytical queries.

**Characteristics:**
- Schema-on-write
- Optimized for SQL queries
- Higher cost for large volumes
- Good for reporting and BI

### Lakehouses

Combine the best of data lakes and data warehouses.

![Databricks Lakehouse](/images/data-and-model-versioning/databricks-data-lakehouse.png)

From https://www.databricks.com/research/lakehouse-a-new-generation-of-open-platforms-that-unify-data-warehousing-and-advanced-analytics

**Characteristics:**
- ACID transactions on data lake storage
- Support for both structured and unstructured data
- Optimized for both analytics and AI/ML
- Open formats (Parquet, Delta, etc.)

## Storage Recommendations

1. **Use appropriate storage tiers**: Hot for active data, cold for archives
2. **Implement lifecycle policies**: Automatically move data between tiers
3. **Compress data**: Use appropriate compression for each data type
4. **Deduplicate**: Avoid storing the same data multiple times
5. **Monitor costs**: Regularly review storage usage and costs
