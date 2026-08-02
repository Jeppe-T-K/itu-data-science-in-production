---
title: Storage Cost
---

# Data Storage Cost

![Data center](/images/data-and-model-versioning/data-center.png)

From https://lpigroup.com/portfolio-item/data-centre-netherlands/

## Storage Costs Money

Guesses for big cloud providers?

<details>
<summary>Answers:</summary>

<table>
  <thead>
    <tr><th>Provider</th><th>Hot / Standard</th><th>Infrequent Access / Cool</th><th>Cold</th><th>Archive</th><th>1TB Hot Storage Cost</th></tr>
  </thead>
  <tbody>
    <tr><td>AWS S3</td><td>$0.023/GB (first 50 TB)</td><td>$0.0125/GB (S3 Standard-IA)</td><td>$0.004/GB (Glacier Instant) · $0.0036/GB (Flexible)</td><td>$0.00099/GB (Glacier Deep Archive)</td><td>$23.00</td></tr>
    <tr><td>Azure Blob</td><td>$0.018/GB (Hot, LRS)</td><td>$0.010/GB (Cool)</td><td>$0.0045/GB (Cold)</td><td>$0.00099/GB (Archive)</td><td>$18.00</td></tr>
    <tr><td>Google Cloud Storage</td><td>$0.020/GB (Standard, regional)</td><td>$0.010/GB (Nearline, regional)</td><td>$0.004/GB (Coldline)</td><td>$0.0024/GB (Archive, US/EU multi-region)</td><td>$20.00</td></tr>
    <tr><td>Oracle OCI</td><td>$0.0255/GB (Object Standard)</td><td>$0.015/GB (Infrequent Access)</td><td>—</td><td>$0.0026/GB (Archive)</td><td>$25.50</td></tr>
  </tbody>
</table>

<p>From <a href="https://www.finout.io/blog/cloud-storage-pricing-comparison">https://www.finout.io/blog/cloud-storage-pricing-comparison</a></p>
</details>

--- 
<details>
<summary style="font-size: 1.5em;">Cost Strategies</summary>

<details>
<summary style="font-size: 1.2em; font-weight: bold;">Cloud vs On-Premise</summary>

**On-Premise Storage:**
- Capital expenditure upfront
- Full control
- Potential security benefits
- Maintenance overhead

**Cloud Storage:**
- Scalable
- Pay-as-you-go
- Managed by provider
- Easy to access from anywhere

</details>

<details>
<summary  style="font-size: 1.2em; font-weight: bold;">Hot vs Cold Storage</summary>

![Backblaze Storage Costs](/images/data-and-model-versioning/backblaze-storage.png)

From https://www.backblaze.com/blog/whats-the-diff-hot-and-cold-data-storage/

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

</details>

<details>
<summary style="font-size: 1.2em; font-weight: bold;">Retention Policies</summary>

![Data storage costs](/images/data-and-model-versioning/data_storage_cost_dual_axis_dkk.png)

How long different types of data should be kept.
- Indefinite is tempting
    - Running costs can explode, though
- Raw data can be kept for longer
    - Processed data can be regenerated

</details>

<details>
<summary style="font-size: 1.2em; font-weight: bold;">Partitioning</summary>

Split data into logical partitions:
- By date (year/month/day)
- By project
- By data type
- By version

</details>
</details>

--- 

<details>
<summary style="font-size: 1.5em;">Storage Recommendations</summary>

1. **Use appropriate storage tiers**: Hot for active data, cold for archives
2. **Implement lifecycle policies**: Automatically move data between tiers
3. **Compress data**: Use appropriate compression for each data type
4. **Deduplicate**: Avoid storing the same data multiple times
5. **Monitor costs**: Regularly review storage usage and costs

</details>
