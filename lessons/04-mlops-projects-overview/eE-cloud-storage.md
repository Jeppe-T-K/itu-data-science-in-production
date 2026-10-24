---
title: Cloud Storage
---

# Exercise 4: Data and Models in S3-Compatible Object Storage

Your pipeline now runs locally, but everything it produces lives on your laptop. If a teammate (or a CI machine) clones your repo, `data/raw/` and `models/` are empty. Git is for code — data and model artifacts belong somewhere else.

In practice, that "somewhere else" is almost always an object store speaking the **S3 API**. The S3 protocol started as an AWS service, but is now the de-facto standard supported by everyone: DigitalOcean Spaces, Google Cloud Storage (S3-compatible API), Azure (via gateways), and self-hosted solutions like [MinIO](https://min.io/). Learning the S3 API once means you can talk to all of them.

In this exercise you will make your pipeline pull its raw data from and push its model artifact to an S3-compatible bucket, first locally with MinIO, then (optionally) against a real cloud provider.

### Exercise 4.1: A local object store with MinIO

MinIO is an S3-compatible object store you can run locally — perfect for development, and it means you don't need cloud credentials to do this exercise. You already know the tool for the job from the containers lesson:

1. <details><summary>Start MinIO in Docker</summary>
   <pre><code>docker run -d --name minio \
     -p 9000:9000 -p 9001:9001 \
     -e MINIO_ROOT_USER=minioadmin \
     -e MINIO_ROOT_PASSWORD=minioadmin \
     minio/minio server /data --console-address ":9001"</code></pre>

   Open http://localhost:9001 and log in with <code>minioadmin</code>/<code>minioadmin</code>. The console looks different from AWS, but every button here has an S3-API equivalent.
   </details>

2. <details><summary>Create a bucket for your project</summary>
   Create a bucket, e.g. <code>my-ds-project</code>, either via the console or with <code>mc</code>.

   <pre><i><u>Given the data stages from the lecture, what "folders" (S3 uses key prefixes, not real folders) would you create inside the bucket?</u></i></pre>
   </details>

### Exercise 4.2: Talk S3 from Python with boto3

`boto3` is the standard Python client for the S3 API — it works against AWS, Spaces, and MinIO alike, you just point it at a different `endpoint_url`.

1. <details><summary>Install and configure credentials <b>outside</b> version control</summary>
   Add <code>boto3</code> to your project dependencies. Then create a <code>.env</code> file (check that <code>.gitignore</code> ignores it — CCDS does this for you):

   <pre><code>AWS_ACCESS_KEY_ID=minioadmin
   AWS_SECRET_ACCESS_KEY=minioadmin
   S3_ENDPOINT_URL=http://localhost:9000
   S3_BUCKET=my-ds-project</code></pre>

   <pre><i><u>Why must credentials never be committed? What could someone do with a leaked key to a real cloud bucket?</u></i></pre>
   </details>

2. <details><summary>Write a small storage helper</summary>
   In your package, add helpers for upload/download. The interesting part is that the client is configured from the environment, so the same code works against MinIO <i>and</i> a real cloud:

   <pre><code>import os
   import boto3

   def s3_client():
       return boto3.client(
           "s3",
           endpoint_url=os.environ["S3_ENDPOINT_URL"],
       )</code></pre>

   Try listing the objects in your bucket from Python (<code>client.list_objects_v2</code>) to verify the connection.
   </details>

3. <details><summary>Upload something</summary>
   Upload the raw data from Exercise 3 to <code>s3://&lt;bucket&gt;/data/raw/...</code> and verify you can see it in the MinIO console (or with <code>mc</code>).

   <pre><i><u>What happens if you upload the same key twice? What does that tell you about using the S3 key as your artifact "versioning"?</u></i></pre>
   </details>

### Exercise 4.3: Put the pipeline on storage

Now make the pipeline actually use the bucket:

1. <details><summary>Ingestion pulls from S3</summary>
   Change the data ingestion step to download the raw data from the bucket to <code>data/raw/</code> if it isn't there already (rather than generating it locally). Local files are now a <i>cache</i> of the bucket, not the source of truth.
   </details>

2. <details><summary>Training publishes to S3</summary>
   After training and evaluation, upload the serialized model and <code>metrics.json</code> to the bucket under a key that includes some notion of version or timestamp, e.g. <code>models/&lt;version&gt;/model.pkl</code>.
   </details>

3. <details><summary>Prove it is reproducible</summary>
   Delete <code>data/</code> and <code>models/</code> locally and re-run <code>make pipeline</code>. Then simulate a fresh teammate: clone your repo into a new directory and run the pipeline there.

   <pre><i><u>What still breaks on a fresh clone that didn't break on your machine? (hint: where do the Python dependencies come from?)</u></i></pre>
   </details>

### Exercise 4.4: Discuss

<pre><i><u>In pairs:

<li>Why do you think basically every MLOps platform (including the tools from the data versioning lesson, whose DVC remotes you have already met) speaks S3?</li>
<li>What are the trade-offs of storing artifacts in a bucket vs. committing them to git vs. attaching them to a CI run?</li>
<li>Your MinIO data disappears when you delete the container. What would you use instead for something you care about?</li></u></i></pre>

<details><summary>Optional: try a real provider</summary>

If you have a cloud account (DigitalOcean gives free credit to students), create a Spaces bucket (or an S3 bucket on AWS), generate an access key, and point your <code>.env</code> at it. Nothing in your code should need to change — that is the point of the S3 API being a standard. Note that Spaces uses <code>https://&lt;region&gt;.digitaloceanspaces.com</code> as the endpoint URL.

</details>

[Next: Exercise 5 - CI/CD](cicd)
