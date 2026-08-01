# Data Version Operations

## Standard Flow

The standard workflow for versioning data with DVC follows these steps:

0. Initialize DVC
1. Add data files to DVC tracking
2. Add DVC metadata files to git
3. Commit the changes to git
4. Push the changes to git
5. Push data to remote storage

---

<details>
<summary style="font-size: 1.5em;">Demo Time!</summary>

### 0. Initialize DVC

```bash
dvc init
```

### 1. Track files with DVC

```bash
dvc add data.csv
```

This creates a `.dvc` file that contains metadata about the data file.

### 2. Git add

```bash
git add data.csv.dvc
```

### 3. Git commit

```bash
git commit -m "Add data.csv to DVC tracking"
```

### 4. Push to Remote

```bash
dvc push
```

This uploads the actual data to your configured remote storage.

</details>

---

<details>
<summary style="font-size: 1.5em;">Info in .dvc Files</summary>

When you run `dvc add`, a `.dvc` file is created with the following information:

<details>
<summary style="font-size: 1.2em;">Demo example</summary>

```yaml
outs:
- md5: a1b2c3d4e5f6...
  size: 1048576
  nfiles: 1
  path: data.csv
```

**Metadata explanation:**

- `path`: The relative path to the data file
- `hash/md5`: The MD5 (or other) hash algorithm/value of the file content
- `size`: Size of the file in bytes
- `nfiles`: Number of files (for directories)

</details>


<details>
<summary style="font-size: 1.2em;">Project example</summary>

```yaml
md5: 4feb90a977597b6b3f3b34dc2d4d3711
frozen: true
deps:
- checksum: '"0fbbe2410507589f580fd0a63f0b10f8"'
  size: 114686
  hash: md5
  path: https://itudsip.hel1.your-objectstorage.com/data/images/metadata.csv
outs:
- md5: 0fbbe2410507589f580fd0a63f0b10f8
  size: 114686
  hash: md5
  path: metadata.csv
```

**Metadata explanation:**

- `md5`: Checksum of the DVC file itself (identifies this specific version of the pipeline stage)
- `frozen: true`: The stage is locked and will not be re-executed even if its dependencies change, until explicitly unlocked with `dvc unlock`
- `deps`: Dependency entries. Only present when dvc import or dvc import-url are used to generate this .dvc file
  - `checksum`: Hash of the dependency file content
  - `size`: Size of the file in bytes
  - `hash`: Hash algorithm used (md5, sha256, etc.)
  - `path`: Path to the dependency (relative to wdir, which defaults to the file's location)
- `outs` (outputs): Files produced by this stage
  - `md5`: Hash of the output file content
  - `size`: Size of the file in bytes
  - `hash`: Hash algorithm used
  - `path`: Path to the file or directory relative to working directory

</details>

</details>

---

<details>
<summary style="font-size: 1.5em;">Release Tags</summary>

![GPT Timeline](/images/data-and-model-versioning/gpt-timeline.png)
From https://www.researchgate.net/figure/Illustration-of-GPT-development-history-and-the-rise-of-ChatGPT-The-development-timeline_fig1_374092520

- Do it through git
  - `git tag -a "v2.0" -m "imagenet v2.0"`

- Examples:
  - GPT
  - Do you have any?

</details>
