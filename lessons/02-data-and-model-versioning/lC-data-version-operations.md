# Data Version Operations

## Standard Flow

The standard workflow for versioning data with DVC follows these steps:

1. Add data files to DVC tracking
2. Commit the changes to Git
3. Push data to remote storage

## Demo Time

Let's walk through the practical steps:

### 1. Initialize DVC

```bash
dvc init
```

### 2. Add Files to DVC

```bash
dvc add data.csv
```

This creates a `.dvc` file that contains metadata about the data file.

### 3. Git Integration

```bash
git add data.csv.dvc .gitignore
git commit -m "Add data.csv to DVC tracking"
```

### 4. Push to Remote

```bash
dvc push
```

This uploads the actual data to your configured remote storage.

## Info in .dvc Files

When you run `dvc add`, a `.dvc` file is created with the following information:

```yaml
outs:
- md5: a1b2c3d4e5f6...
  size: 1048576
  nfiles: 1
  path: data.csv
```

- **path**: The relative path to the data file
- **hash**: The MD5 (or other) hash of the file content
- **size**: The size of the file in bytes
- **nfiles**: Number of files (for directories)

## Release Tags

Use Git tags to mark important versions:

```bash
git tag -a v1.0 -m "First stable dataset"
git push --tags
```

This allows you to easily return to specific data versions later.

## Examples

### GPT Development Timeline

![GPT Timeline](/images/data-and-model-versioning/gpt-timeline.png)

From https://www.researchgate.net/figure/Illustration-of-GPT-development-history-and-the-rise-of-ChatGPT-The-development-timeline_fig1_374092520

Versioning data is just as important as versioning models. The timeline above shows how model versions evolve over time, and the same applies to the data they were trained on.

## Best Practices

1. **Commit early, commit often**: Small, frequent commits make it easier to track changes
2. **Use meaningful messages**: Describe what changed and why
3. **Tag releases**: Mark important milestones with tags
4. **Document changes**: Keep a changelog for major data changes
