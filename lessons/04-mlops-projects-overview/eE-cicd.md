---
title: CI/CD
---

# Exercise 4: Automate with GitHub Actions (CI/CD)

Right now, your pipeline runs when _you_ remember to run it. In the lecture's maturity levels, that is stage 1 at best. The defining trait of stage 2 is that every change to the code is automatically built and delivered - continuous integration and continuous delivery.

There are two parts to getting there. First, express your pipeline as a **Dagger workflow in Go**, the way you learned in the [workflows lesson](https://lasselundstenjensen.github.io/itu-bds-sdse/lessons/workflows-creating-and-running-a-dagger-pipeline/new-dagger-pipeline). Then, let GitHub run that same workflow for you on every push: a workflow file in your repo describes what to do, and GitHub runs it on a fresh virtual machine.

In this exercise you will turn the `make pipeline` chain from Exercise 3 into a Dagger pipeline - and then put it in CI so it produces your model artifact automatically.

## Exercise 4.1: The pipeline as a Dagger workflow

### Exercise 4.1.a: Create the workflow

Carry your project forward from Exercise 3. You will add the pipeline files next to the CCDS structure, so your repo looks like:

```
my-ds-project/
├── go.mod              <- the Dagger pipeline's Go module
├── pipeline.go         <- the Dagger pipeline
├── src/                <- your pipeline steps (dataset, features, train, predict)
├── data/
├── models/
└── Makefile            <- the old way of chaining the steps
```

1. <details><summary>Set up the Go module</summary>

   You need [Go](https://go.dev/dl/) installed, and a running container runtime (Docker Desktop or similar) - Dagger starts its engine through it.

   In the repo root, initialize the module and create the pipeline file, like in the [lesson](https://lasselundstenjensen.github.io/itu-bds-sdse/lessons/workflows-creating-and-running-a-dagger-pipeline/new-dagger-pipeline). We put the module in the repo root rather than its own directory, because the pipeline needs to see your whole project:

   ```bash
   go mod init daggerpipeline
   touch pipeline.go
   ```

   </details>

2. <details><summary>Start from the lesson template</summary>

   Start from the skeleton you know from the lesson: a <code>main</code> that creates a shared context and calls the pipeline function, and inside it a Dagger client, a container built from a Python image, and a step that runs one of your Python files:

   ```go
   package main

   import (
       "context"
       "fmt"

       "dagger.io/dagger"
   )

   func main() {
       ctx := context.Background()

       if err := Pipeline(ctx); err != nil {
           fmt.Println("Error:", err)
           panic(err)
       }
   }

   func Pipeline(ctx context.Context) error {
       client, err := dagger.Connect(ctx)
       if err != nil {
           return err
       }
       defer client.Close()

       // ...
   }
   ```

   Two differences from the lesson's <code>hello.py</code> example:

   - Your code and dependencies live in the **repo root**, the same directory you run from - so the host directory is simply <code>.</code>.
   - Your steps need your project's dependencies installed, not just a plain Python image.
   </details>

3. <details><summary>Fill in the data step</summary>

   Load the project into a Python container, install your dependencies from the lockfile, and run the data step - the same command your <code>make data</code> target runs. Then export the container's <code>data/</code> back to the host:

   ```go
   func Pipeline(ctx context.Context) error {
       client, err := dagger.Connect(ctx)
       if err != nil {
           return err
       }
       defer client.Close()

       // Only copy what the pipeline needs into the container.
       // data/ and models/ are not on the list, so the container
       // always starts from a fresh machine - the pipeline must
       // build everything itself.
       project := client.Host().Directory(".", dagger.HostDirectoryOpts{
           Include: []string{
               "src",
               "pyproject.toml",
               "uv.lock",
           },
       })

       // Base container: uv image with your Python version,
       // with dependencies installed from the lockfile
       python := client.Container().
           From("ghcr.io/astral-sh/uv:python3.14-bookworm").
           WithDirectory("/app", project).
           WithWorkdir("/app").
           WithExec([]string{"uv", "sync"})

       // Stage 1: Data
       data := python.WithExec([]string{"uv", "run", "python", "src/dataset.py"})
       if _, err := data.Directory("data").Export(ctx, "data"); err != nil {
           return err
       }

       return nil
   }
   ```

   Run <code>go mod tidy</code> to fetch the Dagger dependency.

   Notes:

   - The <code>uv</code> commands mirror your Makefile targets: <code>uv sync</code> installs exactly the lockfile-pinned dependencies, and <code>uv run python src/dataset.py</code> is the same command <code>make data</code> runs. Match the image's Python version to the one in your <code>pyproject.toml</code>. If your scaffold does not use uv, use a plain Python image (e.g. <code>python:3.12-bookworm</code>) and <code>pip install -e .</code> instead.
   - <code>Export</code> copies the container's <code>data/</code> directory out to the repo's <code>data/</code> - the same <code>data/</code> you inspected after <code>make data</code> in Exercise 3.
   </details>

4. <details><summary>Add the remaining stages: features, train, predict</summary>

   The pattern is now the same for every stage: take the container, run the step's command, export what the step produced. Because each stage starts from the previous stage's container, the steps chain in order - and Dagger caches every command, so a stage whose inputs did not change is not re-run.

   ```go
       // Stage 2: Features
       features := data.WithExec([]string{"uv", "run", "python", "src/features.py"})
       if _, err := features.Directory("data").Export(ctx, "data"); err != nil {
           return err
       }

       // Stage 3: Train
       train := features.WithExec([]string{"uv", "run", "python", "src/modeling/train.py"})
       if _, err := train.Directory("models").Export(ctx, "models"); err != nil {
           return err
       }

       // Stage 4: Predict
       predict := train.WithExec([]string{"uv", "run", "python", "src/modeling/predict.py"})
       if _, err := predict.Directory("models").Export(ctx, "models"); err != nil {
           return err
       }
   ```

   Adapt the commands to whatever your Makefile targets run, and match the directories your project actually writes to (<code>data/raw/</code>, <code>data/processed/</code>, <code>models/</code>, ...). If a step writes to multiple directories, export each of them.
   </details>

### Exercise 4.1.b: Run the pipeline locally

1. <details><summary>Run it</summary>

   From the repo root:

   ```bash
   go run pipeline.go
   ```

   The first run takes a while - Dagger pulls the Python image and installs your dependencies. Then inspect what appeared under <code>data/</code> and <code>models/</code>, like you did in Exercise 3.
   </details>

2. <details><summary>Feel the caching</summary>

   Run <code>go run pipeline.go</code> again, immediately. Then make a small change to <code>src/dataset.py</code> and run once more.

   - On the second run, nothing re-runs: every command's inputs are unchanged, so Dagger replays the cached layers and only re-exports the artifacts. <code>make</code> re-ran everything, every time.
   - On the third run, the container image and dependency install are cached, but the data step's input (your source code) changed, so the data step - and everything downstream of it - re-runs.
   </details>

## Exercise 4.2: Put it in CI/CD

### Exercise 4.2.a: Create the workflow file

1. <details><summary>Create the workflow file</summary>

   GitHub Actions gives you CI/CD for free: a workflow file in your repo describes what to do on every push or pull request, and GitHub runs it on a fresh virtual machine. Since your pipeline is now a Dagger workflow, you embed it the way you learned in the [workflows lesson](https://lasselundstenjensen.github.io/itu-bds-sdse/lessons/workflows-embedding-in-a-github-workflow/running-workflows).

   Create <code>.github/workflows/ci.yml</code>:

   ```yaml
   name: CI

   on:
     push:
       branches: [main]
     pull_request:

   jobs:
     pipeline:
       runs-on: ubuntu-latest
       steps:
         - name: Checkout code
           uses: actions/checkout@v7.0.1

         - name: Set up Go
           uses: actions/setup-go@v7.0.0
           with:
             go-version: 1.26.1
             cache: false

         - name: Run Dagger pipeline
           uses: dagger/dagger-for-github@v8.4.1
           with:
             verb: run
             args: go run pipeline.go
             version: "0.21.11"
   ```

   Commit it, push, and watch the run under the <b>Actions</b> tab on GitHub.

   The workflow does not restate your pipeline - it runs the <i>same</i> <code>pipeline.go</code> you ran locally. One definition, two environments. That is the payoff for expressing the pipeline as code in 4.1.
   </details>

### Exercise 4.2.b: Watch it run

1. <details><summary>Read the run</summary>

   Remember: the runner is a fresh machine - nothing in <code>data/</code> or <code>models/</code> exists until your pipeline creates it. You already designed for that in 4.1.a, by not copying them into the container.

   The exports land in the repo root on the runner, and both are deleted with the machine when the job ends. The runner needs Go and the Dagger engine - but no Python, no uv, no virtualenv: your pipeline brings its own in the container.
   </details>

### Exercise 4.2.c: Publish the artifact

Continuous _delivery_ means the pipeline's output is produced automatically, not just run.

1. <details><summary>Upload the model</summary>

   Add a step that uploads what the pipeline produced with <a href="https://github.com/actions/upload-artifact"><code>actions/upload-artifact</code></a>:

   ```yaml
         - name: Upload model artifact
           uses: actions/upload-artifact@v4
           with:
             name: model
             path: models/
   ```

   Once the run finishes, download the artifact from the run page. This is exactly how trained models are handed from a training workflow to whatever comes next: a validator, a deployment job, or a colleague.
   </details>

[Back to Introduction](exercise-intro)
