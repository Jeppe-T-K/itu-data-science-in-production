---
title: Create a Dockerfile
---

# Exercise 1: Create a training Dockerfile

> **Note — Learning outcomes**
> <i> By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Write/create a basic Dockerfile that can run a basic DS project</li>
> <li>Explain the workflow and processes for using Docker</li>
> <li>Motivate the typical ways why you would modify Docker containers at runtime</li>
> <li>Describe image repositories and explain their usage</li>
> </ul>


1. <details> <summary> Set up a project folder with the training script </summary>
   Create a new project directory and copy in the training script from <code>resources/train.py</code>:

   <pre><code>mkdir my-docker-project
   cp resources/train.py my-docker-project/
   cd my-docker-project</code></pre>

   The script trains a LogisticRegression classifier on the Iris dataset and saves the model to <code>artifacts/model.pkl</code>. You will also need a <code>requirements.txt</code> listing the packages it depends on.
   </details>
2. <details> <summary> Create train.dockerfile </summary>
   You can use your favourite tool to do this. Or do:

   <pre><code>touch path/to/project/train.dockerfile</code></pre>
   </details>
3. <details> <summary> Find appropiate base-image </summary>
   A <code>python:3.x</code> image, for example <code>python:3.11</code>.
   </details>
4. <details> <summary> Modify example Dockerfile to fit our use-case </summary>
   Inspired from <a href="https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/">this website:</a>

   <pre><code>
   FROM python:3.11
   WORKDIR /usr/local/app
   # Install the application dependencies
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt
   # Copy in the source code
   COPY train.py ./
   # Setup an app user so the container doesn't run as the root user
   RUN useradd app
   USER app
   CMD ["python", "train.py"]
   </code></pre>
   </details>

[Next: Exercise 2 - Build the Image](exercise-build)
