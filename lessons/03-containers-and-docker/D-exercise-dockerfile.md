---
title: Exercise Dockerfile
---

# Exercise 1: Create a training Dockerfile

> [!NOTE]
> **Learning outcomes**
> <i> By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Write/create a basic Dockerfile that can run a basic DS project [e.g., CCDS]</li>
> <li>Explain the workflow and processes for using Docker</li>
> <li>Motivate the typical ways why you would modify Docker containers at runtime</li>
> <li>Describe image repositories and explain their usage</li>
> </ul>


1. <details> <summary> Build out our CCDS repo with a train.py script </summary>
   Go to where you have your projects and run <code>ccds</code>. You can also initialise git with <code>git init</code>.

   To make it easy, there is a [../resources/train.py](../resources/train.py) that you can copy to your project. So for example:

   <pre><code>cp ../resources/train.py path/to/project/modeling/train.py</code></pre>

   </details>
2. <details> <summary> Create train.dockerfile </summary>
   You can use your favourite tool to do this. Or do:

   <pre><code>touch path/to/project/train.dockerfile</code></pre>
   </details>
3. <details> <summary> Find appropiate base-image </summary>
   python:3.9 for example.
   </details>
4. <details> <summary> Modify example Dockerfile to fit our use-case</summary>
   Inspired from <a href="https://docs.docker.com/get-started/docker-concepts/building-images/writing-a-dockerfile/">this website:</a>

   <pre><code>
   FROM python:3.12
   WORKDIR /usr/local/app
   # Install the application dependencies
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt
   # Copy in the source code
   COPY src ./src
   EXPOSE 5000
   # Setup an app user so the container doesn't run as the root user
   RUN useradd app
   USER app
   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
   </code></pre>
   </details>

[Next: Exercise 2 - Build the Image](E-exercise-build.md)
