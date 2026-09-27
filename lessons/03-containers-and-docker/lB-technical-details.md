---
title: Technical Details
---

# Technical Details

Docker is a way to create a virtual instance of compute resources. However, unlike virtual machines with their own complete operating system, Docker uses lightweight containers instead.

The key thing is these containers have their own filesystem, dependencies, processes and such, which allows you to run an application in isolation from your operating system.

---

<details>
<summary style="font-size: 1.5em;">Containers vs Virtual Machines</summary>

### Virtual Machines (VMs)
- **Full OS**: Each VM runs a complete operating system with its own kernel
- **Heavyweight**: Requires significant resources (CPU, memory, storage)
- **Slow startup**: Booting a full OS takes time (seconds to minutes)
- **Strong isolation**: Complete separation from host and other VMs
- **Persistent**: Keeps files and processes open
- **Use case**: Running different OS versions or when full isolation is required (e.g., cyber security)

### Containers
- **Shared kernel**: Containers share the host OS kernel
- **Lightweight**: Only includes application and its dependencies, not a full OS
- **Fast startup**: Typically starts in milliseconds
- **Process-level isolation**: Applications run in isolated user spaces
- **Ephemeral**: Releases resources after completion
- **Use case**: Running multiple applications on the same OS with resource efficiency (e.g., MLOps)

</details>

---

<details>
<summary style="font-size: 1.5em;">Docker Architecture Overview</summary>

![Docker architecture](/images/containers-and-docker/docker-architecture.png)

Docker follows a client-server architecture:

- **Docker Daemon (dockerd)**: The background service that manages Docker objects (images, containers, networks, volumes)
- **Docker Client (docker)**: The CLI interface users interact with
- **Docker Registry**: Stores Docker images (Docker Hub is the default public registry)

When you run `docker build`, the client sends the build request to the daemon, which builds the image. When you run `docker run`, the daemon creates and starts a container from the image.

</details>

---

<details>
<summary style="font-size: 1.5em;">Three Components We Focus On</summary>

1. Dockerfiles
2. Docker images
3. Docker containers

You will explore each of these components more closely through the upcoming exercises, but a short introduction is in order.

### Dockerfiles

**Dockerfiles** are a set of instructions that will be used to build a **Docker image**. These instructions include information about the *base image* and the *commands* that will be executed (such as installing packages, copying files, etc).

A typical Dockerfile might look like:
```dockerfile
FROM python:3.14-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

- `FROM`: Specifies the base image
- `WORKDIR`: Sets the working directory in the container
- `COPY`: Copies files from the build context to the container
- `RUN`: Executes commands during the build process
- `CMD`: Specifies the command to run when the container starts

### Docker Images

A **Docker image** is the output from building the **Dockerfile**. They are:

- **Immutable**: Once built, images cannot be changed (any changes create a new image)
- **Layered**: Each instruction in the Dockerfile creates a layer; these layers are stacked to form the final image
- **Cached**: Each layer is cached, enabling fast rebuilds when only some parts change
- **Versioned**: Images can be tagged and versioned (e.g., `myapp:v1.0`, `myapp:latest`)

These images can be pushed to and pulled from image repositories. [Docker Hub](https://hub.docker.com/) is the default public registry, but organizations often use private registries (AWS ECR, Google Container Registry, etc.).

The layered structure means that when you rebuild an image after changing only one line in your application code, Docker will reuse all the cached layers up to the point where the change occurs, resulting in fast and efficient build times.

### Docker Containers

When you want to run the **Docker image**, it spins it up in a **Docker container** as an isolated runtime environment. The container lifecycle is:

1. **Create**: `docker create` creates a container from an image but doesn't start it
2. **Start/Run**: `docker run` creates and starts a container in one command
3. **Execute**: The container runs the specified command
4. **Stop**: `docker stop` gracefully stops a running container
5. **Exit**: The container shuts down and releases compute resources
6. **Remove**: `docker rm` removes the container (storage is released)

Although they are based on the immutable **Docker image**, it is possible to customize the **Docker container** at runtime. This can be done by:
- Overriding the default command
- Mounting volumes for persistent storage
- Passing environment variables
- Publishing ports

This approach also ensures the same environment for *development* and *production* work, eliminating the "works on my machine" problem.

</details>

[Next: Industry Usage](lC-industry-usage)
