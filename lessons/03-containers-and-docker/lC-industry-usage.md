---
title: Docker in Practice
---

# Docker in Practice

Docker has become a standard tool in the software industry for packaging and deploying applications. Its adoption spans from small startups to large enterprises, particularly in DevOps and MLOps workflows.

---

<details>
<summary style="font-size: 1.5em;">Common Use Cases in Industry</summary>

### Development Environment Standardization

One of the most widespread uses of Docker is standardizing development environments across teams.
This is especially valuable in data science teams where projects often have complex dependency chains (specific versions of Python, ML libraries, system libraries).

### Microservices Architecture

In modern software development, applications are increasingly built as collections of microservices rather than monolithic applications.

- **One service per container**: Each microservice runs in its own container
- **Independent scaling**: Services can be scaled independently based on demand
- **Technology diversity**: Different services can use different technology stacks
- **Isolated failures**: A failure in one service doesn't crash the entire application

At Maersk you might for example see Docker being used multiple ways:
- Data pipelines with tenant-specific images built on top of main image
- Internal MLOps orchestrator
- AI/ML (sub)projects written in either Python or R (or both)
- Various internal applications built by different teams

### ML Model Serving and Training Environments

In machine learning workflows, Docker plays several crucial roles:

**Training Environments:**
- Pre-configured containers with all necessary dependencies (Python, CUDA, ML frameworks)
- Reproducible training across different hardware configurations
- Easy sharing of training environments between researchers
- Version control of training environments alongside model code

**Model Serving:**
- Model inference APIs packaged as containers
- Consistent environment from training to production
- Easy deployment across different serving platforms (cloud, on-premise, edge)
- Quick rollback to previous model versions

### CI/CD Pipelines

Docker is a fundamental building block for Continuous Integration and Continuous Deployment pipelines:

- **Build stage**: Code is compiled and dependencies are installed in a container
- **Test stage**: Automated tests run in isolated containers
- **Deploy stage**: Application containers are deployed to staging/production

Benefits:
- Consistent build environments across all pipeline stages
- Parallel execution of tests in separate containers
- Easy cleanup of build artifacts after pipeline completion
- Reproducible builds that can be traced and debugged

</details>

---

<details>
<summary style="font-size: 1.5em;">Orchestration</summary>

When you have many different containers working together, you can orchestrate them in different ways. It is not something that we will be focusing much on here, but it gives you some context to what you might encounter in some form or another.

A simple approach *Docker Compose* is for manual setup of multiple containers. It's useful for development work, and it is a single YAML configuration file that essentially has instructions on how to run multiple containers at once.

A more complex approach is *Kubernetes*, which is for a more consistently deployed service. It automates deployment, scaling and management of containerized applications. This means you have a cluster of machines that runs the pods that hold the containers. It uses more advanced health checks, load balancing storage orchestration etc, which is for example useful for when web sites need to serve higher demands.

</details>

---

<details>
<summary style="font-size: 1.5em;">Image Repositories</summary>

### Docker Hub

[Docker Hub](https://hub.docker.com/) is Docker's default public registry:
- Hosts official images for popular software (Python, Node.js, PostgreSQL, etc.)
- Allows users to share their own images publicly
- Provides build automation (automatically build images when source code changes)
- Offers both free and paid tiers

### Private Registries

For enterprise use, private registries provide:
- Control over who can access images
- Compliance with security and data privacy requirements
- Integration with existing authentication systems
- On-premise or cloud-based hosting

Popular private registry options:
- **AWS Elastic Container Registry (ECR)**: Managed Docker registry on AWS
- **Google Container Registry (GCR)**: Google Cloud's registry service
- **Azure Container Registry (ACR)**: Microsoft's offering on Azure
- **GitHub Container Registry**: Integrated with GitHub repositories
- **Self-hosted**: Options like Docker Registry, Harbor, or Nexus

</details>

---

<details>
<summary style="font-size: 1.5em;">Versioning and Rollback Strategies</summary>

You can tag your images so you can easily find the exact image you want/need.

**Tagging strategies:**
- **Semantic versioning**: `v1.0.0`, `v1.1.0` for release versions
- **Git SHA**: `git-abc1234` for traceability to source code
- **Timestamp**: `2025-09-27` for build date tracking
- **Environment**: `dev`, `staging`, `prod` for different environments
- **Latest**: Floating tag for the most recent build (use with caution)

</details>

---

<details>
<summary style="font-size: 1.5em;">Best Practices for Production</summary>

1. **Start with the slimmest image** and choose the right image for your use-case
2. **Handle secrets properly** (e.g., API credentials) - never hardcode them in images, but read them from environment variables which can be specified at runtime
3. **Decouple applications** by focusing on a single concern per container
4. **Pin base image versions** but update them often to get security fixes
5. **Test with CI/CD pipelines** to ensure your containers work correctly

</details>

[Next: Docker Alternatives](lD-alternatives)
