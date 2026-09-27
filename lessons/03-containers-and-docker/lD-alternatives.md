---
title: Docker Alternatives
---

# Docker Alternatives

While Docker is the most popular containerization platform, they have a licensing setup so it's only free for: personal use, education, non-commercial open source projects amd small businesses (fewer than 250 employees OR less than $10 million in annual revenue). 

Luckily several alternatives exist for multiple different use cases and requirements, so for the sake of
completeness, here's a list of some for the interested people. Don't spend too much time on this, and consider doing your own research into each if you want to look more into FOSS products.

---

<details>
<summary style="font-size: 1.5em;">Podman</summary>

**Podman** (Pod Manager) is the most direct alternative to Docker and is often directly interchangeable.

### Features

- **Drop-in replacement**: Podman's CLI is designed to be compatible with Docker's CLI. Most `docker` commands work with `podman` with minimal changes (`alias docker=podman` is a common setup).
- **Daemonless**: Unlike Docker, which requires a background daemon (dockerd), Podman runs containers directly through fork/exec in user space. There is no root-owned daemon to attack.
- **Docker compatibility**: Pulls images from Docker Hub, builds from Dockerfiles, and works with Docker Compose (via `podman-compose` or the compose provider in newer versions).
- **Pod support**: Manages groups of containers as pods, mirroring the Kubernetes pod concept, which eases later migration to Kubernetes.

### Differences

- **Security model**: Rootless by default for regular users, using user namespaces. No setuid binaries. Docker historically ran everything through a root daemon, which is a larger attack surface.
- **Systemd integration**: `podman generate systemd` (and Quadlet in newer versions) produces systemd units for service management; Docker relies on its own daemon restart policies.
- **No daemon means no single point of failure**: A crashing build does not take down other running containers. The trade-off is weaker support for some Docker tooling that expects a live daemon API (Podman exposes a compatibility socket to mitigate this).

</details>

---

<details>
<summary style="font-size: 1.5em;">nerdctl</summary>

**nerdctl** (contaiNERD CTL) is a Docker-compatible CLI for containerd, the runtime that Kubernetes itself uses. [GitHub](https://github.com/containerd/nerdctl)

### Features

- **Docker-compatible UX**: Same command feel as `docker` and `podman`; CLI syntax follows Podman conventions. [Medium (author)](https://medium.com/nttlabs/nerdctl-359311b32d0e)
- **Compose support**: `nerdctl compose up` for multi-container stacks. [dev.to](https://dev.to/lovestaco/nerdctl-a-docker-compatible-cli-for-containerd-4i2l)
- **Rootless and UserNS-Remap modes**: Runs containers fully unprivileged, or remaps UIDs under a root containerd. [dev.to](https://dev.to/lovestaco/nerdctl-a-docker-compatible-cli-for-containerd-4i2l)
- **Cutting-edge containerd features**: Lazy pulling (eStargz, Nydus, OverlayBD, SOCI), image encryption via ocicrypt, and IPFS distribution. [GitHub](https://github.com/containerd/nerdctl)

### Differences

- **Goal**: Not competing with Docker. It exists to expose containerd features Docker has not adopted yet, and is useful for debugging Kubernetes nodes since Kubernetes containers live in containerd. [Medium](https://medium.com/nttlabs/nerdctl-359311b32d0e)
- **Requirements**: Needs containerd, CNI plugins for `nerdctl run`, and a separate BuildKit daemon for `nerdctl build`. Docker ships all of this as one bundle. [GitHub](https://github.com/containerd/nerdctl)
- **Flag coverage**: Not every Docker CLI flag is implemented; some unimplemented flags reflect containerd design, not missing functionality. [GitHub docs](https://github.com/containerd/nerdctl/blob/main/docs/command-reference.md)

</details>

---

<details>
<summary style="font-size: 1.5em;">Apptainer (formerly Singularity)</summary>

**Apptainer** is the dominant container system for HPC (high-performance computing), designed for multi-tenant clusters with untrusted users. [CIQ](https://ciq.com/products/apptainer)

### Features

- **Single-file SIF images**: Containers are packaged as one portable, verifiable file, easy to share across HPC environments. [CIQ](https://ciq.com/products/apptainer)
- **Rootless since inception**: Built for unprivileged users in shared environments; supports user-namespace mode without setuid. [Rootless Containers](https://rootlesscontaine.rs/getting-started/apptainer/)
- **Signing and encryption**: Cryptographic signing of images, plus native MPI support and Slurm/PBS scheduler integration. [CIQ](https://ciq.com/products/apptainer)
- **Runs Docker images**: Can execute Docker/OCI images directly, so existing images stay usable. [RUG HPC docs](https://docs.gcc.rug.nl/hyperchicken/apptainer/)

### Differences

- **Application-level, not system-level**: User IDs inside are mapped to a single user; it cannot run many system-level applications and does not fully meet OCI runtime requirements. [Rootless Containers](https://rootlesscontaine.rs/getting-started/apptainer/)
- **HPC-centric model**: Runs jobs as the invoking user, integrates with batch schedulers, no long-lived daemon or service model. Docker's daemon-and-service model is a poor fit for shared clusters where no user is trusted. [Apptainer docs](https://apptainer.org/admin-docs/2.6/security.html)
- **Fork history**: The original project moved to the Linux Foundation and was renamed Apptainer; a pre-fork open-source version continues as SingularityCE with setuid as default. [Rootless Containers](https://rootlesscontaine.rs/getting-started/apptainer/) Note that CIQ's page is a vendor site, so treat its marketing framing accordingly.

</details>

---

<details>
<summary style="font-size: 1.5em;">LXC</summary>

**LXC** (Linux Containers) is the older OS-level virtualization project that predates Docker and produces system containers rather than application containers. [Docker blog](https://www.docker.com/blog/lxc-vs-docker/)

### Features

- **System containers**: Runs a full OS userspace including systemd/PID 1, more like a lightweight VM than a single-process container. [Ars Technica forum](https://arstechnica.com/civis/threads/lcx-versus-docker-what-and-why.1508201/)
- **Kernel-native isolation**: Uses namespaces and cgroups directly for isolation and per-container resource management. [Docker blog](https://www.docker.com/blog/lxc-vs-docker/)
- **Persistent, long-lived environments**: Designed for hosts that run multiple services and need direct OS control, rather than ephemeral app packaging.

### Differences

- **No image-centric workflow**: Docker's core value is packaging, shipping, and running applications as immutable images; LXC is oriented toward long-running machine-like environments. [CyberPanel guide](https://cyberpanel.net/blog/lxc-vs-docker)
- **Different audience**: LXC suits admins wanting a lightweight VM replacement with OS control; Docker suits developers deploying apps consistently. [Docker blog](https://www.docker.com/blog/lxc-vs-docker/)
- **Weaker portability story**: No equivalent of Docker Hub image distribution as a first-class concept; migration is between directories and snapshots rather than images. Note that the two comparison sources here are blog/vendor content, not primary documentation.

</details>

---

<details>
<summary style="font-size: 1.5em;">Kaniko</summary>

**Kaniko** is not a runtime alternative but a build-tool alternative: it builds and pushes container images from Dockerfiles without any Docker daemon. [kaniko.org](https://kaniko.org/)

### Features

- **Daemonless builds**: Executes each Dockerfile command in userspace and snapshots the filesystem, then pushes the result to a registry; no Docker daemon or CLI involved. [Google Cloud Blog](https://cloud.google.com/blog/products/containers-kubernetes/introducing-kaniko-build-container-images-in-kubernetes-and-google-container-builder-even-without-root-access)
- **Unprivileged**: Runs in standard Kubernetes pods without privileged mode or root, avoiding the Docker-in-Docker security problem in CI/CD. [OneUptime](https://oneuptime.com/blog/post/2026-02-09-kaniko-build-images-no-docker/view)
- **Dockerfile compatible**: Supports standard Dockerfile workflows, multi-stage builds, layer caching, and registry credentials via Kubernetes secrets. [OneUptime](https://oneuptime.com/blog/post/2026-02-09-kaniko-build-images-no-docker/view)

### Differences

- **Build only**: Cannot run containers. You still need Docker, Podman, or a Kubernetes runtime to execute what it builds.
- **Kubernetes-native context**: Meant to run as a pod with the build context in a volume or bucket; it is not a laptop dev tool.
- **Project status**: Google archived the original repo in June 2025; it continues via the Chainguard fork maintained by the original creators. Mention this when choosing it for long-term CI infrastructure. [ComputingForGeeks](https://computingforgeeks.com/build-container-images-using-kaniko-in-kubernetes/)

</details>

[Next: Exercise 0 - Installation](eA-exercise-setup)
