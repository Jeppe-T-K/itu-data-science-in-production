---
title: Docker Quiz
---

# Quiz: Containers and Docker

*Each question has exactly one correct answer, marked with ✅.*

1. Why would you use Docker containers over virtual machines?

   - ✅- Containers are more lightweight and start much faster, because they share the host OS kernel instead of each running a full operating system
   - Containers give stronger isolation than VMs, since every container runs its own kernel
   - You wouldn't; VMs are cheaper on resources because they don't need a container runtime
   - Containers keep running in the background even after you close your terminal, unlike VMs

2. What happens when a container finishes running the command it was started with?

   - ✅ It stops and releases its compute resources, but its storage layer still exists until the container is removed with `docker rm`
   - It stays up and remains interactive, waiting for new commands
   - It automatically restarts the command unless you pass the `-s` (stop) argument to `docker run`
   - It is deleted immediately, including everything it wrote to disk

3. Is Docker always free to use?

   - ✅ No — Docker Desktop requires a paid license for large companies (250+ employees or $10M+ revenue), but Docker Engine is open source, so free alternatives like Podman can be used instead
   - No — Dockerfiles use proprietary technology, so you lock yourself into the Docker ecosystem and cannot switch later
   - Yes — all features are free, but you have to pay if you store more than 50 GB of images
   - Yes — Docker is fully free for everyone, including companies of any size

4. What is the working directory inside the container built from this Dockerfile?

   ```dockerfile
   FROM python:3.11
   WORKDIR /usr/local/app
   COPY requirements.txt ./
   RUN pip install --no-cache-dir -r requirements.txt
   COPY train.py ./
   CMD ["python", "train.py"]
   ```

   - ✅ `/usr/local/app`
   - `/` (the root directory)
   - `/app`
   - The directory where the Dockerfile is located on your machine

5. What does the `-v` argument to `docker run` do, and when would you use it?

   - ✅ It mounts a directory from the host into the container — useful when your training script should write artifacts to your own disk, so they survive after the container stops
   - It sets environment variables inside the container — useful for passing API keys
   - It makes the container run in verbose mode, printing extra logs
   - It gives the container a custom name, so you can refer to it in later commands

6. What does the `-p` argument to `docker run` do, and when would you use it?

   - ✅ It maps a port on your machine to a port in the container (`-p 10000:8080`)
   - It limits how many CPU cores the container may use (`-p 4`)
   - It sets the password for logging into the container (`-p lookimaplaintextpassword`)
   - It prints which ports were available on the host when the container started (`-p`)

7. What does the `-d` argument to `docker run` do, and when would you use it?

   - ✅ It runs the container detached in the background instead of in your terminal
   - It runs the container in debug mode, showing the full Docker daemon log
   - It does a dry run: the command is validated but nothing actually executes
   - It deletes the container automatically as soon as it exits

8. You run `docker rmi iris-train:latest` and get the error:

   ```text
   Error response from daemon: conflict: unable to delete 90c46295c455 (cannot be forced) -
   image is being used by stopped container 4f2b6c9e1a7d
   ```

   Why does this happen?

   - ✅ A stopped container based on that image still exists on disk, so the image is still referenced by it — you must remove the container with `docker rm` first
   - The image is hosted online, so deleting it would break dependencies for other users
   - The image is cached as a base layer for multiple other images, and Docker cannot delete shared layers
   - You can never delete images by name; only by image id

9. `docker images` is an alias for which other command?

   - ✅ `docker image ls`
   - `docker ls --image`
   - `docker ps -a`
   - `docker list images`

10. What happens when you run `docker run mlflow` but the `mlflow` image is not on your machine?

    - ✅ Docker first tries to pull the image from the default registry (Docker Hub) and then runs it, if the pull succeeds
    - It fails immediately with the error `image not found` — you must pull manually first
    - It first tries to build the image from a local `Dockerfile` in your current directory
    - It runs an empty container and waits for you to install mlflow inside it

11. Bonus: What is Docker's AI agent called?

    - ✅ Gordon
    - Whale
    - Rekcod
    - Dan
