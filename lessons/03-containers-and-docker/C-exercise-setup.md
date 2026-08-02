---
title: Exercise Setup
---

# Exercise 0: Installation

As described in the previous section, Docker is really a way to run code in a reproducible way.

## Exercise 0: Installation

1. <details> <summary> Install Docker Desktop+Engine</summary>
   If you haven't already, you need to make sure to install <a href="https://docs.docker.com/engine/install/">Docker Engine</a>. The easiest way is to do it through <a href="https://docs.docker.com/desktop/">Docker Desktop</a>. For Windows users, you will also need to install <a href="https://learn.microsoft.com/en-us/windows/wsl/install">WSL</a> if you haven't already.
   </details>
2. <details> <summary> Test your Docker installation is working</summary>
   In the terminal, run <code>docker run hello-world</code>

   If it works, great! If not, try googling the issue.

   A common issue is that the Docker daemon not running. If you installed it with Docker Desktop, make sure that program is running.
   </details>

[Next: Exercise 1 - Create a Dockerfile](D-exercise-dockerfile.md)
