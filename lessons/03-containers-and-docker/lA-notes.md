---
title: Introduction to Docker
---

# Prelude

Before getting started with using Docker, it's a good idea to motivate and explain _what_ Docker actually is so you don't just type commands and see stuff happen.

---

<details>
<summary style="font-size: 1.5em;">Why?</summary>

![Works on my machine](https://blog.codinghorror.com/content/images/2025/05/works-on-my-machine-v2-2025-jon-galloway-2-1.png)

To avoid the above from happening. Or even shorter -- reproducibility.

Docker essentially allows running code in an isolated and agnostic way from the operating system, making it crucial for reproducible, automated workflows for both normal DevOps and MLOps.

Itemised list for convenience:

* Reproducible and consistent delivery
* CI/CD pipelines
* Portable and collaborative
* Efficient use of resources

> **Note:** Fun fact: Docker is written in Go.

Standardization is key for team collaboration. Without it, data scientists and developers waste time debugging environment differences rather than building out the product. Docker solves this by packaging applications and all their dependencies into consistent, portable units.

</details>

[Next: Technical Details](lB-technical-details)
