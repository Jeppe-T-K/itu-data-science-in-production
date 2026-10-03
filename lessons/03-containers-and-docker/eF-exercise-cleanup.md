---
title: Clean up
---

# Exercise 5: Clean up

1. <details> <summary>Check local images</summary>
   <code>docker images</code>

   What about images that are not tagged or only exist as intermediate layers?

   <details><summary>Show all images</summary><code>docker images -a</code></details>

   How much space are your images taking up in total?

   <details><summary>Show the total size of your images</summary><code>docker system df</code> gives you an overview of how much disk space Docker is using, including the total size of all your images. <code>docker images</code> also shows the size of each individual image.</details>
   </details>

2. <details> <summary>Remove images</summary>
   Try removing one of your images by name:

   <code>docker rmi iris-train:latest</code>

   <pre>Question: The image is still listed by docker images. Why?</pre>
   <details><summary>Answer</summary>Removing an image by <i>name</i> only removes that <i>tag</i>. If the same image has other tags (like <code>latest</code> and the registry name <code>jeppetk/iris-train:initial</code>), the image itself is still referenced by the remaining tags and stays on disk.</details>

   Now try removing it by <i>image id</i> instead:

   <code>docker rmi 90c46295c455</code>

   <details><summary>What happens?</summary>Removing by image id deletes the image completely (all tags pointing to it are removed), as long as nothing else references it.</details>

   Did you get an error like <code>image is being used by stopped container</code>?

   <details><summary>The common issue: images attached to containers</summary>You cannot remove an image that a container is based on. This goes for <b>running</b> containers too: if you try <code>docker rmi</code> on an image that a running container uses, Docker refuses with <code>image is being used by running container</code> — and even <code>docker rmi -f</code> cannot fully delete it while the container is running. You need to stop the container first, remove it (see step 3), and then remove the image. <code>docker rmi -f</code> does force removal past <i>stopped</i> containers — use with care.</details>
   </details>

3. <details> <summary>Remove containers</summary>
   Stopped containers pile up quickly. Remove one by name or id:

   <code>docker rm &lt;CONTAINER_ID&gt;</code>

   Or all stopped containers at once:

   <code>docker container prune</code>

   <pre>Question: What happens to the data the container wrote to a mounted volume?</pre>
   <details><summary>Answer</summary>Nothing! Removing a container does <b>not</b> delete your data. Anything written to a volume or bind mount (like <code>-v ./artifacts:/usr/local/app/artifacts</code>) still lives on your computer, and you can access it again from a new container mounting the same volume.</details>

   The <code>-v</code> flag on <code>docker rm</code> exists for the opposite reason: <code>docker rm -v &lt;CONTAINER_ID&gt;</code> removes the <i>anonymous volumes</i> that Docker created for the container. Named volumes and bind mounts are never touched, so your mounted data stays on your computer either way.
   </details>

4. <details> <summary>Prune images and containers</summary>
   Docker offers a broom for sweeping everything unused away in one go:

   <code>docker image prune</code>

   <code>docker container prune</code>

   You may have seen the word <i>dangling</i> in the output. What does it mean?

   <details><summary>What dangling images are</summary>A dangling image is an image layer that no longer has a tag and is not referenced by any tagged image. This typically happens when you rebuild an image with the same tag (like <code>latest</code>): the old image loses its tag and becomes <code>&lt;none&gt;:&lt;none&gt;</code>. They show up in <code>docker images</code> as <code>&lt;none&gt;</code> and are safe candidates for cleanup.</details>

   <b>Warning:</b> pruning cannot be undone. Usually that is fine — you can build your own images again from the Dockerfile, and pull the rest down again. But an image with no Dockerfile behind it is gone for good. <code>docker image prune -a</code> removes <i>all</i> unused images, not just dangling ones, so think twice before adding <code>-a</code>.
   </details>

---

[Back to Introduction](introduction)
