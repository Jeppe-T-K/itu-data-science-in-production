---
title: Exercise Push Pull
---

# Exercise 4: Push and pull

1. <details><summary> Find an interesting <a href="https://hub.docker.com/search?categories=Machine+Learning+%26+AI">AI/ML image</a> and pull it</summary>
   <code>docker pull jupyter/scipy-notebook</code>

   Or run it directly:
   <code>docker run jupyter/scipy-notebook</code>

   <pre>Question: Suggested arguments for running this image?</pre> 
   <details><summary>My suggestions:</summary><code>-p 10000:8888 -d</code></details>
   </details>

2. <details><summary> Pushing your image</summary>
   This will require setting up a user that can push to whichever registry you want to push it to. Generally it would follow this structure:
   <pre><code>
   docker login &lt;REGISTRY_HOST&gt;:&lt;REGISTRY_PORT&gt;
   docker tag &lt;IMAGE_ID&gt; &lt;REGISTRY_HOST&gt;:&lt;REGISTRY_PORT&gt;/&lt;APPNAME&gt;:&lt;APPVERSION&gt;
   docker push &lt;REGISTRY_HOST&gt;:&lt;REGISTRY_PORT&gt;/&lt;APPNAME&gt;:&lt;APPVERSION&gt;
   </code></pre>

   An example would look like:
   <pre><code>
   docker login repo.company.com:3456
   docker tag 19fcc4aa71ba repo.company.com:3456/myapp:0.1
   docker push repo.company.com:3456/myapp:0.1
   </code></pre>

   Or in our case with default Docker Hub:

   <pre><code>
   docker login --username jeppetk
   docker tag 90c46295c455 jeppetk/iris-train:initial
   docker push jeppetk/iris-train
   </code></pre>
   </details>

3. <details><summary> What do the hashes represent when pushing/pulling?</summary>
   It's the different layers. This can be confirmed with the  command <code>docker inspect 90c46295c455</code>
   </details>  

[Next: Exercise 5 - Clean up](H-exercise-cleanup.md)
