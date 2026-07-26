# Exercise 3: Run your image

1. <details> <summary> Run your latest image </summary>
   The basic <code>docker run &lt;image&gt;</code> is pretty straight-forward. But what is your image id?

   <details><summary>Hint to find list of images</summary><code>docker images</code></details>
   </details>

2. <details> <summary> Solving issues with different arguments </summary>
   <a href="https://docs.docker.com/reference/cli/docker/container/run/">Full list of arguments here</a>
  
   -i (--interactive) for going "into" the container and run commands interactively.

   -d (--detach) to don't have the process run in your terminal but detached in the background instead.

   -p (--publish)
   <details><code>docker run -p 10000:8080 iris-train</code></details>

   -v (--volume) for mounting a directory, allowing you to access files there outside the container.
   <details><code>docker run -v ./artifacts:/usr/local/app/artifacts iris-train</code></details>

   -e (--env) is for setting environment variables, which can sometimes be useful whenever using env variables in your code. 
   </details>

3. <details> <summary> Check running containers </summary>
   Checking running containers can be done by <code>docker ps</code>. What about stopped containers?
   <details><code>docker ps -a</code></details>

   </details>

[Next: Exercise 4 - Push and Pull](G-exercise-push-pull.md)
