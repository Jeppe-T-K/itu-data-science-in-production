---
title: Exercise Build
---

# Exercise 2: Build the image

1. <details> <summary> Build the train.dockerfile </summary>
   Normally you just need to run <code>docker build .</code>, but since we want to use a specific .dockerfile, we have to modify it slightly:
   <details><summary>Hint:</summary><code>docker build -f train.dockerfile .</code></details>

   </details>

2. <details> <summary> Add a meaningful name by version/tag </summary>
   Just building the image can make it hard to find again.

   Adding tags when building the image is typically the easiest way to deal with this. You can also add tags after it has been built but then you need to know the image id.

   <details><summary>Hint:</summary><code>docker build -f train.dockerfile -t iris-train:1.0.0 .</code></details>
   </details>

3. <details> <summary> Changing files/fixing errors </summary>
   It's possible that you need to rebuild the image because of some errors. Can you spot what the error is now?
   
   <details><summary>Hint:</summary>Python version and requirements.txt does not have scikit-learn</details>
   
   <pre>Question: why is it faster building it now?</pre>
   </details>

[Next: Exercise 3 - Run your Image](F-exercise-run.md)
