# Exercise 3: Collaboration

You have your project set up locally, and you are tracking the relevant files through git. But what you really want is to make sure you set up a remote, say, in Github, so you can start collaborating on the same codebase.

Since you will be using github.com in this course, now is a good time to get started. If you don't want to use your normal github.com user (if you have one, not the ITU one) for this course, feel free to create a dummy one.

For this exercise, we want to simulate a a workflow where you (and a partner, if possible) contribute with code to a public repository. In this exercise you will therefore try out 1) setting up a new github repo, 2) push and pull new code to the repo, and 3) clone your partner's code.

1. <details> <summary> Create new repo on github.com </summary></details>
2. <details> <summary> Push your code to this new remote (<i>hint: <a href="https://lasselundstenjensen.github.io/itu-bds-sdse/lessons/git-basics/remote-and-fetch">these lessons covered it</a></i>)</summary>
   <tt>git remote add origin git@github.com:&lt;username&gt;/&lt;new_repo&gt;.git</tt>

   <tt>git push</tt>

   <pre><i><u>We are actually missing an argument to git push. Can you find out what?</u></i></pre>
   <details><summary>Hint:</summary>
   <tt>git push --set-upstream origin main</tt>
   </details>
   </details>
3. <details> <summary> Try to work on a partner's repo </summary>
   Go to a new unversioned/ungitted directory (<i>~/Projects for me</i>) and clone a partner's repo:

   <tt>git clone git@github.com:&lt;username&gt;/&lt;new_repo&gt;.git</tt>

   Now you have a local copy of the code that you can work with!
   </details>
4. <details> <summary> Try and push changes to your partner's repo </summary>
   Good developer practice is to not work directly on main since that is reserved for production code. Instead create a new branch:

   <tt>git checkout -b w07-model-training-script</tt>

   Next step is to make some meaningful changes. As hinted with the branch name, maybe you want to create the training script for the model.

   For now let's not be concerned with how train.py should look like. It depends on the project and such, but it generally takes data and model configurations as input and outputs a trained model. For now, let's just <tt>touch &lt;project_name&gt;/modeling/train.py</tt>.

   And then follow the typical git flow:

   <tt>git add &lt;project_name&gt;/modeling/train.py</tt>
   
   <tt>git commit -m "feat: model training script added"</tt>
   
   <tt>git push</tt>

   <pre><i><u>How does this git push differ from when you pushed to your own repo?</u></i></pre>

   </details>

This was a very preliminary example of working together on code. There are more aspects to it, such as branching strategies, code reviews and pull requests, but that will be covered in later lectures. For now, pat yourselves on the back for actually starting a data science project with a more clear strategy than 80% of companies!

[Back to Introduction](A-introduction.md)
