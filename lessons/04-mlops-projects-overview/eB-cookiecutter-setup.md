---
title: Cookiecutter Setup
---

# Exercise 1: Cookiecutter Setup

Make sure Cookiecutter Data Science is installed:
`pipx install cookiecutter-data-science`

Try running it with:
`ccds --version`

Does this command run and show a version number above 2.something? Great!

### Exercise 1.1: Start your DS project

Let's start a new project using Cookiecutter, inspired by CCDS's documentation on [their homepage](https://cookiecutter-data-science.drivendata.org/using-the-template/):

1. <details> <summary> Initialise CCDS in directory </summary>
   In the terminal, run <code>ccds</code>

   <pre><i><u>Discuss what each option does</u></i></pre>
   </details>

2. <details> <summary> Initialise git in the repo (if relevant)</summary>
   In the terminal, run <code>git init</code>
   </details>

3. <details><summary>Start tracking new files with git</summary>
   <tt> git add .</tt>

   <tt> git commit -m "Initial CCDS commit"</tt>

   <pre><i><u>What does <code>git add .</code> do?
   Also: what's in .gitignore?</u></i></pre>

  </details>

### Exercise 1.2: Push to GitHub

The pipeline you will build in the coming exercises ends up being tested in CI/CD, so the code needs a home on github.com. If you don't want to use your normal github.com user for this course, feel free to create a dummy one.

1. <details><summary>Create a new repo on github.com</summary>
   Create an empty repository (no README, no .gitignore, no license - you already have all of those).
   </details>

2. <details> <summary> Push your code to this new remote (<i>hint: <a href="https://lasselundstenjensen.github.io/itu-bds-sdse/lessons/git-basics/remote-and-fetch">these lessons covered it</a>)</i></summary>
   <tt>git remote add origin git@github.com:&lt;username&gt;/&lt;new_repo&gt;.git</tt>

   <tt>git push</tt>

   <pre><i><u>We are actually missing an argument to git push. Can you find out what?</u></i></pre>
   <details><summary>Hint:</summary>
   <tt>git push --set-upstream origin main</tt>
   </details>
   </details>

3. <details> <summary> Work on a branch </summary>
   Good developer practice is to not work directly on main since that is reserved for production code. Instead create a new branch for the coming exercises:

   <tt>git checkout -b w04-project-setup</tt>

   <pre><i><u>Why is it a good idea to keep main "green" (i.e. always working)?</u></i></pre>
   </details>

From here on, work on feature branches and merge them to main via pull requests once the exercises are done - that is the same workflow you will use in any real project.
