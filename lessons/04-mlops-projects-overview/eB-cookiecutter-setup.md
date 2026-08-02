---
title: Cookiecutter Setup
---

# Exercise 0-1: Installation and Setup

Make sure Cookiecutter Datascience is installed:
`pipx install cookiecutter-data-science`

Try running it with:
`ccds --version`

Does this command run and show a version number above 2.something? Great!

### Exercise 1: Start your DS project

Next we want to start using Cookiecutter with one of our projects. Let's be inspired by CCDS's documentation on [their homepage](https://cookiecutter-data-science.drivendata.org/using-the-template/):

1. <details> <summary> Initialise CCDS in directory </summary>
   In the terminal, run <code>ccds</code>

   <pre><i><u>Discuss in pairs what each option does</u></i></pre>
   </details>
   
2. <details> <summary> Initialise git in the repo (if relevant)</summary> 
   In the terminal, run <code>git init</code>
   </details>

3. <details><summary>Start tracking new files with git</summary>
   <tt> git add .</tt>

   <tt> git commit -m "Initial DDCS commit"</tt>

   <pre><i><u>What does <code>git add .</code> do?
   Also: what's in .gitignore?</u></i></pre>

  </details>

[Next: Exercise 1 - Project Structure](eC-project-structure.md)
