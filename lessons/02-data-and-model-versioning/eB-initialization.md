---
title: Initialization
---

# Exercise 1: Initialise DVC in repository

Next we want to start using DVC with one of our projects. Let's follow DVC's documentation on [getting started](https://dvc.org/doc/start):

1. Run `dvc init` 
   <details> <summary>Getting an <i>ERROR: failed to initiate DVC - path is not tracked by any supported SCM tool (e.g. Git)</i> error? </summary> 

   Make sure you have initialised git in your directory. Run the following to start a git repo <pre>git init</pre>

   If this does not work, try and run it with the --subdir argument if you are initialising this in a subdirectory of your project. <pre> dvc init --subdir</pre>
</details>

2. Check which files DVC added to the repo (_what is each file used for?_)
   <details> <summary>Hint </summary>
   <tt> git status </tt>
   
   1. .dvc/.gitignore (dvc-specific things for git to ignore, like local config.local)
   2. .dvc/config (project-level dvc config, keeps tracks of various settings like remotes, local path to auth, etc)
   3. .dvcignore (file types for dvc to ignore if e.g. adding directories)
 </details>

3. Git commit new files
   <details> <summary>Hint </summary> 
   <tt> git commit -m "YOUR COMMIT MSG"</tt>
 </details>
