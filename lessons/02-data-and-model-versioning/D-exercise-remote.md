# Exercise 3: Using a remote

Remotes work in a similar way as with git: it's some place else that keeps track of you data. DVC stores the data in an efficient binary format and allows you to fetch the data again easily and quickly when needed for e.g. ML stuff.

> **Note**: we will simply download the data locally and work with it there, but if you have a remote set up already with data, you can follow [this method](https://dvc.org/doc/user-guide/data-management/importing-external-data#how-importing-external-data-works) for using "external data" (not necessary for now).

1. Configure remote ("local") with `dvc remote add -d <remote_name> /localpath/to/remote`
   <details> <summary>Want to use actual remote?</summary> 
    You're welcome to try and set it up. Take your pick from <a href="https://dvc.org/doc/user-guide/data-management/remote-storage"> all these options </a>
    </details>

2. Add and commit changes to .dvc/config
   <details> <summary>Hint: you know the flow now with git</summary> 
    <pre> git add .dvc/config
    git commit -m "Added remote to dvc" </pre>
    </details>

3. Push DVC data to remote
   <details> <summary>Hint: It's essentially the same as git</summary> 
    <pre> dvc push </pre>
    </details>

4. You accidentally (or intentionally) deleted your local data in /data/raw. How would you fetch it again?
   <details> <summary>Hint: It's essentially the same as git</summary> 
    <pre> dvc pull </pre>
    </details>

[Next: Exercise 4 - Switching Between Versions](E-exercise-versions.md)
