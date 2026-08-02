# Exercise 2: Start tracking files

It's time to add files that we want to track with DVC. We don't want to include the files in our repo but instead track and use references to the data stored elsewhere.

There's multiple [storage solutions in DVC](https://dvc.org/doc/user-guide/data-management/remote-storage). Generally they require some kind of authentication so that you can read and write from the location but that is beyond the scope for now. Instead we will just use a "local" remote -- that is, just a different directory on your laptop than this project and pretend it's a "remote" remote.

1. Fetch data locally:
    ```bash
    mkdir -p data/raw
    wget https://raw.githubusercontent.com/Jeppe-T-K/itu-data-science-in-production/main/w04/resources/coco_small.jpg -P data/raw/
    ```

2. Add file to tracking with DVC
   <details> <summary>Hint: It's essentially the same as git</summary> 
    <pre> dvc add data
    </pre>
    </details>

3. Ensure metadata files are tracked by git (_which files are added?_)
   <details> <summary>Hint: add and commit newly created files</summary> 
    <pre> git add data.dvc .gitignore
    git commit -m "Added data/ to dvc"
    </pre>
    </details>

4. What info does each .dvc file contain?
   <details> <summary>Hint: Open the files in a text editor or on unix based systems (e.g. Linux or Mac) you can use `less` or `cat` in the terminal to view them </summary> 
    less (view individual files): 
    <pre> 
       less *.dvc # use :n, :p, q, to go to next file, previous file and quit
    </pre>
    cat (prints the contents of all files):
      <pre> 
       cat *.dvc
    </pre>
    note: The * here just means anything so in this case any file name with the .dvc extension
    </details>
