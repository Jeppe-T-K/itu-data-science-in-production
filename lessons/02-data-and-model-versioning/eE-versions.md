---
title: Versions
---

# Exercise 4: Switching between data versions

Quite often you don't use the raw data directly in your models but run it through a data pipeline to create a cleansed or ML-ready dataset. That also means that the files/artifacts you create and want to keep track of can change with every run of your code.

For this exercise, we will essentially follow the steps in Exercise 1, but in addition we will switch to another branch in git and make changes to our cleansed data. The challenge is then to recover the original data on the main branch!

1. Create a "cleansed" version of your data (or use this pre-made)
    ```sh
    mkdir -p data/cleansed
    wget https://raw.githubusercontent.com/Jeppe-T-K/itu-data-science-in-production/main/w04/resources/coco_edited_small.jpg -P data/cleansed
    ```

2. Add new data to dvc + git
   <details> <summary>Hint: repeat steps 2-3 from Exercise 1</summary> 
    <pre> 
    dvc add data/
    git add data.dvc
    git commit -m "Added cleansed data"
    (git push)
    dvc push
    </pre>
    </details>

3. Create new branch in git
   <details> <summary>Hint</summary> 
    <pre> git checkout -b "my_branch_name"
    </pre>
    </details>

4. Modify your cleansed data (_save to same file name!_)


5. Add changed file to dvc + git
   <details> <summary>Hint: repeat steps from step 2</summary> 
    <pre> 
    dvc add data/
    git add data.dvc
    git commit -m "Added cleansed data from new method"
    (git push)
    dvc push
    </pre>
    </details>

6. To back to the main branch and check out your cleansed data (_is it the same that you originally added to dvc?_)
   <details> <summary>Hint</summary> 
    No, it's not. Not unless you made a mistake, anyway.

    To go back to the main branch, you can run `git checkout main`

    <pre> 
    dvc pull
    </pre>
    </details>
