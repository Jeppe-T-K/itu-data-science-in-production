# Exercise 5: Import from URL

Sometimes you rely on external data sources, for example if a remote system updates the data file or if DVC is not used by that system. In order to do that, we can import directly from a url:

1. Import the data 
   <details> <summary>Hint: Use the import-url command</summary> 
    <pre> 
    mkdir data_imported
    dvc import-url https://raw.githubusercontent.com/Jeppe-T-K/itu_sdse_2024/9b7b05cf2bd3551f7d723d6510b8fdd2a0df9b66/w06/resources/data/cleansed/coco_cropped.png data_imported/coco_edited_highres.jpeg
    git add data_imported/*
    git commit -m "Added cleansed data"
    (git push)
    dvc push
    </pre>
    </details>
2. How can you avoid downloading data directly? And why would you want that?
   <details> <summary>There are two commands</summary>

    1. --no-download

    2. --to-remote

    --no-download simply creates the DVC file.

    --to-remote also transfers the file to your DVC remote.
    </details>

3. How do you make sure the tracked data is updated?
   <details> <summary>You need to <i>update</i> via DVC</summary>
   dvc update --to-remote data_imported/coco_edited_highres.jpeg
    </details>

## Outro
And that's it! Now you've done the whole flow of tracking raw data, creating + modifying a cleansed dataset, and switching between different data versions.

Much of the same thinking about version control of big files also applies to models. However, the flow for creating, testing, evaluating and deploying models through _experiments_ usually require a bit more infrastructure, so before going crazy with exercises for that, we'll instead focus next time on creating a project that follows a nice, clean structure to make our lives easier.
