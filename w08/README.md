# Week 8: Model experimentation, selection and monitoring


## Overview

### Agenda

 * [08:15 – 10:00] – Exercises: MLflow + warnings
 * [10:15 – 12:00] – Lecture: DORA metrics and model experimentation, selection and monitoring

### Preparation

For the exercises:

* https://MLflow.org/docs/latest/ml/ (use as reference material, don't do exercises)

For the lecture:

* https://www.mlebook.com/wiki/doku.php (Chapter 9.3 + 9.4 + 9.5)
* https://ml-ops.org/content/mlops-principles (Monitoring in particular)
* https://www.datadoghq.com/knowledge-center/dora-metrics/ (or other DORA metric google searches is fine)
* https://neptune.ai/blog/how-to-monitor-your-models-in-production-guide (skimming it is fine)
* https://dvc.org/doc/use-cases/experiment-tracking (skimming it is fine)


### Notes

* Have fun!

## Exercises

> [!NOTE]
> **Learning goals**
> <i>By the end of the exercises, we expect you to be able to do the following:</i>
> <ul>
> <li>Organise ML experiments using common tools</li>
> <li>Motivate how this can be used to deploy models</li>
> <li>Explain how to detect concept and model drift</li>
> </ul>

This text needs to be replaced with an introduction to the exercises

### Exercise 0: Installation

Make sure MLflow and scipy is installed:
`pip install MLflow`
`pip install scipy`
`pip install scikit-learn`


### Exercise 1: Run an ML experiment

This text needs to be replace with an explanation of what a single MLflow run really is and how different models can be logged. Basically follow https://mlflow.org/docs/latest/ml/tracking/quickstart/


1. <details> <summary> Log example model with MLflow</summary>
   First <code>import mlflow</code>
   
   Then set the MLflow experiment via <code>mlflow.set_experiment("My experiment name")</code>
   
   Lastly start an MLflow run and log relevant things:
   ```python
   with mlflow.start_run():
       mlflow.log_params(params)
       mlflow.log_metric("mse", mse)
       mlflow.sklearn.log_model(
         lr, registered_model_name="Lasso_Regression_Model"
        )
   ```

   <pre><i><u>Discuss in pairs what each option does</u></i></pre>
   </details>
   
2. <details> <summary> Run training with different parameters</summary> 
   In the terminal, run <code>git init</code>
   </details>

3. <details> <summary> Compare the two models </summary>
   This can be done via the printed output or via the mlflow ui.

   In the terminal
   </details>

4. <details><summary>What are the different logged artifacts?</summary>
   <tt> git add .</tt>

   <tt> git commit -m "Initial DDCS commit"</tt>

   <pre><i><u>What does <code>git add .</code> do?
   Also: what's in .gitignore?</u></i></pre>

  </details>

### Exercise 2: Interlude + project teaser

Let's take a step back and reflect on what we have done so far, and more importantly, _why_ and _how_ it ties into what you've learned so far.

<figure>
  <img src="https://cloud.google.com/static/architecture/images/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning-2-manual-ml.svg" alt="my alt text"/>
  <figcaption><i>Google's ML Maturity level 0, from <a href="https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning">https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning</a></i></figcaption>
</figure>

ML projects quite often follow a very manual process. For example, you might have a Python notebook that does some processing of your data and outputs it _somewhere_. You then run another notebook or script that takes this data, does _some_ training (maybe also the evaluation all at once) before sharing the results _somehow_.

This is not sustainable for a productionised ML project multiple reasons.

1. People will have _no_ idea how to re-use/contribute to any of your code.
2. If you go on vacation and nobody is there to re-run the model, tough luck to the company (_and no, while it seems tempting to make yourself indispensible to the company this way, it's not worth it_).
3. Notebooks are terrible for any production code!
   1. You _always_ end up with code with a non-linear flow where you have to run cell 1&rarr;3&rarr;2&rarr;5&rarr;12&rarr;etc before it runs "correctly".
   2. Version control of notebooks is... weird.
   3. They tend to do everything at once.
   4. [Kaggle competitions](https://www.kaggle.com/code) != production-ready code 

There are more bad coding patterns than it's possible to list, so let's instead turn to some guiding principles that you can follow. Based on [this list of opinions from CCDS](https://cookiecutter-data-science.drivendata.org/opinions), these relate to:

* Data versioning strategy (essentially Week 4's lecture)
* Notebooks for exploration, source code for repetition
* Keep your modeling organized
* Build from the environment up
* Keep secrets and configuration out of version control
* Adapt to your use-case

Each of these are explained in more detail in the link, so you are encouraged to read it through yourself.

#### Project teaser
Given this information, let's see how an MLOps project that is very close to the actual implementation could look like: [MLOps monolith notebook](https://github.com/lasselundstenjensen/itu-sdse-project/blob/main/notebooks/main.ipynb)


### Exercise 3: Collaboration

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
