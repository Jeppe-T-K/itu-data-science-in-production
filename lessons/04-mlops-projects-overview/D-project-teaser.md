# Exercise 2: Project Teaser

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

[Next: Exercise 3 - Collaboration](E-collaboration.md)
