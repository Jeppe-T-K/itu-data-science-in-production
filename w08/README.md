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

   In the terminal run <code>mlflow ui</code>
   </details>

4. <details><summary>What exactly is logged for each run?</summary>
   Check the mlruns directory or through the UI
  </details>

5. <details><summary>Try to use autologging</summary>
   Add <code>mlflow.autolog()</code> and remove other MLflow logging
  </details>

### Exercise 2: Running a custom model

Sometimes you can't use a standard SKLearn or similarly supported model, so you have to define your own.

1. <details><summary>Define a pyfunc.PythonModel</summary>
    Example:

    ```python
   import pandas as pd
   from typing import List, Dict
   from mlflow.pyfunc import PythonModel
   from mlflow.models import set_model


   class BasicModel(PythonModel):
      def linear(self, numbers):
         return [x for x in numbers]

      def predict(self, context, model_input) -> List[float]:
         if isinstance(model_input, pd.DataFrame):
               model_input = list(model_input.iloc[0].values())
         return self.linear(model_input)


   # This tells MLflow which object to use for inference
   set_model(BasicModel())
    ```
  </details>

2. <details><summary>Do a run with this model</summary>
    Skip the training step and replace the lr model with the basic model.
    
    Log the model with 
    ```python
    mlflow.pyfunc.log_model(
        python_model=lr,
        name="basic_linear_model",
    )
    ```
  </details>



### Exercise 3: Check performance of deployed model

Once a model is deployed, you need to monitor the performance and such.

Here we test how we can log two things: data drift, which is when your input variable drifts, and concept drift, which is when the fundamental relationship between your dependent and independent variable changes.

1. <details> <summary> Register your model </summary></details>
2. <details> <summary> Run the data evaluation with the data drift evaluation </summary></details>
3. <details> <summary> Run the model evaluation with the concept drift data </summary></details>

This was a very preliminary example of working together on code. There are more aspects to it, such as branching strategies, code reviews and pull requests, but that will be covered in later lectures. For now, pat yourselves on the back for actually starting a data science project with a more clear strategy than 80% of companies!
