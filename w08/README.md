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

There are 4 python files in this directory for these exercises:
* [data_util.py](data_util.py), which you can use for generating the data (imagine it is the output of the data processing pipelines)
* [example_model.py](example_model.py), which you can use as the basic SKLearn model
* [basic_mlflow_training.py](basic_mlflow_training.py), which you can use to guide you on how to create a custom model/function and set up MLflow
* [basic_mlflow_evaluation.py](basic_mlflow_evaluation.py), which you can use for evaluating the (created) models using new data and check for drifts.

As always for the most effective learning, try to use the MLflow documentation and figure out how to solve the exercises yourself first before consulting the basic MLflow scripts.

### Exercise 0: Installation

Make sure MLflow and scipy is installed:
`pip install mlflow`
`pip install scipy`
`pip install scikit-learn`


### Exercise 1: Run an ML experiment

For this exercise, the goal is to use MLflow to create an experiment and log relevant artifacts and metrics for different ML models. Much of what you need for this is under the https://mlflow.org/docs/latest/ml/tracking/quickstart/ page.

MLflow has a lot of in-built "flavours" that make it easy to run experiments for well-known ML libraries such as SKlearn or Tensorflow. For this exercise we can use the SKlearn model in 


1. <details> <summary> Edit the example_model.py script to log model run with MLflow</summary>
   First <code>import mlflow</code>
   
   Then set the MLflow experiment via <code>mlflow.set_experiment("My experiment name")</code>
   
   Lastly start an MLflow run and log relevant things:
   ```python
   with mlflow.start_run():
       mlflow.log_params(params)
       mlflow.log_metric("mean_squared_error", mse)
       mlflow.sklearn.log_model(
         lr, registered_model_name="lasso_regression_model"
       )
   ```
   </details>
   
2. <details> <summary> Run training with different parameters</summary> 
   For example set alpha in params to 0.5.
   </details>

3. <details> <summary> Compare the two models </summary>
   This can be done via the printed output or via the mlflow ui.

   In the terminal run <code>mlflow ui</code>
   </details>

4. <details><summary>What exactly is logged for each run?</summary>
   Check the mlruns directory or through the UI under "artifacts"
  </details>

5. <details><summary>Try to use autologging</summary>
   Add <code>mlflow.autolog()</code> and remove other MLflow logging
  </details>

### Exercise 2: Running a custom model

Sometimes you can't use a standard SKLearn or similarly supported model, so you have to define your own. Since we actually know the data generating function in data_util.py, we could simply create a model that predicts f(x) = x.


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

1. <details> <summary> Register your model </summary>
   This can be done when logging the model and adding the `registered_model_name` argument to the function 
</details>

2. <details> <summary> Run the data evaluation with the data drift data. How do you expect the model error to change? Do you expect to see a difference in the distribution of the X variable?</summary>
   This is available through the get_data_drift_data function.

   Given our simple linear model, we don't expect the error to change. However, in realistic scenarios, our model might not extrapolate and/or interpolate properly for this new distribution, so we still need to keep an eye on model performance.

   We do expect the distribution of the X variable to change. That is essentially the concept of data drift.
</details>

3. <details> <summary> Run the model evaluation with the concept drift data. How do you expect the model error to change? Do you expect to see a difference in the distribution of the X variable? </summary>
   This is available through the get_data_concept_data function.

   Now we expect the error of the model to increase since the fundamental proportional relationship of the data has changed (constant of 10 is added). This will be visible in the MLflow UI.

   The distribution of our X variable is the same, so we don't expect any data drift/distribution change. 

</details>


And that is it! You can do a lot more with MLflow, and you might see some more complex commands later in the course, but remember: start simple and then add complexity later.
