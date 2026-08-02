---
title: Custom Models
---

# Exercise 2: Running a custom model

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
        name="Basic Linear Model",
    )
    ```
  </details>

[Next: Exercise 3 - Check Performance of Deployed Model](D-model-monitoring.md)
