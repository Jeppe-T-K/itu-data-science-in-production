---
title: Model Monitoring
---

# Exercise 3: Check performance of deployed model

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

[Back to Introduction](A-introduction.md)
