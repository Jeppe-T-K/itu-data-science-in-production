---
title: "Lead Time for Changes: In Production"
---

# Lead Time for Changes: Putting It in Production

---

## Databricks ML experiments

At a larger scale, the same experiment-tracking concepts look like this – e.g. in Databricks:

![Databricks ML experiments](/images/model_experimentation/databricks-ml-experiments.png)

Each experiment tracks:

- **Artifacts**
  - Model file
  - Training results
  - Configurations
- Managed MLflow implementation under the hood

---

## Putting it in production

You have run many experiments – now what?

- **Multiple models** to choose between
- **Staging → production**: promote a model through stages before serving it to users
- **Go with the best metric one** – that is the obvious heuristic

But beware: **recent != better**.

The newest model was probably trained on the newest data, but:

- It might not be better on the metrics that matter
- The old model was not tested on the new data

Although, in this case it was…

![Model versions](/images/model_experimentation/model-versions.png)
