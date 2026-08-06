---
title: Model Component
---

# Model Component

---

## Black box

![Model black box](/images/mlops-projects-overview/black-box.svg)

---

## Model training

![mlmastery-training-graph.png](/images/mlops-projects-overview/mlmastery-training-graph.png)
From https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

- Don’t get too attached to one model

- Feature + model engineering

- Random search for hyperparameters

→ Robustness is important

---

## Model evaluation/testing

![Ensemble training](/images/mlops-projects-overview/encord-training-note.png)
From https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/

- Holdout test set

- Multiple metrics to measure…
    - Error
    - Bias

- Must be better than baseline

---

## Model validation/packaging

| Format | Open-Format | Vendor | File Extension | License | ML Tools & Platforms Support | Human-readable | Compression |
|---|---|---|---|---|---|---|---|
| "almagnation" | — | — | — | — | — | — | ✅ |
| PMML | ✅ | DMG | .pmml | AGPL | R, Python, Spark | ✅ (XML) | ❌ |
| PFA | ✅ | DMG | JSON | | PFA-enabled runtime | ✅ (JSON) | ❌ |
| ONNX | ✅ | SIG LFAI | .onnx | | TF, CNTK, Core ML, MXNet, ML.NET | ❌ | ✅ |
| TF Serving Format | ✅ | Google | .pf | | Tensor Flow | ❌ | g-zip |
| Pickle Format | ✅ |  | .pkl | | scikit-learn | ❌ | g-zip |
| JAR/POJO | ✅ |  | .jar | | H2O | ❌ | ✅ |
| HDF | ✅ |  | .h5 | | Keras | ❌ | ✅ |
| MLEAP | ✅ |  | .jar/.zip | | Spark, TF, scikit-learn | ❌ | g-zip |
| Torch Script | ❌ |  | .pt | | PyTorch | ❌ | ✅ |
| Apple .mlmodel | ❌ | Apple | .mlmodel | | TensorFlow, scikit-learn, Core ML | — | ✅ |

Adapted from https://ml-ops.org/content/three-levels-of-ml-software#model-machine-learning-pipelines

- Unit/integration testing

- Different models, different save formats

