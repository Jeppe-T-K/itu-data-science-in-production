---
title: "Change Failure Rate: Managing Failures"
---

# Change Failure Rate: Managing Failures

---

## Failures happen

- Any change carries a risk
- Failure = subpar performance:
  - F1-score, precision, etc
  - User experience
- Every release? Every 3rd? … how often do you accept to fail?

![Failures happen](/images/model_experimentation/failures-happen.png)

As discussed before – nothing is perfect, and models in particular can degrade quietly.

---

## Managing risk

You don't have to roll a change out to everyone at once. Only rollout partly, with different strategies:

- **Pilot studies** – ship to a small, selected group first
- **A/B tests** – compare the new model against the old on random user groups
- **Shadow testing** – run the new model in production without serving its predictions, and compare offline

![A/B testing rollout](/images/model_experimentation/ab-testing-rollout.png)

From https://mlinproduction.com/ab-test-ml-models-deployment-series-08/

As discussed before – gradual rollouts turn a potentially big failure into a small, contained one, and they give you the data to decide whether to continue.
