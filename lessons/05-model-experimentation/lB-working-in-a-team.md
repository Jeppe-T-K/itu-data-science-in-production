---
title: Working in a Team
---

# Working in a Team – Tech Wise

Before we dive into metrics, a quick look at what it actually means to work on a data science product together with other people.

As a running example, consider **the APEX team**: a team maintaining an internal A/B testing tool.

They continuously deliver new requests, such as:

- **Features**: new functionality in the tool
- **Metrics**: new measurements to report
- **Bugs in setup**: fixes to the experiment setup
- **Resume when?**: re-opening old work that was parked

![The APEX team's internal A/B testing tool](/images/model_experimentation/apex-team-tool.png)

Multiple people contribute changes to the same tool, at different speeds, with different levels of context. That is exactly the situation the DORA metrics are designed to shed light on: how fast does work get to users, and how safely does it land there?

> **Note — Tooling**
> This is also why we care about tooling in this course: shared tools, shared conventions and shared measurements are what make teamwork in a technical setting scale.
