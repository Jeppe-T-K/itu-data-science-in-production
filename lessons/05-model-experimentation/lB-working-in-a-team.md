---
title: Developing a(ny) software product
---

# Setting the context

![Example Gantt chart from gantt.com](/images/model_experimentation/gantt-chart-1.png)

Whenever you are working on a project, you want to make sure that you deliver the right thing on time.
Unsurprisingly, there are many different tools and methods available to ensure the success of the project and team.

An example of a project management tool is a Gantt chart (as shown above), which many of you may have seen in some form or another.
It can be useful for projects where the completion of one step is a prerequisite for the next steps.


---

## Software team performance

Software projects, however, tend to be less linear, less waterfall'y, and have different risks than for example construction of a house.
And AI/ML projects? A whole other set of challenges with data and models.

So how can we know if the project and team are performing as expected?

While we will talk more about working in a team and agile processes later, one tool for evaluating the performance of the team is _DORA metrics_.

In the following sections we will cover what they mean for any software team before relating it to AI/ML projects.

---

## Example: Apex Manager

![The APEX team's internal A/B testing tool](/images/model_experimentation/apex-team-tool.png)

An example from my work is the **APEX team**: a team maintaining an internal A/B testing tool.

They continuously deliver new requests, such as:

- **Features**: new functionality in the tool, like the Audience Manager, improved statistical methods, or new results UI.
- **Bugs**: fixes to either the experiment setup (involving other teams) or in own product (all software has bugs).
- **Resuming work**: re-opening experiments that were parked


Multiple people contribute changes to the same tool, at different speeds, with different levels of context. That is exactly the situation the DORA metrics are designed to shed light on: how fast does work get to users, and how safely does it land there?
