---
id: model-risk-scoring
title: Interpreting model risk scores
publisher: Wolfsberg Group
source_url: https://www.wolfsberg-principles.com/wolfsberg-group-statements
document: "Wolfsberg Statement on Effective Monitoring for Suspicious Activity"
year: 2009
patterns: [model_risk_scoring, behavioural_similarity]
---

## A score is an ordering, not a verdict

Monitoring guidance is consistent that automated systems rank activity for
human attention rather than determine its character. A high score means the
activity resembles the population the system was fitted to treat as risky. It
does not establish that any particular transfer is illicit, and it carries no
information the underlying features did not.

Alert thresholds follow from review capacity rather than from any natural
boundary in the data. Where a fixed number of cases can be examined, the
question a monitoring system answers is which cases those should be.

## Agreement between independent models

Where two models with different inputs score the same activity as risky, the
agreement is worth noting, but its value depends on how independent they really
are. Models trained on the same data and the same labels frequently agree for
the same reasons, and their concurrence adds less than it appears to.

Agreement is most informative when the models see genuinely different things:
one reading the attributes of an entity, another reading its position among
counterparties. Even then, corroboration between models is a reason to
prioritise a review, not a substitute for evidence about the specific activity.
