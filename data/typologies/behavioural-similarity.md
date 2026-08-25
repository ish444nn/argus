---
id: behavioural-similarity
title: Behavioural similarity to known cases
publisher: FATF
source_url: https://www.fatf-gafi.org/en/publications/Digitaltransformation/Opportunities-challenges-new-technologies-for-aml-cft.html
document: "Opportunities and Challenges of New Technologies for AML/CFT"
year: 2021
patterns: [behavioural_similarity, model_risk_scoring, layering]
---

## Comparison against established cases

The FATF review of new analytical technologies describes a shift from
rule-based detection towards methods that compare activity against previously
established cases. Where a rule asks whether behaviour crosses a threshold, a
similarity method asks whether behaviour resembles activity already determined
to be illicit, using representations learned from confirmed examples.

The appeal is coverage of patterns nobody wrote a rule for. Laundering methods
change faster than rule sets, and techniques that generalise from confirmed
cases can surface variants that no explicit threshold anticipates.

## Conditions for the comparison to mean anything

Two conditions determine whether a similarity finding is worth acting on.

The comparison set must consist of cases whose outcome is genuinely known, and
known at the time the comparison is made. Drawing on cases that were themselves
never resolved, or that postdate the activity under review, produces a
confident figure with nothing behind it.

The resemblance must also be expressed in terms an analyst can interrogate. A
similarity score alone is not reviewable. Naming the specific prior cases
matched, and how closely, lets a reviewer inspect those cases and judge whether
the resemblance is meaningful or an artefact of the representation.
