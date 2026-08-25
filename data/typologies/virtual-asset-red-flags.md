---
id: virtual-asset-red-flags
title: Red flag indicators for virtual asset transfers
publisher: FATF
source_url: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Virtual-assets-red-flag-indicators.html
document: "Virtual Assets Red Flag Indicators of Money Laundering and Terrorist Financing"
year: 2020
patterns: [virtual_assets, layering, structuring, behavioural_similarity]
---

## Transaction-pattern indicators

FATF groups virtual-asset red flags by the aspect of activity they concern.
The transaction-pattern group is the one visible in on-chain data without any
customer information, and it is the group that applies to a pseudonymous
transaction graph.

Indicators in this group include transfers structured into amounts just below
reporting thresholds, a new address immediately transacting the full balance it
received, value moved through several addresses in rapid succession with no
intervening economic activity, and funds deposited to or withdrawn from an
address linked to previously reported illicit activity.

## Reading indicators together

FATF is explicit that a single indicator rarely justifies a conclusion. The
indicators are designed to be read in combination, and their weight depends on
context that a transaction graph alone does not supply.

This matters when the underlying data is anonymised. Where amounts, timestamps
and counterparty identities are unavailable, only the structural indicators
survive: how value fans out or consolidates, how many hops it takes, and what
company an address keeps. Conclusions drawn from those alone should be
expressed as patterns consistent with a typology, not as findings about the
purpose of the transfer.
