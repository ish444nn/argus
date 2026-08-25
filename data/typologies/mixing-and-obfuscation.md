---
id: mixing-and-obfuscation
title: Mixing, chain-hopping and deliberate obfuscation
publisher: FATF
source_url: https://www.fatf-gafi.org/en/publications/Fatfrecommendations/Guidance-rba-virtual-assets-2021.html
document: "Updated Guidance for a Risk-Based Approach to Virtual Assets and VASPs"
year: 2021
patterns: [layering, virtual_assets, network_association]
---

## Obfuscation techniques

Where transfers are publicly visible, laundering shifts from concealing that a
transfer happened to concealing which transfers belong together. FATF guidance
describes several techniques with this purpose: mixing and tumbling services
that pool value from many participants and redistribute it so inputs cannot be
matched to outputs, chain-hopping between different assets or networks to break
a single traceable history, and the use of large numbers of intermediate
addresses used once and abandoned.

## Why structure survives obfuscation

These techniques defeat naive tracing, but they leave structural traces of
their own. Pooling produces characteristic convergence and divergence around a
small number of points. Single-use intermediate addresses produce long runs of
minimal-degree nodes. Automated distribution produces regularity that manual
activity does not.

The practical consequence for analysis is that obfuscation tends to make
activity structurally distinctive rather than structurally invisible. A
transaction whose position in the network resembles previously confirmed
illicit activity is informative precisely because the resemblance is a
by-product of the method used, which is harder to vary than any individual
transfer.
