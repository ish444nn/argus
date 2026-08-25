# The interface

Why Argus looks the way it does.

## The use scene

An analyst on a large monitor for a whole shift, reading dense tabular evidence
and deciding whether to escalate. Everything below follows from that:

- **Cool dark slate**, not white and not warm — it does not glare after an hour.
- **Near-zero radius, no shadows.** Structure comes from 1px rules and four
  surface steps. Floating rounded cards waste vertical space and imply a
  hierarchy the content does not have.
- **Density over air.** 13px table body, tabular numerals, tight rhythm. An
  analyst comparing forty rows needs them on one screen.
- **Desktop first.** It degrades to one column and never overflows
  horizontally, but it is not pretending to be a phone app.

## The organising idea: provenance

Argus holds three kinds of knowledge, and the whole product depends on never
confusing them:

| | Meaning | Rail | Colour |
|---|---|---|---|
| **measured** | Argus counted it | solid | steel blue |
| **model** | a model inferred or wrote it | dashed | violet |
| **cited** | quoted from a published source | double | green |

Every information block carries its provenance on its left edge, with a mono
eyebrow naming it. The rail differs by **line style** as well as colour, so the
distinction survives greyscale and colour-blindness.

This is the signature element, and it is not decoration. It is the reason a
reader never has to work out whether they are looking at a fact, an inference,
or a quotation — the question the rest of the system spends most of its effort
keeping answerable.

Confidence sits inside the assessment panel but wears the **measured** rail,
because it is computed from evidence rather than stated by the model. That is
the single easiest distinction to lose and the most expensive one to lose.

## Deliberately not Last Call

The previous project is warm paper (`#fbf8f3`), a Fraunces serif display, an
ember accent, soft warm shadows, 8–14px radii, mobile-first. Argus inverts every
one of those axes: cool dark slate, no serif anywhere, functional-only colour,
no shadows, 2px radius, desktop-dense. They should not read as siblings.

It also avoids the two looks that AI-generated design currently defaults to —
cream + serif + terracotta, and near-black + one acid accent. Argus is dark, but
its dominant impression is achromatic; colour appears only where it carries
meaning, and there are three functional hues rather than one bright one.

## Type

**IBM Plex Sans**, **IBM Plex Mono**, **IBM Plex Sans Condensed**. Institutional
rather than fashionable, and Plex has real character in its humanist/grotesque
hybrid forms.

Mono is used aggressively — every identifier, score, count and timestamp — which
is both functional (tabular alignment down a column) and gives the interface a
texture that distinguishes it from a generic sans-serif dashboard. Condensed
uppercase carries every eyebrow and table head, creating hierarchy through
width and letterspacing rather than size, which keeps rows short.

## Colour is never the only signal

- **Risk** reads three ways at once: a four-step ladder whose *filled step
  count* carries the band, the numeral, and colour. The ladder alone is
  sufficient.
- **Selected row** gets an inset left marker as well as a background.
- **Flagged counterparties** in the ego graph get a ring as well as a fill.
- **Status** is always a word, never a bare dot.

## Screens

**Overview** — what has been processed, what is waiting, is the machinery
behaving. Every figure is counted from Postgres; where nothing has been
recorded, the panel says so rather than drawing an empty axis. The risk
distribution deliberately spans *all scored transactions* with the alerted slice
marked, because a distribution of queued cases would put every row in the top
band and show nothing.

**Queue** — the screen an analyst lives on. Filters and sort live in the URL, so
a view can be shared, bookmarked and restored by the back button.

**Case** — its own route, not a docked panel: a case is the unit of work and
deserves the full width and a URL. Reading order runs left to right, **evidence
before interpretation**. What Argus measured and the sources it quoted occupy
the wider column; the assessment and the decision sit to the right and stay in
view while the evidence scrolls. That ordering is an argument: the evidence
comes first and the interpretation is answerable to it.

## Data visualisation

Four visuals, all backed by rows:

- **Risk distribution** — scored population by band, alerted portion marked.
- **Alert budget** — realised rate against the configured budget.
- **Composition bars** — evidence kinds, case lifecycle, decisions.
- **Ego graph** — the transaction's real one-hop neighbourhood, senders left,
  recipients right, so direction is the horizontal axis and needs no arrowheads.
  Capped at 24 with `truncated` surfaced, because a partial picture must say it
  is partial.

Nothing is drawn from invented numbers, and there is no chart present only to
fill a panel.

## States

Loading (skeletons that match the shape of what is coming), empty, error and
stale all share one visual treatment, so they read as the same family. Empty
states say what to do next rather than apologising. A background investigation
shows a pulsing marker and polls only while it is actually running.

## Accessibility floor

Semantic landmarks and headings; a skip link; keyboard-operable table rows with
Enter/Space; `aria-sort` on sortable headers; visible focus rings never removed;
`role="status"` on background progress and `role="alert"` on errors; labelled
form controls; `prefers-reduced-motion` respected; no meaning carried by colour
alone.
