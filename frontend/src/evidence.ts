/**
 * The evidence vocabulary, in one place.
 *
 * Two things live here because they have to agree everywhere: what each
 * evidence kind is *called*, and what colour it is drawn in. A kind that is
 * "Historical similarity" on the case page and blue in the overview chart must
 * be both of those on every other screen, and the way to guarantee that is for
 * there to be exactly one place to look it up.
 *
 * `graph_model_corroboration` is in the map but is not evidence. It is the
 * graph model's own probability for the transaction — the *second opinion*,
 * one of the three signals reported at the top of a case, alongside the risk
 * score and the evidence confidence. It is kept here only so that the one
 * place that names it can name it consistently.
 *
 * The distinction that has to survive: `structural_similarity` also comes from
 * GraphSAGE and *is* evidence. It is a measurement — this transaction sits
 * near these named, previously-confirmed illicit ones — not the model's
 * opinion about this transaction.
 */

export type Origin = "measured" | "model" | "cited";

export type EvidenceMeta = {
  /** What the analyst reads. */
  label: string;
  /** One line explaining what produced it, for a tooltip. */
  hint: string;
  /** Which of the three kinds of knowledge it is. */
  origin: Origin;
  /** Its colour in every chart, legend and meter. */
  colour: string;
  /** False for the second opinion, which is a signal rather than a finding. */
  observed: boolean;
};

export const EVIDENCE: Record<string, EvidenceMeta> = {
  structural_similarity: {
    label: "Historical similarity",
    hint: "Transactions from the training period whose network position the graph model represents the same way.",
    origin: "measured",
    colour: "var(--cat-1)",
    observed: true,
  },
  heuristic: {
    label: "Structural pattern",
    hint: "A network shape computed from the graph — relay, fan-out, fan-in, layering chain, dense cluster.",
    origin: "measured",
    colour: "var(--cat-2)",
    observed: true,
  },
  confirmed_neighbour: {
    label: "Confirmed counterparty",
    hint: "A connected transaction an analyst has confirmed, or one known illicit from an earlier batch.",
    origin: "measured",
    colour: "var(--cat-3)",
    observed: true,
  },
  flagged_neighbour: {
    label: "Flagged counterparty",
    hint: "A connected transaction the model has already flagged into the queue.",
    origin: "measured",
    colour: "var(--cat-4)",
    observed: true,
  },
  typology_reference: {
    label: "Typology source",
    hint: "A passage retrieved from published AML guidance and quoted verbatim.",
    origin: "cited",
    colour: "var(--cat-5)",
    observed: true,
  },
  graph_model_corroboration: {
    label: "Second opinion",
    hint: "The graph model's own probability for this transaction. A signal, not a finding.",
    origin: "model",
    colour: "var(--model)",
    observed: false,
  },
};

export function evidenceMeta(kind: string): EvidenceMeta {
  return (
    EVIDENCE[kind] ?? {
      label: kind.replace(/_/g, " "),
      hint: "",
      origin: "measured",
      colour: "var(--line-2)",
      observed: true,
    }
  );
}

export const TYPOLOGY_LABELS: Record<string, string> = {
  structuring: "Structuring",
  funnelling: "Funnelling",
  layering: "Layering",
  mixing_or_obfuscation: "Mixing or obfuscation",
  mule_network: "Mule network",
  network_association: "Network association",
  no_clear_typology: "No clear typology",
};
