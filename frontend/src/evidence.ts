/**
 * The evidence vocabulary, in one place.
 *
 * The database stores six `kind` values; the interface shows six labels. They
 * are the same six things — the mapping exists so a reader never meets a raw
 * enum, not to introduce a seventh category.
 *
 * The one that needs care is `graph_model_corroboration`. It is shown as
 * **Second opinion**, because that is what it is: GraphSAGE's own probability,
 * quoted beside the case. It is deliberately distinct from **Historical
 * similarity**, which also uses GraphSAGE but is a *measurement* — this
 * transaction sits near these named, previously-confirmed illicit ones —
 * rather than the model's opinion about this transaction.
 *
 * That difference is why one counts towards evidence confidence and the other
 * does not, so the labels have to make it legible.
 */

export type Origin = "measured" | "model" | "cited";

export type EvidenceMeta = {
  /** What the analyst reads. */
  label: string;
  /** One line explaining what produced it. */
  hint: string;
  /** Which of the three kinds of knowledge it is. */
  origin: Origin;
  /** Whether it contributes to evidence confidence, and why not if it doesn't. */
  countsTowardConfidence: boolean;
  excludedBecause?: string;
};

export const EVIDENCE: Record<string, EvidenceMeta> = {
  structural_similarity: {
    label: "Historical similarity",
    hint: "Transactions from the training period whose network position the graph model represents the same way.",
    origin: "measured",
    countsTowardConfidence: true,
  },
  heuristic: {
    label: "Structural pattern",
    hint: "A network shape computed from the graph. Thresholds come from the whole dataset's degree distribution.",
    origin: "measured",
    countsTowardConfidence: true,
  },
  confirmed_neighbour: {
    label: "Confirmed counterparty",
    hint: "A connected transaction an analyst has confirmed, or one known illicit from an earlier batch.",
    origin: "measured",
    countsTowardConfidence: true,
  },
  flagged_neighbour: {
    label: "Flagged counterparty",
    hint: "A connected transaction the model has already flagged into the queue.",
    origin: "measured",
    countsTowardConfidence: true,
  },
  graph_model_corroboration: {
    label: "Second opinion",
    hint: "The graph model's own probability for this transaction, quoted for comparison.",
    origin: "model",
    countsTowardConfidence: false,
    excludedBecause:
      "This is a model's score, not evidence. Counting it would let a case look well-supported on nothing but a second model agreeing.",
  },
  typology_reference: {
    label: "Typology source",
    hint: "A passage retrieved from published AML guidance and quoted verbatim.",
    origin: "cited",
    countsTowardConfidence: false,
    excludedBecause:
      "A retrieved passage explains why a signal matters; it is not itself a signal.",
  },
};

export function evidenceMeta(kind: string): EvidenceMeta {
  return (
    EVIDENCE[kind] ?? {
      label: kind.replace(/_/g, " "),
      hint: "",
      origin: "measured",
      countsTowardConfidence: true,
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

/**
 * A colour per typology.
 *
 * Previously every typology drew from the model palette, so a chart of two
 * different assessments rendered as one undifferentiated violet bar and told
 * the reader nothing. These are distinguishable hues within the existing
 * restrained palette; `no_clear_typology` is deliberately neutral, because
 * "we could not tell" should not look like a finding.
 */
export const TYPOLOGY_COLOURS: Record<string, string> = {
  structuring: "var(--risk-3)",
  funnelling: "var(--cited)",
  layering: "var(--model)",
  mixing_or_obfuscation: "var(--measured)",
  mule_network: "var(--risk-4)",
  network_association: "#7f9fd4",
  no_clear_typology: "var(--line-2)",
};
