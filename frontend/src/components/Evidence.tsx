import type { EvidenceItem } from "../api/client";
import { Meter, Note, ProvLabel } from "./ui";

/**
 * Observed evidence.
 *
 * Grouped by kind rather than listed flat, because the groups answer different
 * questions and an analyst reads them in a different order: what does this
 * resemble, what shape is it, who is it connected to, what does the other
 * model think.
 *
 * Every item shows the transaction or rule it came from and its contribution
 * to confidence. That traceability is the point of the evidence model, so it
 * sits on the face of each row rather than behind a disclosure.
 */

const GROUPS: { kinds: string[]; title: string; blurb: string }[] = [
  {
    kinds: ["structural_similarity"],
    title: "Historical similarity",
    blurb:
      "Transactions from the training period whose network position the graph model represents the same way.",
  },
  {
    kinds: ["heuristic"],
    title: "Structural patterns",
    blurb:
      "Network shapes computed from the graph. Thresholds come from the whole dataset's degree distribution.",
  },
  {
    kinds: ["flagged_neighbour", "confirmed_neighbour"],
    title: "Counterparties",
    blurb:
      "Connected transactions the system already distrusts, and why it distrusts them.",
  },
  {
    kinds: ["graph_model_corroboration"],
    title: "Second opinion",
    blurb: "The neighbourhood-aware model's independent score. It does not decide the queue.",
  },
];

function Row({ item }: { item: EvidenceItem }) {
  const heuristic =
    item.details && typeof item.details.heuristic === "string"
      ? (item.details.heuristic as string)
      : null;
  const baseRate =
    item.details && typeof item.details.base_rate === "number"
      ? (item.details.base_rate as number)
      : null;

  return (
    <li className="border-t border-[var(--line)] py-3 first:border-t-0 first:pt-0">
      <div className="flex items-start gap-4">
        <div className="min-w-0 flex-1">
          <p className="text-[var(--text)]">{item.summary}</p>
          <p className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-[var(--text-3)]">
            {item.neighbour_tx_id !== null && (
              <span className="num">
                from tx {item.neighbour_tx_id}
                {item.neighbour_timestep !== null && ` · batch ${item.neighbour_timestep}`}
              </span>
            )}
            {heuristic && <span className="num">{heuristic}</span>}
            {baseRate !== null && (
              <span>fires on {(baseRate * 100).toFixed(1)}% of all transactions</span>
            )}
          </p>
        </div>
        <div className="shrink-0 text-right">
          <Meter value={item.strength} width={56} />
          <p className="num mt-1 text-[11px] text-[var(--text-3)]">
            {item.strength.toFixed(2)} × {item.weight.toFixed(2)}
          </p>
        </div>
      </div>
    </li>
  );
}

export function EvidenceList({ items }: { items: EvidenceItem[] }) {
  const observed = items.filter((item) => item.kind !== "typology_reference");

  if (!observed.length) {
    return (
      <Note title="No supporting evidence">
        Nothing corroborated this transaction beyond its score. The structural
        heuristics need degrees it does not have, and no historical match cleared
        the similarity threshold.
      </Note>
    );
  }

  const groups = GROUPS.map((group) => ({
    ...group,
    items: observed.filter((item) => group.kinds.includes(item.kind)),
  })).filter((group) => group.items.length);

  const kinds = new Set(observed.map((item) => item.kind)).size;

  return (
    <div className="space-y-5">
      <ProvLabel
        origin="measured"
        detail={`${observed.length} item${observed.length === 1 ? "" : "s"} across ${kinds} kind${kinds === 1 ? "" : "s"}`}
      />

      {groups.map((group) => (
        <section key={group.title}>
          <h3 className="text-[var(--text)]">{group.title}</h3>
          <p className="mb-2 mt-0.5 max-w-[74ch] text-[11px] text-[var(--text-3)]">
            {group.blurb}
          </p>
          <ul className="border-t border-[var(--line)] pt-3">
            {group.items.map((item) => (
              <Row key={item.id} item={item} />
            ))}
          </ul>
        </section>
      ))}

      <p className="border-t border-[var(--line)] pt-3 text-[11px] text-[var(--text-3)]">
        Confidence combines these deterministically. Items of one kind combine with
        diminishing returns and can never exceed that kind&rsquo;s weight, so
        corroboration across kinds counts for more than repetition within one.
      </p>
    </div>
  );
}
