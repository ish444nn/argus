import type { EvidenceItem } from "../api/client";
import { evidenceMeta } from "../evidence";
import { Meter, Note, ProvLabel } from "./ui";

/**
 * Observed evidence — what Argus found out about the transaction.
 *
 * Grouped by kind rather than listed flat, because the groups answer different
 * questions and an analyst reads them in a different order: what does this
 * resemble, what shape is it, who is it connected to.
 *
 * The graph model's own score is not here. It is a signal, reported with the
 * risk score and the confidence at the top of the page; a model's opinion of
 * the transaction is not a finding about it. Typology passages are not here
 * either — they are quoted in full in their own panel.
 *
 * Every row shows the transaction or rule it came from and its contribution to
 * confidence. That traceability is the point of the evidence model, so it sits
 * on the face of each row rather than behind a disclosure.
 */

const GROUPS: { kinds: string[]; title: string }[] = [
  { kinds: ["structural_similarity"], title: "Historical similarity" },
  { kinds: ["heuristic"], title: "Structural patterns" },
  { kinds: ["confirmed_neighbour"], title: "Confirmed counterparties" },
  { kinds: ["flagged_neighbour"], title: "Flagged counterparties" },
];

function Row({ item }: { item: EvidenceItem }) {
  const meta = evidenceMeta(item.kind);
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
        <div
          className="shrink-0 text-right"
          title="Strength × the weight of this kind = its contribution to confidence"
        >
          <Meter value={item.strength} width={56} colour={meta.colour} />
          <p className="num mt-1 text-[11px] text-[var(--text-3)]">
            {item.strength.toFixed(2)} × {item.weight.toFixed(2)}
          </p>
        </div>
      </div>
    </li>
  );
}

export function EvidenceList({ items }: { items: EvidenceItem[] }) {
  const observed = items.filter((item) => evidenceMeta(item.kind).observed);
  const groups = GROUPS.map((group) => ({
    ...group,
    items: observed.filter((item) => group.kinds.includes(item.kind)),
  })).filter((group) => group.items.length);

  if (!groups.length) {
    return (
      <Note title="No supporting evidence">
        Nothing corroborated this transaction beyond its score. The structural
        heuristics need degrees it does not have, and no historical match cleared
        the similarity threshold.
      </Note>
    );
  }

  const shown = groups.reduce((sum, group) => sum + group.items.length, 0);

  return (
    <div className="space-y-5">
      <ProvLabel
        origin="measured"
        detail={`${shown} item${shown === 1 ? "" : "s"} across ${groups.length} kind${groups.length === 1 ? "" : "s"}`}
      />

      {groups.map((group) => {
        const meta = evidenceMeta(group.kinds[0]);
        return (
          <section key={group.title}>
            <h3 className="flex items-center gap-2 text-[var(--text)]" title={meta.hint}>
              <span className="size-2 shrink-0" style={{ background: meta.colour }} aria-hidden />
              {group.title}
            </h3>
            <ul className="mt-2 border-t border-[var(--line)] pt-3">
              {group.items.map((item) => (
                <Row key={item.id} item={item} />
              ))}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
