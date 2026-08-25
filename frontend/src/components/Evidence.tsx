import type { EvidenceItem } from "../api/client";

/**
 * Evidence display.
 *
 * Every item shows what it rests on, not just what it claims: a similarity or
 * neighbour item names the transaction it came from, and each shows its
 * contribution to the deterministic confidence score. That traceability is the
 * point of the whole evidence model, so it is on the face of the card rather
 * than hidden behind a details toggle.
 */

const KIND_LABELS: Record<string, string> = {
  heuristic: "Structural pattern",
  structural_similarity: "Structural similarity",
  graph_model_corroboration: "Graph model second opinion",
  flagged_neighbour: "Flagged neighbour",
  confirmed_neighbour: "Confirmed neighbour",
  typology_reference: "Typology reference",
};

const KIND_STYLES: Record<string, string> = {
  heuristic: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  structural_similarity: "bg-violet-500/15 text-violet-300 ring-violet-500/30",
  graph_model_corroboration: "bg-sky-500/15 text-sky-300 ring-sky-500/30",
  flagged_neighbour: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
  confirmed_neighbour: "bg-rose-500/20 text-rose-200 ring-rose-500/40",
  typology_reference: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
};

function StrengthBar({ value }: { value: number }) {
  return (
    <div className="h-1 w-16 overflow-hidden rounded-full bg-white/10">
      <div
        className="h-full rounded-full bg-zinc-300"
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  );
}

export function EvidenceCard({ item }: { item: EvidenceItem }) {
  const heuristic =
    item.details && typeof item.details.heuristic === "string"
      ? (item.details.heuristic as string)
      : null;

  return (
    <li className="rounded-lg border border-white/10 bg-zinc-900/50 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={`rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
            KIND_STYLES[item.kind] ?? "bg-white/10 text-zinc-300 ring-white/20"
          }`}
        >
          {KIND_LABELS[item.kind] ?? item.kind}
        </span>
        {heuristic && (
          <span className="font-mono text-xs text-zinc-500">{heuristic}</span>
        )}
        <span className="ml-auto flex items-center gap-2 font-mono text-xs text-zinc-500">
          <StrengthBar value={item.strength} />
          {item.strength.toFixed(2)} &times; {item.weight.toFixed(2)} ={" "}
          <span className="text-zinc-300">{item.contribution.toFixed(3)}</span>
        </span>
      </div>

      <p className="mt-2 text-sm text-zinc-200">{item.summary}</p>

      {item.neighbour_tx_id !== null && (
        <p className="mt-2 font-mono text-xs text-zinc-500">
          source: transaction {item.neighbour_tx_id}
          {item.neighbour_timestep !== null && ` · time step ${item.neighbour_timestep}`}
        </p>
      )}
    </li>
  );
}

export function EvidenceList({ items }: { items: EvidenceItem[] }) {
  if (!items.length) {
    return (
      <p className="rounded-lg border border-dashed border-white/10 p-4 text-sm text-zinc-500">
        No supporting evidence was found for this transaction. The structural
        heuristics need degrees this transaction does not have, and no
        historical match cleared the similarity threshold.
      </p>
    );
  }

  const total = items.reduce((sum, item) => sum + item.contribution, 0);

  return (
    <>
      <p className="mb-3 text-xs text-zinc-500">
        {items.length} item{items.length === 1 ? "" : "s"}, total contribution{" "}
        <span className="font-mono text-zinc-300">{total.toFixed(3)}</span>. Confidence
        is computed from these deterministically in Phase 4 — never self-reported by a
        language model.
      </p>
      <ul className="space-y-2">
        {items.map((item) => (
          <EvidenceCard key={item.id} item={item} />
        ))}
      </ul>
    </>
  );
}
