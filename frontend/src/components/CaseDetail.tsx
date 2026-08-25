import { useQuery } from "@tanstack/react-query";
import { getCase } from "../api/client";
import { EvidenceList } from "./Evidence";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-zinc-900/50 px-3 py-2">
      <dt className="text-xs text-zinc-500">{label}</dt>
      <dd className="mt-0.5 font-mono text-sm text-zinc-100">{value}</dd>
    </div>
  );
}

export function CaseDetail({ caseId, onClose }: { caseId: number; onClose: () => void }) {
  const { data, isPending, error } = useQuery({
    queryKey: ["case", caseId],
    queryFn: () => getCase(caseId),
  });

  return (
    <aside className="flex h-full flex-col border-l border-white/10 bg-zinc-950">
      <header className="flex items-center justify-between border-b border-white/10 px-5 py-3">
        <h2 className="text-sm font-semibold">
          Case {caseId}
          {data && (
            <span className="ml-2 font-mono text-xs font-normal text-zinc-500">
              tx {data.tx_id}
            </span>
          )}
        </h2>
        <button
          onClick={onClose}
          className="rounded px-2 py-1 text-xs text-zinc-400 hover:bg-white/5 hover:text-zinc-100"
        >
          Close
        </button>
      </header>

      <div className="flex-1 overflow-y-auto px-5 py-4">
        {isPending && <p className="text-sm text-zinc-400">Loading case…</p>}
        {error && <p className="text-sm text-rose-400">{(error as Error).message}</p>}

        {data && (
          <>
            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
              <Stat label="Risk score" value={data.risk_score.toFixed(4)} />
              <Stat label="Queue rank" value={data.queue_rank ? `#${data.queue_rank}` : "—"} />
              <Stat label="Time step" value={String(data.timestep)} />
              <Stat
                label="Graph score"
                value={data.graph_score !== null ? data.graph_score.toFixed(4) : "—"}
              />
              <Stat
                label="Confidence"
                value={data.confidence !== null ? data.confidence.toFixed(3) : "Phase 4"}
              />
              <Stat label="Status" value={data.status} />
            </dl>

            <p className="mt-2 font-mono text-[11px] text-zinc-600">
              scored by {data.model_version}
              {data.alert_budget !== null &&
                ` · alert budget ${(data.alert_budget * 100).toFixed(0)}%`}
            </p>

            <section className="mt-6">
              <h3 className="text-xs font-semibold uppercase tracking-wide text-zinc-500">
                Neighbourhood
              </h3>
              <dl className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3">
                <Stat label="In-degree" value={String(data.neighbourhood.in_degree)} />
                <Stat label="Out-degree" value={String(data.neighbourhood.out_degree)} />
                <Stat label="Total degree" value={String(data.neighbourhood.total_degree)} />
                <Stat label="Neighbours" value={String(data.neighbourhood.neighbour_count)} />
                <Stat
                  label="Same batch"
                  value={String(data.neighbourhood.same_batch_neighbours)}
                />
                <Stat label="Chain length" value={String(data.neighbourhood.chain_length)} />
              </dl>
              {data.neighbourhood.same_batch_neighbours ===
                data.neighbourhood.neighbour_count && (
                <p className="mt-2 text-[11px] text-zinc-600">
                  All neighbours are in this batch — the dataset has no
                  cross-time-step edges, which is why historical context comes from
                  structural similarity instead.
                </p>
              )}
            </section>

            <section className="mt-6">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-zinc-500">
                Evidence
              </h3>
              <EvidenceList items={data.evidence} />
            </section>

            <p className="mt-6 border-t border-white/10 pt-3 text-[11px] text-zinc-600">
              Ground-truth label:{" "}
              <span className="font-mono text-zinc-500">{data.label}</span> — shown
              because this is a research dataset. The investigation pipeline never
              reads it and no evidence is derived from it.
            </p>
          </>
        )}
      </div>
    </aside>
  );
}
