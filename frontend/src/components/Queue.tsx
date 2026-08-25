import { useQuery } from "@tanstack/react-query";
import { getBatches, getQueue } from "../api/client";

function RiskCell({ value }: { value: number }) {
  return (
    <span className="font-mono text-sm text-zinc-100">{value.toFixed(4)}</span>
  );
}

export function Queue({
  selectedCaseId,
  onSelect,
  timestep,
  onTimestepChange,
}: {
  selectedCaseId: number | null;
  onSelect: (caseId: number) => void;
  timestep: number | undefined;
  onTimestepChange: (timestep: number | undefined) => void;
}) {
  const batches = useQuery({ queryKey: ["batches"], queryFn: getBatches });
  const queue = useQuery({
    queryKey: ["queue", timestep],
    queryFn: () => getQueue({ timestep, limit: 200 }),
  });

  return (
    <section className="flex h-full flex-col">
      <header className="flex flex-wrap items-center gap-3 border-b border-white/10 px-5 py-3">
        <h2 className="text-sm font-semibold">Risk queue</h2>
        {queue.data && (
          <span className="text-xs text-zinc-500">
            {queue.data.total} case{queue.data.total === 1 ? "" : "s"}
          </span>
        )}
        <select
          value={timestep ?? ""}
          onChange={(e) =>
            onTimestepChange(e.target.value ? Number(e.target.value) : undefined)
          }
          className="ml-auto rounded border border-white/10 bg-zinc-900 px-2 py-1 text-xs text-zinc-200"
        >
          <option value="">All batches</option>
          {batches.data?.map((run) => (
            <option key={run.timestep} value={run.timestep}>
              Time step {run.timestep} ({run.queued_count} of {run.scored_count})
            </option>
          ))}
        </select>
      </header>

      <div className="flex-1 overflow-y-auto">
        {queue.isPending && <p className="px-5 py-6 text-sm text-zinc-400">Loading queue…</p>}
        {queue.error && (
          <p className="px-5 py-6 text-sm text-rose-400">
            {(queue.error as Error).message}
          </p>
        )}
        {queue.data && queue.data.items.length === 0 && (
          <p className="px-5 py-6 text-sm text-zinc-400">
            No cases yet. Replay a batch to populate the queue.
          </p>
        )}

        {queue.data && queue.data.items.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead className="sticky top-0 bg-zinc-950 text-xs uppercase tracking-wide text-zinc-500">
              <tr className="border-b border-white/10">
                <th className="px-5 py-2 font-medium">Rank</th>
                <th className="px-2 py-2 font-medium">Transaction</th>
                <th className="px-2 py-2 font-medium">Risk</th>
                <th className="px-2 py-2 font-medium">Graph</th>
                <th className="px-2 py-2 font-medium">Batch</th>
                <th className="px-5 py-2 text-right font-medium">Evidence</th>
              </tr>
            </thead>
            <tbody>
              {queue.data.items.map((entry) => (
                <tr
                  key={entry.case_id}
                  onClick={() => onSelect(entry.case_id)}
                  className={`cursor-pointer border-b border-white/5 hover:bg-white/5 ${
                    selectedCaseId === entry.case_id ? "bg-white/10" : ""
                  }`}
                >
                  <td className="px-5 py-2 font-mono text-xs text-zinc-500">
                    {entry.queue_rank ?? "—"}
                  </td>
                  <td className="px-2 py-2 font-mono text-xs text-zinc-300">
                    {entry.tx_id}
                  </td>
                  <td className="px-2 py-2">
                    <RiskCell value={entry.risk_score} />
                  </td>
                  <td className="px-2 py-2 font-mono text-xs text-zinc-400">
                    {entry.graph_score !== null ? entry.graph_score.toFixed(3) : "—"}
                  </td>
                  <td className="px-2 py-2 font-mono text-xs text-zinc-500">
                    {entry.timestep}
                  </td>
                  <td className="px-5 py-2 text-right font-mono text-xs text-zinc-400">
                    {entry.evidence_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
