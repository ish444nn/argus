import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { getBatches, getQueue, type QueueEntry } from "../api/client";
import { Badge, Note, RiskLadder, Skeleton } from "../components/ui";

/**
 * The risk queue — the screen an analyst lives on.
 *
 * Filters and sort live in the URL, so a view can be shared, bookmarked and
 * restored by the back button. That matters more here than anywhere else: an
 * analyst working a batch wants to come back to exactly the slice they left.
 *
 * Risk reads three ways at once — a four-step ladder, the numeral, and colour
 * — so the scan works without relying on hue.
 */

const PAGE_SIZE = 50;

const DECISION_TONE: Record<string, "bad" | "neutral" | "warn"> = {
  confirmed: "bad",
  dismissed: "neutral",
  needs_more_evidence: "warn",
};

const DECISION_LABEL: Record<string, string> = {
  confirmed: "Confirmed",
  dismissed: "Dismissed",
  needs_more_evidence: "More evidence",
};

function StatusCell({ entry }: { entry: QueueEntry }) {
  if (entry.latest_decision) {
    return (
      <Badge tone={DECISION_TONE[entry.latest_decision] ?? "neutral"}>
        {DECISION_LABEL[entry.latest_decision] ?? entry.latest_decision}
      </Badge>
    );
  }
  if (entry.status === "investigating") {
    return (
      <span className="eyebrow flex items-center gap-1.5 !text-[var(--model)]">
        <span className="pulse-dot size-1.5 rounded-full bg-[var(--model)]" aria-hidden />
        Investigating
      </span>
    );
  }
  if (entry.status === "failed") {
    return <Badge tone="bad">Failed</Badge>;
  }
  if (entry.status === "ready") {
    return <span className="eyebrow !text-[var(--text-2)]">Awaiting review</span>;
  }
  return <span className="eyebrow">Not investigated</span>;
}

export function Queue() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();

  const timestep = params.get("timestep");
  const sortBy = params.get("sort") ?? "risk_score";
  const descending = params.get("dir") !== "asc";
  const undecided = params.get("undecided") === "1";
  const page = Number(params.get("page") ?? 0);

  const patch = (next: Record<string, string | null>) => {
    const merged = new URLSearchParams(params);
    for (const [key, value] of Object.entries(next)) {
      if (value === null) merged.delete(key);
      else merged.set(key, value);
    }
    if (!("page" in next)) merged.delete("page");
    setParams(merged, { replace: true });
  };

  const batches = useQuery({ queryKey: ["batches"], queryFn: getBatches });
  const queue = useQuery({
    queryKey: ["queue", timestep, sortBy, descending, undecided, page],
    queryFn: () =>
      getQueue({
        timestep: timestep ? Number(timestep) : undefined,
        undecidedOnly: undecided,
        sortBy,
        descending,
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
      }),
    placeholderData: (previous) => previous,
  });

  const total = queue.data?.total ?? 0;
  const shown = queue.data?.items.length ?? 0;
  const from = total ? page * PAGE_SIZE + 1 : 0;
  const to = page * PAGE_SIZE + shown;
  const isStale = queue.isPlaceholderData && queue.isFetching;

  const toggleSort = (value: string) => {
    if (sortBy === value) patch({ dir: descending ? "asc" : "desc" });
    else patch({ sort: value, dir: "desc" });
  };

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-[var(--line)] px-6 py-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-cond text-[1.5rem] font-semibold tracking-[0.02em]">
              Risk queue
            </h1>
            <p className="mt-1 text-[var(--text-2)]">
              {queue.isPending ? (
                "Loading…"
              ) : (
                <>
                  <span className="num">{total.toLocaleString()}</span> case
                  {total === 1 ? "" : "s"}
                  {timestep && <> in batch {timestep}</>}
                  {undecided && <> awaiting a decision</>}
                </>
              )}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="batch">
              Filter by batch
            </label>
            <select
              id="batch"
              className="select"
              value={timestep ?? ""}
              onChange={(e) => patch({ timestep: e.target.value || null })}
            >
              <option value="">All batches</option>
              {batches.data?.map((run) => (
                <option key={run.timestep} value={run.timestep}>
                  Batch {run.timestep} — {run.queued_count} of{" "}
                  {run.scored_count.toLocaleString()}
                </option>
              ))}
            </select>

            <button
              className="btn"
              aria-pressed={undecided}
              onClick={() => patch({ undecided: undecided ? null : "1" })}
              style={
                undecided
                  ? {
                      borderColor: "var(--measured)",
                      background: "color-mix(in oklab, var(--measured) 18%, var(--surface-2))",
                    }
                  : undefined
              }
            >
              Undecided only
            </button>

            <button
              className="btn"
              onClick={() => queue.refetch()}
              disabled={queue.isFetching}
            >
              {queue.isFetching ? "Refreshing" : "Refresh"}
            </button>
          </div>
        </div>
      </header>

      <div className="min-h-0 flex-1 overflow-auto">
        {queue.isPending && (
          <div className="p-6">
            <Skeleton rows={10} />
          </div>
        )}

        {queue.error && (
          <div className="p-6">
            <Note kind="error" title="Could not load the queue">
              {(queue.error as Error).message}
            </Note>
          </div>
        )}

        {queue.data && queue.data.items.length === 0 && (
          <div className="p-6">
            <Note title={undecided ? "Nothing awaiting a decision" : "No cases here"}>
              {undecided
                ? "Every case in this view has been decided. Clear the filter to see them."
                : "Replay a batch to populate the queue, or clear the batch filter."}
            </Note>
          </div>
        )}

        {queue.data && queue.data.items.length > 0 && (
          <table
            className="tbl"
            style={{ opacity: isStale ? 0.55 : 1, transition: "opacity 120ms linear" }}
          >
            <caption className="sr-only">
              Flagged transactions, sorted by {sortBy}, {descending ? "highest" : "lowest"}{" "}
              first
            </caption>
            <thead>
              <tr>
                <th scope="col" style={{ width: 56 }}>
                  <button
                    className="sortable"
                    onClick={() => toggleSort("queue_rank")}
                    aria-sort={
                      sortBy === "queue_rank"
                        ? descending
                          ? "descending"
                          : "ascending"
                        : "none"
                    }
                  >
                    Rank {sortBy === "queue_rank" && (descending ? "↓" : "↑")}
                  </button>
                </th>
                <th scope="col">Transaction</th>
                <th scope="col" style={{ width: 150 }}>
                  <button
                    className="sortable"
                    onClick={() => toggleSort("risk_score")}
                    aria-sort={
                      sortBy === "risk_score"
                        ? descending
                          ? "descending"
                          : "ascending"
                        : "none"
                    }
                  >
                    Risk {sortBy === "risk_score" && (descending ? "↓" : "↑")}
                  </button>
                </th>
                <th scope="col" style={{ width: 110 }}>
                  <button
                    className="sortable"
                    onClick={() => toggleSort("graph_score")}
                    aria-sort={
                      sortBy === "graph_score"
                        ? descending
                          ? "descending"
                          : "ascending"
                        : "none"
                    }
                  >
                    Graph {sortBy === "graph_score" && (descending ? "↓" : "↑")}
                  </button>
                </th>
                <th scope="col" style={{ width: 70 }}>
                  Batch
                </th>
                <th scope="col" style={{ width: 90 }}>
                  Evidence
                </th>
                <th scope="col" style={{ width: 100 }}>
                  Confidence
                </th>
                <th scope="col" style={{ width: 150 }}>
                  Status
                </th>
              </tr>
            </thead>
            <tbody>
              {queue.data.items.map((entry) => (
                <tr
                  key={entry.case_id}
                  tabIndex={0}
                  onClick={() => navigate(`/cases/${entry.case_id}`)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      navigate(`/cases/${entry.case_id}`);
                    }
                  }}
                >
                  <td className="num text-[var(--text-3)]">{entry.queue_rank ?? "—"}</td>
                  <td className="num">{entry.tx_id}</td>
                  <td>
                    <span className="flex items-center gap-2">
                      <RiskLadder score={entry.risk_score} />
                      <span className="num">{entry.risk_score.toFixed(4)}</span>
                    </span>
                  </td>
                  <td className="num text-[var(--text-2)]">
                    {entry.graph_score !== null ? entry.graph_score.toFixed(3) : "—"}
                  </td>
                  <td className="num text-[var(--text-3)]">{entry.timestep}</td>
                  <td className="num text-[var(--text-2)]">{entry.evidence_count}</td>
                  <td className="num text-[var(--text-2)]">
                    {entry.confidence !== null ? entry.confidence.toFixed(3) : "—"}
                  </td>
                  <td>
                    <StatusCell entry={entry} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {total > PAGE_SIZE && (
        <footer className="flex items-center justify-between border-t border-[var(--line)] px-6 py-3">
          <p className="text-[12px] text-[var(--text-3)]">
            <span className="num">{from}</span>–<span className="num">{to}</span> of{" "}
            <span className="num">{total.toLocaleString()}</span>
          </p>
          <div className="flex gap-2">
            <button
              className="btn"
              disabled={page === 0}
              onClick={() => patch({ page: String(page - 1) })}
            >
              Previous
            </button>
            <button
              className="btn"
              disabled={to >= total}
              onClick={() => patch({ page: String(page + 1) })}
            >
              Next
            </button>
          </div>
        </footer>
      )}
    </div>
  );
}
