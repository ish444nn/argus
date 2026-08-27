import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { getQueue, type QueueEntry } from "../api/client";
import { Badge, Note, RiskLadder, Skeleton } from "../components/ui";

/**
 * The risk queue — the screen an analyst lives on.
 *
 * Sort lives in the URL, so a view can be shared, bookmarked and restored by
 * the back button. That matters more here than anywhere else: an analyst
 * working through the queue wants to come back to the order they left.
 *
 * Every column the eye scans down can be sorted, and nothing else is offered.
 * The queue is one page of the highest-risk work; slicing it further was
 * machinery for a problem an analyst does not have.
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

/**
 * A sortable column head.
 *
 * One component so every sortable column carries the identical affordance —
 * the arrow appears only on the active column, and `aria-sort` follows it, so
 * a screen reader is told the same thing the arrow says.
 */
function SortHeader({
  label,
  field,
  sortBy,
  descending,
  onSort,
  title,
}: {
  label: string;
  field: string;
  sortBy: string;
  descending: boolean;
  onSort: (field: string) => void;
  title?: string;
}) {
  const active = sortBy === field;
  return (
    <button
      className="sortable"
      onClick={() => onSort(field)}
      aria-sort={active ? (descending ? "descending" : "ascending") : "none"}
      title={title}
    >
      {label}
      <span aria-hidden style={{ color: active ? "var(--measured)" : "transparent" }}>
        {active && !descending ? "↑" : "↓"}
      </span>
    </button>
  );
}

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

  const sortBy = params.get("sort") ?? "risk_score";
  const descending = params.get("dir") !== "asc";
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

  const queue = useQuery({
    queryKey: ["queue", sortBy, descending, page],
    queryFn: () =>
      getQueue({
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
                  {total === 1 ? "" : "s"} · selected by risk score, ranked
                  within each batch
                </>
              )}
            </p>
          </div>

          <button
            className="btn"
            onClick={() => queue.refetch()}
            disabled={queue.isFetching}
          >
            {queue.isFetching ? "Refreshing" : "Refresh"}
          </button>
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
            <Note title="No cases here">
              Replay a batch to populate the queue.
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
                <th scope="col" style={{ width: 100 }}>
                  <SortHeader
                    label="Batch rank"
                    field="queue_rank"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="Position within its own batch, by risk score. Rank 1 is the highest-risk transaction in that batch."
                  />
                </th>
                <th scope="col" style={{ width: 84 }}>
                  <SortHeader
                    label="Batch"
                    field="timestep"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="The Elliptic time step this transaction was scored in."
                  />
                </th>
                <th scope="col">Transaction</th>
                <th scope="col" style={{ width: 160 }}>
                  <SortHeader
                    label="Risk score"
                    field="risk_score"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="XGBoost. The only signal that decides queue membership: each batch is ranked by it and cut at the alert budget."
                  />
                </th>
                <th scope="col" style={{ width: 140 }}>
                  <SortHeader
                    label="Second opinion"
                    field="graph_score"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="GraphSAGE's own score. Shown for comparison; it does not decide queue membership."
                  />
                </th>
                <th scope="col" style={{ width: 122 }}>
                  <SortHeader
                    label="Confidence"
                    field="confidence"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="Evidence confidence: how strongly the gathered evidence supports the case. Computed from the evidence, not from a model, and it decides nothing."
                  />
                </th>
                <th scope="col" style={{ width: 160 }}>
                  <SortHeader
                    label="Status"
                    field="status"
                    sortBy={sortBy}
                    descending={descending}
                    onSort={toggleSort}
                    title="Where the case sits in the workflow. Ascending puts the least progressed first."
                  />
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
                  <td className="num text-[var(--text-3)]">{entry.timestep}</td>
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
