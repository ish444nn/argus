import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAvailableBatches,
  getHealth,
  startReplay,
  type BatchAvailability,
} from "../api/client";
import { Note, Panel, Skeleton } from "./ui";

/**
 * Replaying a batch.
 *
 * A "batch" is one Elliptic time step. Replaying it scores every transaction
 * in that time step with XGBoost, ranks them, and takes the top slice into the
 * queue. Time steps 1–34 are training data and the API refuses them, so only
 * the test range is offered here at all.
 *
 * Replay runs on the Celery worker, which is local-only — there is no hosted
 * worker. Rather than hiding the control on the deployed demo and pretending
 * the feature does not exist, the panel reads the health endpoint and says
 * plainly why the button is disabled.
 */

function Row({
  batch,
  onReplay,
  busy,
  disabled,
}: {
  batch: BatchAvailability;
  onReplay: (timestep: number) => void;
  busy: boolean;
  disabled: boolean;
}) {
  return (
    <li className="flex items-center gap-3 border-t border-[var(--line)] py-2 first:border-t-0">
      <span className="num w-14 shrink-0">{batch.timestep}</span>
      <span className="num flex-1 text-[12px] text-[var(--text-3)]">
        {batch.transactions.toLocaleString()} transactions
      </span>
      {batch.replayed ? (
        <span className="eyebrow !text-[var(--text-3)]">replayed</span>
      ) : (
        <button
          className="btn"
          onClick={() => onReplay(batch.timestep)}
          disabled={busy || disabled}
        >
          {busy ? "Starting…" : "Replay"}
        </button>
      )}
    </li>
  );
}

export function BatchReplay() {
  const queryClient = useQueryClient();
  const [started, setStarted] = useState<number | null>(null);
  const [showAll, setShowAll] = useState(false);

  const available = useQuery({
    queryKey: ["batches", "available"],
    queryFn: getAvailableBatches,
    refetchInterval: 20_000,
  });
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const workerUp = health.data?.dependencies?.worker?.status === "ok";

  const replay = useMutation({
    mutationFn: (timestep: number) => startReplay(timestep),
    onSuccess: (_data, timestep) => {
      setStarted(timestep);
      queryClient.invalidateQueries({ queryKey: ["batches"] });
      queryClient.invalidateQueries({ queryKey: ["overview"] });
    },
  });

  const batches = available.data?.batches ?? [];
  const pending = batches.filter((b) => !b.replayed);
  // Unreplayed batches are the point of the panel, so they come first and the
  // already-done ones stay behind a toggle.
  const shown = showAll ? batches : pending.slice(0, 6);

  return (
    <Panel
      title="Replay a batch"
      meta={
        available.data
          ? `time steps ${available.data.replayable_range[0]}–${available.data.replayable_range[1]}`
          : undefined
      }
      actions={
        batches.length > pending.length && (
          <button className="btn" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Only pending" : "Show all"}
          </button>
        )
      }
    >
      <div className="panel-body">
        {available.isPending && <Skeleton rows={4} />}

        {available.error && (
          <Note kind="error" title="Could not list batches">
            {(available.error as Error).message}
          </Note>
        )}

        {available.data && (
          <>
            <p className="mb-3 max-w-[70ch] text-[11px] text-[var(--text-3)]">
              Each batch is one Elliptic time step. Replaying scores every
              transaction in it, ranks them, and takes the top{" "}
              <span className="num">
                {(available.data.alert_budget * 100).toFixed(1)}%
              </span>{" "}
              into the queue. Time steps before{" "}
              <span className="num">{available.data.replayable_range[0]}</span> are
              training data and are never scored.
            </p>

            {pending.length === 0 && !showAll ? (
              <Note title="Every batch has been replayed">
                All {batches.length} test time steps are in the queue. Replaying one
                again is safe and idempotent — use “Show all”.
              </Note>
            ) : (
              <ul>
                {shown.map((batch) => (
                  <Row
                    key={batch.timestep}
                    batch={batch}
                    busy={replay.isPending && replay.variables === batch.timestep}
                    disabled={!workerUp || replay.isPending}
                    onReplay={(ts) => replay.mutate(ts)}
                  />
                ))}
              </ul>
            )}

            {started !== null && !replay.isError && (
              <p className="mt-3 flex items-center gap-2 text-[12px] text-[var(--text-2)]">
                <span
                  className="pulse-dot size-1.5 rounded-full bg-[var(--model)]"
                  aria-hidden
                />
                <span role="status">
                  Batch <span className="num">{started}</span> queued on the worker.
                  Scores and cases appear as it finishes.
                </span>
              </p>
            )}

            {replay.error && (
              <p className="mt-3 text-[12px] text-[var(--bad)]" role="alert">
                {(replay.error as Error).message}
              </p>
            )}

            {!workerUp && !health.isPending && (
              <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
                Replay is disabled: no Celery worker is reachable. The worker and
                Redis run locally only — the hosted demo serves a precomputed
                snapshot, so its queue is fixed.
              </p>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}
