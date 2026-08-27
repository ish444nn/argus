import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getAvailableBatches,
  getHealth,
  removeBatch,
  startReplay,
  type BatchAvailability,
  type BatchRemoved,
} from "../api/client";
import { Note, Panel, Skeleton } from "./ui";

/**
 * Batches.
 *
 * A batch is one Elliptic time step. Replaying it scores every transaction in
 * that time step, ranks them, and takes the top slice into the queue. Time
 * steps 1–34 are training data and the API refuses them, so only the test
 * range appears here at all.
 *
 * Removing a replayed batch undoes that: the run, its scores and its cases
 * go, and the time step returns to the list ready to replay again. This is a
 * fixed research dataset rather than a live feed, so being able to clear a
 * batch you are done with is worth more than protecting a replay you can
 * reproduce in a minute. Cases an analyst has already decided on survive, and
 * the result says how many did.
 *
 * Replay needs the Celery worker, which is local-only. Rather than hiding the
 * control on the deployed demo and pretending the feature does not exist, the
 * panel reads the health endpoint and disables it with a reason.
 */

function Row({
  batch,
  busy,
  disabled,
  confirming,
  onReplay,
  onAskRemove,
  onConfirmRemove,
  onCancelRemove,
}: {
  batch: BatchAvailability;
  busy: boolean;
  disabled: boolean;
  confirming: boolean;
  onReplay: (timestep: number) => void;
  onAskRemove: (timestep: number) => void;
  onConfirmRemove: (timestep: number) => void;
  onCancelRemove: () => void;
}) {
  return (
    <li className="flex items-center gap-3 border-t border-[var(--line)] py-2 first:border-t-0">
      <span className="num w-10 shrink-0">{batch.timestep}</span>
      <span className="num flex-1 truncate text-[12px] text-[var(--text-3)]">
        {batch.transactions.toLocaleString()}
      </span>

      {confirming ? (
        // Inline rather than a modal: the row is the context, and a dialog to
        // confirm one reversible delete interrupts more than it protects.
        <span className="flex items-center gap-2">
          <span className="text-[11px] text-[var(--text-2)]">Remove?</span>
          <button
            className="btn"
            onClick={() => onConfirmRemove(batch.timestep)}
            disabled={busy}
            style={{ borderColor: "var(--bad)", color: "var(--bad)" }}
          >
            {busy ? "Removing…" : "Yes"}
          </button>
          <button className="btn" onClick={onCancelRemove} disabled={busy}>
            No
          </button>
        </span>
      ) : batch.replayed ? (
        <button
          className="btn"
          onClick={() => onAskRemove(batch.timestep)}
          disabled={busy}
          title="Delete this batch's run, scores and cases. Reviewed cases are kept."
        >
          Remove
        </button>
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
  const [removed, setRemoved] = useState<BatchRemoved | null>(null);
  const [confirming, setConfirming] = useState<number | null>(null);

  const available = useQuery({
    queryKey: ["batches", "available"],
    queryFn: getAvailableBatches,
    refetchInterval: 20_000,
  });
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const workerUp = health.data?.dependencies?.worker?.status === "ok";

  const refresh = () => {
    queryClient.invalidateQueries({ queryKey: ["batches"] });
    queryClient.invalidateQueries({ queryKey: ["overview"] });
    queryClient.invalidateQueries({ queryKey: ["queue"] });
    queryClient.invalidateQueries({ queryKey: ["applied-budget"] });
  };

  const replay = useMutation({
    mutationFn: (timestep: number) => startReplay(timestep),
    onSuccess: (_data, timestep) => {
      setRemoved(null);
      setStarted(timestep);
      refresh();
    },
  });

  const remove = useMutation({
    mutationFn: (timestep: number) => removeBatch(timestep),
    onSuccess: (result) => {
      setStarted(null);
      setRemoved(result);
      setConfirming(null);
      refresh();
    },
  });

  const batches = available.data?.batches ?? [];
  const busy = replay.isPending || remove.isPending;
  const error = replay.error ?? remove.error;

  return (
    <Panel
      title="Batches"
      meta={
        available.data
          ? `time steps ${available.data.replayable_range[0]}–${available.data.replayable_range[1]}`
          : undefined
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
            <div className="flex items-center gap-3 pb-1 text-[11px] text-[var(--text-3)]">
              <span className="eyebrow w-10 shrink-0">Batch</span>
              <span className="eyebrow flex-1">Transactions</span>
              <span
                className="eyebrow"
                title={`Replaying takes the top ${(available.data.alert_budget * 100).toFixed(1)}% of a batch into the queue.`}
              >
                Action
              </span>
            </div>

            <ul className="max-h-[15.5rem] overflow-y-auto">
              {batches.map((batch) => (
                <Row
                  key={batch.timestep}
                  batch={batch}
                  busy={
                    (replay.isPending && replay.variables === batch.timestep) ||
                    (remove.isPending && remove.variables === batch.timestep)
                  }
                  disabled={!workerUp || busy}
                  confirming={confirming === batch.timestep}
                  onReplay={(ts) => {
                    setConfirming(null);
                    replay.mutate(ts);
                  }}
                  onAskRemove={setConfirming}
                  onConfirmRemove={(ts) => remove.mutate(ts)}
                  onCancelRemove={() => setConfirming(null)}
                />
              ))}
            </ul>

            {started !== null && !replay.isError && (
              <p className="mt-3 flex items-center gap-2 border-t border-[var(--line)] pt-2 text-[12px] text-[var(--text-2)]">
                <span
                  className="pulse-dot size-1.5 rounded-full bg-[var(--model)]"
                  aria-hidden
                />
                <span role="status">
                  Batch <span className="num">{started}</span> queued on the worker.
                </span>
              </p>
            )}

            {removed && (
              <p
                className="mt-3 border-t border-[var(--line)] pt-2 text-[12px] text-[var(--text-2)]"
                role="status"
              >
                Batch <span className="num">{removed.timestep}</span> removed —{" "}
                <span className="num">{removed.cases_removed}</span> case
                {removed.cases_removed === 1 ? "" : "s"} and{" "}
                <span className="num">{removed.scores_removed.toLocaleString()}</span> scores
                deleted.
                {removed.reviewed_retained > 0 && (
                  <>
                    {" "}
                    <span className="num">{removed.reviewed_retained}</span> reviewed case
                    {removed.reviewed_retained === 1 ? "" : "s"} kept.
                  </>
                )}
              </p>
            )}

            {error && (
              <p className="mt-3 text-[12px] text-[var(--bad)]" role="alert">
                {(error as Error).message}
              </p>
            )}

            {!workerUp && !health.isPending && (
              <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
                Replay needs the Celery worker, which runs locally only. The hosted
                demo serves a precomputed snapshot.
              </p>
            )}
          </>
        )}
      </div>
    </Panel>
  );
}
