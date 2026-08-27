import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  applyBudget,
  getHealth,
  getOverview,
  type Overview as OverviewData,
} from "../api/client";
import { BUDGETS, budgetLabel, DEFAULT_BUDGET, useBudget } from "../budget";
import { BatchReplay } from "../components/BatchReplay";
import { evidenceMeta } from "../evidence";
import { Distribution, Note, Panel, Skeleton, Stat } from "../components/ui";

/**
 * Operations overview.
 *
 * Every number is counted from Postgres. There is no chart here that is not
 * backed by rows, and where nothing has been recorded the panel says so
 * rather than drawing an empty axis.
 *
 * The screen answers three questions in order: what has been processed, what
 * is waiting for me, and is the machinery behaving.
 */

const RISK_COLOURS = [
  "var(--risk-1)",
  "var(--risk-1)",
  "var(--risk-2)",
  "var(--risk-3)",
  "var(--risk-4)",
];

function RiskBands({ data }: { data: OverviewData }) {
  const max = Math.max(...data.risk_distribution.map((b) => b.count), 1);
  const total = data.risk_distribution.reduce((sum, b) => sum + b.count, 0);
  if (!total) return <p className="text-[var(--text-3)]">Nothing scored yet</p>;

  return (
    <ul className="space-y-2.5">
      {data.risk_distribution.map((band, i) => (
        <li key={band.band} className="flex items-center gap-3 text-[12px]">
          <span className="num w-[4.5rem] shrink-0 text-[var(--text-3)]">{band.band}</span>
          <span className="relative h-4 flex-1 bg-[var(--surface-2)]">
            {/* Everything scored in this band. */}
            <span
              className="absolute inset-y-0 left-0"
              style={{
                width: `${(band.count / max) * 100}%`,
                background: RISK_COLOURS[i],
                opacity: 0.3,
              }}
            />
            {/* What the budget takes from it. The low bands hold tens of
                thousands of transactions and the high ones hold hundreds, so
                on a shared linear scale the selected slice of a high band is
                a fraction of a pixel wide -- and the selected slice is the
                entire point of the chart. A floor of 2px keeps it visible
                without misstating the width, and the numerals carry the
                actual quantity. */}
            {band.would_alert > 0 && (
              <span
                className="absolute inset-y-0 left-0 border-r"
                style={{
                  width: `max(2px, ${(band.would_alert / max) * 100}%)`,
                  background: RISK_COLOURS[i],
                  borderColor: "var(--text)",
                }}
              />
            )}
          </span>
          <span className="num w-[5.5rem] shrink-0 text-right">
            <span
              style={{
                color: band.would_alert ? "var(--text)" : "var(--text-3)",
              }}
            >
              {band.would_alert.toLocaleString()}
            </span>
            <span className="text-[var(--text-3)]">/{band.count.toLocaleString()}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}

/**
 * Where a tick belongs under a discrete range input.
 *
 * The thumb's centre travels from half a thumb in to half a thumb from the far
 * end, never across the full width, so laying seven labels out with
 * `justify-between` puts every one of them slightly off its own value. This is
 * that travel, expressed exactly; `--range-thumb` is the same token the thumb
 * itself is sized from.
 */
function tickLeft(index: number, count: number): string {
  const fraction = count > 1 ? index / (count - 1) : 0;
  return `calc(var(--range-thumb) / 2 + (100% - var(--range-thumb)) * ${fraction})`;
}

/**
 * What the chosen budget is doing right now.
 *
 * Derived from data, never from the number itself. `canonical` means the
 * stored queue was genuinely built at this budget; `exploring` means the
 * screen is previewing a budget nobody has applied yet, so the queue still
 * holds the other one; `applying` means the batches are being rebuilt.
 */
type BudgetState = "canonical" | "exploring" | "applying" | "mixed";

/** Budgets are decimal fractions read back through JSON; compare them as such. */
function sameBudget(a: number | null, b: number | null): boolean {
  if (a === null || b === null) return false;
  return Math.abs(a - b) < 1e-9;
}

const STATE_META: Record<BudgetState, string> = {
  canonical: "applied to the queue",
  exploring: "preview — not applied",
  applying: "rebuilding the queue",
  mixed: "batches disagree",
};

function BudgetControl({
  budget,
  onChange,
  data,
  state,
  progress,
  onApply,
  canApply,
  error,
}: {
  budget: number;
  onChange: (value: number) => void;
  data: OverviewData;
  state: BudgetState;
  progress: { done: number; total: number };
  onApply: () => void;
  canApply: boolean;
  error: string | null;
}) {
  const preview = data.budget_preview;
  const index = Math.max(0, BUDGETS.indexOf(budget));
  const defaultIndex = BUDGETS.indexOf(DEFAULT_BUDGET);
  const applying = state === "applying";
  const applied = data.applied_alert_budget;

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <p className="num text-[1.5rem] leading-none text-[var(--text)]">
          {budgetLabel(budget)}
        </p>
        <p className="text-right text-[11px] text-[var(--text-3)]">
          <span className="num text-[var(--text-2)]">
            {(state === "canonical"
              ? data.batches.queued
              : preview.selected
            ).toLocaleString()}
          </span>{" "}
          of {preview.scored.toLocaleString()}
        </p>
      </div>

      <input
        type="range"
        className="range mt-3"
        min={0}
        max={BUDGETS.length - 1}
        step={1}
        value={index}
        disabled={applying}
        onChange={(e) => onChange(BUDGETS[Number(e.target.value)])}
        aria-label="Alert budget"
        aria-valuetext={`${budgetLabel(budget)}, ${STATE_META[state]}`}
        title="The fraction of each scored batch that becomes an alert. Applying it re-runs the selection, so the queue changes."
      />

      {/* Ticks double as the scale and the affordance: seven stops, each mark
          sitting exactly where the thumb stops on it. */}
      <div className="relative mt-1.5 h-9">
        {BUDGETS.map((option, i) => (
          <button
            key={option}
            className="group absolute top-0 flex -translate-x-1/2 flex-col items-center gap-1 py-0.5"
            style={{ left: tickLeft(i, BUDGETS.length) }}
            onClick={() => !applying && onChange(option)}
            tabIndex={-1}
            aria-hidden
          >
            <span
              className="h-1.5 w-px"
              style={{
                background:
                  i === index
                    ? "var(--measured)"
                    : sameBudget(option, applied)
                      ? "var(--text-2)"
                      : i === defaultIndex
                        ? "var(--text-3)"
                        : "var(--line-2)",
              }}
            />
            <span
              className="num text-[10px] leading-none transition-colors group-hover:text-[var(--text-2)]"
              style={{ color: i === index ? "var(--text)" : "var(--text-3)" }}
            >
              {budgetLabel(option)}
            </span>
            {i === defaultIndex && (
              <span className="text-[9px] uppercase leading-none tracking-[0.08em] text-[var(--text-3)]">
                default
              </span>
            )}
          </button>
        ))}
      </div>

      {/* The action, and only when there is something to do. Changing the
          budget changes which transactions are alerts, which means re-running
          the selection -- so it is an operation with a duration, not a
          setting that takes effect on release. */}
      <div className="mt-1 border-t border-[var(--line)] pt-2.5">
        {applying ? (
          <p
            className="flex items-center gap-2 text-[12px] text-[var(--text-2)]"
            role="status"
            aria-live="polite"
          >
            <span
              className="pulse-dot size-1.5 rounded-full bg-[var(--measured)]"
              aria-hidden
            />
            Rebuilding the queue — batch{" "}
            <span className="num">
              {Math.min(progress.done + 1, progress.total)}
            </span>{" "}
            of <span className="num">{progress.total}</span>
          </p>
        ) : state === "canonical" ? (
          <p className="text-[12px] text-[var(--text-3)]">
            The queue holds the top{" "}
            <span className="num text-[var(--text-2)]">{budgetLabel(budget)}</span> of
            every batch.
          </p>
        ) : (
          <>
            <button
              className="btn btn-primary w-full !justify-center"
              onClick={onApply}
              disabled={!canApply}
              title={
                canApply
                  ? undefined
                  : "Rebuilding the queue runs the scoring job, which needs the local Celery worker."
              }
            >
              Rebuild the queue at {budgetLabel(budget)}
            </button>
            <p className="mt-2 text-[11px] text-[var(--text-3)]">
              {applied === null
                ? "Batches were built at different budgets."
                : `The queue is still the top ${budgetLabel(applied)}.`}
            </p>
          </>
        )}
        {error && (
          <p className="mt-2 text-[11px] text-[var(--bad)]" role="alert">
            {error}
          </p>
        )}
      </div>
    </div>
  );
}

export function Overview() {
  const { budget, setBudget } = useBudget();
  const queryClient = useQueryClient();

  // The budget an apply is in flight for, or null. Cleared by the data
  // itself: the run is finished when every batch has stopped and the stored
  // queue reports the budget we asked for.
  const [applyingFor, setApplyingFor] = useState<number | null>(null);

  const { data, isPending, error, refetch, isFetching } = useQuery({
    queryKey: ["overview", budget],
    queryFn: () => getOverview(budget),
    placeholderData: (previous) => previous,
    // Rebuilding the queue is a job with a duration, so the screen watches it
    // rather than waiting for the next slow poll.
    refetchInterval: (query) =>
      (query.state.data?.batches.running ?? 0) > 0 || applyingFor !== null ? 1500 : 20_000,
  });

  const health = useQuery({ queryKey: ["health"], queryFn: getHealth });
  const workerUp = health.data?.dependencies?.worker?.status === "ok";

  const apply = useMutation({
    mutationFn: () => applyBudget(budget),
    onMutate: () => setApplyingFor(budget),
    onError: () => setApplyingFor(null),
  });

  // Every derived state below reads from the response, not from the number.
  const applied = data?.applied_alert_budget ?? null;
  const running = data?.batches.running ?? 0;
  const settled =
    data !== undefined && running === 0 && sameBudget(applied, applyingFor);
  const applying = apply.isPending || (applyingFor !== null && !settled);

  const state: BudgetState = applying
    ? "applying"
    : applied === null
      ? "mixed"
      : sameBudget(applied, budget)
        ? "canonical"
        : "exploring";

  // When the rebuild lands, the queue's contents have changed underneath any
  // cached page of it. Drop them so the Queue screen cannot show the old
  // selection while this one shows the new count.
  const wasApplying = useRef(false);
  useEffect(() => {
    if (wasApplying.current && !applying) {
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      queryClient.invalidateQueries({ queryKey: ["batches"] });
      queryClient.invalidateQueries({ queryKey: ["applied-budget"] });
    }
    wasApplying.current = applying;
  }, [applying, queryClient]);

  return (
    <div className="mx-auto max-w-[1500px] px-6 py-6">
      <header className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-cond text-[1.5rem] font-semibold tracking-[0.02em]">
            Operations
          </h1>
          <p className="mt-1 max-w-[70ch] text-[var(--text-2)]">
            Replay state and queue composition, counted from the database.
          </p>
        </div>
        <button
          className="btn"
          onClick={() => refetch()}
          disabled={isFetching}
          aria-label="Refresh operations data"
        >
          {isFetching ? "Refreshing" : "Refresh"}
        </button>
      </header>

      {isPending && <Skeleton rows={6} />}

      {error && (
        <Note kind="error" title="Cannot reach the API">
          {(error as Error).message}. Check that the stack is running, then refresh.
        </Note>
      )}

      {data && (
        <>
          {data.batches.runs === 0 ? (
            <Note title="Nothing replayed yet">
              Replay a batch to populate the queue. Time steps{" "}
              <span className="num">{data.replay_range[0]}</span>–
              <span className="num">{data.replay_range[1]}</span> are available;
              earlier ones are training data and are never scored.
            </Note>
          ) : (
            <>
              <dl className="grid grid-cols-2 gap-x-6 gap-y-4 lg:grid-cols-5">
                <Stat
                  label="Batches replayed"
                  value={data.batches.runs}
                  hint={
                    data.batches.latest_timestep !== null
                      ? `latest time step ${data.batches.latest_timestep}`
                      : undefined
                  }
                />
                <Stat
                  label="Transactions scored"
                  value={data.batches.scored.toLocaleString()}
                  hint="by the primary model"
                />
                <Stat
                  label="Alerts raised"
                  value={data.batches.queued.toLocaleString()}
                  hint="top slice of each batch by rank"
                />
                <Stat
                  label="Awaiting review"
                  value={data.cases.awaiting_review.toLocaleString()}
                  tone={data.cases.awaiting_review ? "var(--risk-3)" : undefined}
                  hint="no analyst decision recorded"
                />
                <Stat
                  label="Investigated"
                  value={`${data.cases.ready}/${data.cases.total}`}
                  hint={
                    data.cases.failed
                      ? `${data.cases.failed} failed`
                      : "reports written"
                  }
                />
              </dl>

              {/* Bento.
                  Twelve columns, and each card takes the width its content
                  actually needs rather than a third because there are three of
                  them. Two rows of unequal spans read as a composition; six
                  identical thirds read as a form.
                  `items-stretch` rather than `items-start`: the cards in a row
                  carry different amounts of content, and letting each stop at
                  its own height left the row with a ragged bottom edge that
                  read as an accident rather than a decision. */}
              <div className="mt-5 grid grid-cols-1 items-stretch gap-4 lg:grid-cols-2 xl:grid-cols-12">
                <Panel
                  title="Risk distribution"
                  // The budget is a top-k within each batch, not across the
                  // pool, so saying which makes the band counts add up.
                  meta={`top ${budgetLabel(budget)} of each batch`}
                  className="xl:col-span-5"
                >
                  <div
                    className="panel-body transition-opacity duration-150"
                    style={{ opacity: applying ? 0.5 : 1 }}
                  >
                    <RiskBands data={data} />
                  </div>
                </Panel>

                <Panel
                  title="Alert budget"
                  meta={STATE_META[state]}
                  className="xl:col-span-4"
                >
                  <div className="panel-body">
                    <BudgetControl
                      budget={budget}
                      onChange={setBudget}
                      data={data}
                      state={state}
                      progress={{
                        done: data.batches.runs - running,
                        total: data.batches.runs,
                      }}
                      onApply={() => apply.mutate()}
                      canApply={workerUp && !applying}
                      error={apply.error ? (apply.error as Error).message : null}
                    />
                  </div>
                </Panel>

                <Panel
                  title="Investigation state"
                  meta="case lifecycle"
                  className="xl:col-span-3"
                >
                  <div className="panel-body">
                    <Distribution
                      items={[
                        { label: "Report written", count: data.cases.ready, colour: "var(--ok)" },
                        {
                          label: "Running",
                          count: data.cases.investigating,
                          colour: "var(--model)",
                        },
                        {
                          label: "Not started",
                          count: data.cases.queued,
                          colour: "var(--line-2)",
                        },
                        { label: "Failed", count: data.cases.failed, colour: "var(--bad)" },
                      ]}
                    />
                    {data.cases.ready > 0 && (
                      <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
                        <span className="num text-[var(--model)]">
                          {data.cases.model_written}
                        </span>{" "}
                        written by the model,{" "}
                        <span className="num text-[var(--text-2)]">
                          {data.cases.rule_written}
                        </span>{" "}
                        by rule.
                      </p>
                    )}
                  </div>
                </Panel>

                <div className="xl:col-span-5 flex">
                  <BatchReplay />
                </div>

                <Panel
                  title="Evidence gathered"
                  meta="across all cases"
                  className="xl:col-span-4"
                >
                  <div className="panel-body">
                    <Distribution
                      items={Object.entries(data.evidence).map(([kind, count]) => {
                        const meta = evidenceMeta(kind);
                        return { label: meta.label, count, colour: meta.colour };
                      })}
                      emptyLabel="No evidence gathered yet"
                    />
                  </div>
                </Panel>

                <Panel
                  title="Analyst decisions"
                  meta="latest per case"
                  className="xl:col-span-3"
                >
                  <div className="panel-body">
                    <Distribution
                      items={[
                        {
                          label: "Confirmed",
                          count: data.decisions.confirmed ?? 0,
                          colour: "var(--confirmed)",
                        },
                        {
                          label: "Dismissed",
                          count: data.decisions.dismissed ?? 0,
                          colour: "var(--dismissed)",
                        },
                        {
                          label: "Needs more evidence",
                          count: data.decisions.needs_more_evidence ?? 0,
                          colour: "var(--more)",
                        },
                      ]}
                      emptyLabel="No decisions recorded yet"
                    />
                    <Link to="/queue" className="btn mt-3 w-full !justify-center">
                      Review the queue
                    </Link>
                  </div>
                </Panel>
              </div>

              <p className="mt-6 border-t border-[var(--line)] pt-3 text-[11px] text-[var(--text-3)]">
                Typology corpus: {data.corpus.chunks} passages from{" "}
                {data.corpus.sources} sources across {data.corpus.publishers}{" "}
                publishers, embedded by{" "}
                <span className="num">{data.corpus.embedding_model ?? "—"}</span>.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
