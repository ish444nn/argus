import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getOverview, type Overview as OverviewData } from "../api/client";
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

function pct(value: number) {
  return `${(value * 100).toFixed(2)}%`;
}

function RiskBands({ data }: { data: OverviewData }) {
  const max = Math.max(...data.risk_distribution.map((b) => b.count), 1);
  const total = data.risk_distribution.reduce((sum, b) => sum + b.count, 0);
  if (!total) return <p className="text-[var(--text-3)]">Nothing scored yet</p>;

  return (
    <div>
      <ul className="space-y-2">
        {data.risk_distribution.map((band, i) => (
          <li key={band.band} className="flex items-center gap-3 text-[12px]">
            <span className="num w-16 shrink-0 text-[var(--text-3)]">{band.band}</span>
            <span className="relative h-3.5 flex-1 bg-[var(--surface-2)]">
              {/* Everything scored in this band. */}
              <span
                className="absolute inset-y-0 left-0"
                style={{
                  width: `${(band.count / max) * 100}%`,
                  background: RISK_COLOURS[i],
                  opacity: 0.35,
                }}
              />
              {/* What the selected budget would take from it. */}
              {band.would_alert > 0 && (
                <span
                  className="absolute inset-y-0 left-0 border-r-2"
                  style={{
                    width: `${(band.would_alert / max) * 100}%`,
                    background: RISK_COLOURS[i],
                    borderColor: "var(--text)",
                  }}
                  title={`${band.would_alert} would be alerted at ${pct(data.alert_budget)}`}
                />
              )}
            </span>
            <span className="num w-24 shrink-0 text-right text-[var(--text-2)]">
              {band.would_alert.toLocaleString()}
              <span className="text-[var(--text-3)]">/{band.count.toLocaleString()}</span>
            </span>
          </li>
        ))}
      </ul>
      <p
        className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]"
        title="Solid: selected into the queue at the current budget. Faint: scored, not selected."
      >
        <span className="num text-[var(--text-2)]">{total.toLocaleString()}</span>{" "}
        scored · selected / in band
      </p>
    </div>
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

function BudgetControl({
  budget,
  onChange,
  data,
  pending,
}: {
  budget: number;
  onChange: (value: number) => void;
  data: OverviewData;
  pending: boolean;
}) {
  const preview = data.budget_preview;
  const index = Math.max(0, BUDGETS.indexOf(budget));
  const defaultIndex = BUDGETS.indexOf(DEFAULT_BUDGET);

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <p className="num text-[1.5rem] leading-none text-[var(--text)]">
          {budgetLabel(budget)}
        </p>
        <p className="text-right text-[11px] text-[var(--text-3)]" role="status">
          {pending ? (
            <span className="flex items-center justify-end gap-1.5 text-[var(--text-2)]">
              <span
                className="pulse-dot size-1.5 rounded-full bg-[var(--measured)]"
                aria-hidden
              />
              Recounting
            </span>
          ) : (
            <>
              <span className="num text-[var(--text-2)]">
                {preview.selected.toLocaleString()}
              </span>{" "}
              of {preview.scored.toLocaleString()} selected
            </>
          )}
        </p>
      </div>

      <input
        type="range"
        className="range mt-3"
        min={0}
        max={BUDGETS.length - 1}
        step={1}
        value={index}
        onChange={(e) => onChange(BUDGETS[Number(e.target.value)])}
        aria-label="Alert budget"
        aria-valuetext={`${budgetLabel(budget)}${budget === DEFAULT_BUDGET ? ", the default" : ""}`}
        title="How much of each scored batch enters the queue. Re-selects from the existing scores; it does not retrain the model or change a risk score."
      />

      {/* Ticks double as the scale and the affordance: seven stops, each mark
          sitting exactly where the thumb stops on it. */}
      <div className="relative mt-1.5 h-9">
        {BUDGETS.map((option, i) => (
          <button
            key={option}
            className="group absolute top-0 flex -translate-x-1/2 flex-col items-center gap-1 py-0.5"
            style={{ left: tickLeft(i, BUDGETS.length) }}
            onClick={() => onChange(option)}
            tabIndex={-1}
            aria-hidden
          >
            <span
              className="h-1.5 w-px"
              style={{
                background:
                  i === index
                    ? "var(--measured)"
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
    </div>
  );
}

export function Overview() {
  const { budget, setBudget } = useBudget();
  const { data, isPending, error, refetch, isFetching } = useQuery({
    queryKey: ["overview", budget],
    queryFn: () => getOverview(budget),
    placeholderData: (previous) => previous,
    refetchInterval: 20_000,
  });

  // The previous budget's figures are still on screen until the API answers
  // for the new one. Comparing what was asked for with what the response was
  // counted at is the real state of the request — no timer involved.
  const recounting = data !== undefined && data.alert_budget !== budget;

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
                  them. The risk distribution has five labelled bands and wants
                  the room; the decisions card is three rows and does not. Two
                  rows of unequal spans read as a composition; six identical
                  thirds read as a form. */}
              <div className="mt-6 grid grid-cols-1 items-start gap-4 lg:grid-cols-2 xl:grid-cols-12">
                <Panel
                  title="Risk distribution"
                  // The budget is a top-k within each batch, not across the
                  // pool, so saying which makes the band counts add up.
                  meta={`top ${budgetLabel(budget)} of each batch`}
                  className="xl:col-span-6"
                >
                  <div
                    className="panel-body transition-opacity duration-150"
                    style={{ opacity: recounting ? 0.5 : 1 }}
                  >
                    <RiskBands data={data} />
                  </div>
                </Panel>

                <Panel
                  title="Alert budget"
                  meta={budget === DEFAULT_BUDGET ? "canonical" : "exploring"}
                  className="xl:col-span-3"
                >
                  <div className="panel-body">
                    <BudgetControl
                      budget={budget}
                      onChange={setBudget}
                      data={data}
                      pending={recounting}
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

                <div className="xl:col-span-5">
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
