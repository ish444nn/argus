import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getOverview, type Overview as OverviewData } from "../api/client";
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

const EVIDENCE_LABELS: Record<string, string> = {
  structural_similarity: "Structural similarity",
  graph_model_corroboration: "Graph-model corroboration",
  heuristic: "Structural heuristic",
  flagged_neighbour: "Flagged neighbour",
  confirmed_neighbour: "Confirmed neighbour",
  typology_reference: "Typology citation",
};

const TYPOLOGY_LABELS: Record<string, string> = {
  structuring: "Structuring",
  funnelling: "Funnelling",
  layering: "Layering",
  mixing_or_obfuscation: "Mixing / obfuscation",
  mule_network: "Mule network",
  network_association: "Network association",
  no_clear_typology: "No clear typology",
};

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
              <span
                className="absolute inset-y-0 left-0"
                style={{
                  width: `${(band.count / max) * 100}%`,
                  background: RISK_COLOURS[i],
                  opacity: 0.55,
                }}
              />
              {band.alerted > 0 && (
                <span
                  className="absolute inset-y-0 left-0 border-r-2"
                  style={{
                    width: `${(band.alerted / max) * 100}%`,
                    background: RISK_COLOURS[i],
                    borderColor: "var(--text)",
                  }}
                  title={`${band.alerted} alerted`}
                />
              )}
            </span>
            <span className="num w-16 shrink-0 text-right text-[var(--text-2)]">
              {band.count.toLocaleString()}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
        <span className="num text-[var(--text-2)]">{total.toLocaleString()}</span>{" "}
        transactions scored. The solid segment is the portion alerted — the cut falls
        inside the top band, so scoring high is not by itself enough to reach the
        queue.
      </p>
    </div>
  );
}

function BudgetBar({ data }: { data: OverviewData }) {
  const realised = data.batches.realised_alert_rate;
  if (realised === null) return null;
  // Scale the bar so the configured budget sits at 70% of the width — the
  // point is how close realised sits to budget, not its absolute magnitude.
  const scale = data.alert_budget / 0.7;
  const width = Math.min(100, (realised / scale) * 100);
  const budgetAt = 70;

  return (
    <div>
      <div className="relative h-6 border border-[var(--line)] bg-[var(--surface-2)]">
        <div
          className="h-full"
          style={{ width: `${width}%`, background: "color-mix(in oklab, var(--measured) 40%, transparent)" }}
        />
        <div
          className="absolute inset-y-0 border-l border-dashed"
          style={{ left: `${budgetAt}%`, borderColor: "var(--text-2)" }}
        >
          <span className="absolute -top-0.5 left-1.5 whitespace-nowrap text-[10px] text-[var(--text-2)]">
            budget {pct(data.alert_budget)}
          </span>
        </div>
      </div>
      <p className="mt-2 text-[11px] text-[var(--text-3)]">
        <span className="num text-[var(--text-2)]">{pct(realised)}</span> of scored
        transactions were alerted across {data.batches.runs} batch
        {data.batches.runs === 1 ? "" : "es"}. The budget is applied by ranking each
        batch and taking its top slice, so the realised rate tracks it rather than
        drifting.
      </p>
    </div>
  );
}

export function Overview() {
  const { data, isPending, error, refetch, isFetching } = useQuery({
    queryKey: ["overview"],
    queryFn: getOverview,
    refetchInterval: 20_000,
  });

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

              <div className="mt-6 grid grid-cols-1 items-start gap-4 xl:grid-cols-3">
                <Panel title="Alert budget" meta="realised vs configured">
                  <div className="panel-body">
                    <BudgetBar data={data} />
                  </div>
                </Panel>

                <Panel title="Risk distribution" meta="all scored transactions">
                  <div className="panel-body">
                    <RiskBands data={data} />
                  </div>
                </Panel>

                <Panel title="Investigation state" meta="case lifecycle">
                  <div className="panel-body">
                    <Distribution
                      items={[
                        {
                          label: "Report written",
                          count: data.cases.ready,
                          colour: "var(--ok)",
                        },
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
                        {
                          label: "Failed",
                          count: data.cases.failed,
                          colour: "var(--bad)",
                        },
                      ]}
                    />
                    {data.cases.ready > 0 && (
                      <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
                        Of the written reports,{" "}
                        <span className="num text-[var(--model)]">
                          {data.cases.model_written}
                        </span>{" "}
                        came from the model and{" "}
                        <span className="num text-[var(--text-2)]">
                          {data.cases.rule_written}
                        </span>{" "}
                        were built by rule.
                      </p>
                    )}
                  </div>
                </Panel>
              </div>

              <div className="mt-4 grid grid-cols-1 items-start gap-4 xl:grid-cols-3">
                <Panel title="Evidence gathered" meta="across all cases">
                  <div className="panel-body">
                    <Distribution
                      items={Object.entries(data.evidence).map(([kind, count]) => ({
                        label: EVIDENCE_LABELS[kind] ?? kind,
                        count,
                        colour:
                          kind === "typology_reference"
                            ? "var(--cited)"
                            : kind === "graph_model_corroboration"
                              ? "var(--model)"
                              : "var(--measured)",
                      }))}
                      emptyLabel="No evidence gathered yet"
                    />
                  </div>
                </Panel>

                <Panel title="Typology assessment" meta="model-assigned">
                  <div className="panel-body">
                    <Distribution
                      items={Object.entries(data.typologies).map(([key, count]) => ({
                        label: TYPOLOGY_LABELS[key] ?? key,
                        count,
                        colour: "var(--model)",
                      }))}
                      emptyLabel="No cases investigated yet"
                    />
                  </div>
                </Panel>

                <Panel title="Analyst decisions" meta="latest per case">
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
                    <Link
                      to="/queue?undecided=1"
                      className="btn mt-3 w-full !justify-center"
                    >
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
