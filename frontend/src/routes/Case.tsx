import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { getCase } from "../api/client";
import { EgoGraph } from "../components/EgoGraph";
import { EvidenceList } from "../components/Evidence";
import { Assessment, TypologySources } from "../components/Investigation";
import { Review } from "../components/Review";
import { Note, Panel, ProvLabel, RiskLadder, Skeleton } from "../components/ui";

/**
 * The case dossier.
 *
 * Its own route rather than a docked panel: a case is the unit of work, it
 * deserves the full width, and a URL an analyst can send to a colleague.
 *
 * Reading order runs left to right, evidence before interpretation. What Argus
 * measured occupies the wider column; what a model made of it, and the
 * decision, sit to the right. That ordering is an argument — the evidence
 * comes first and the interpretation is answerable to it.
 */

function Identity({
  label,
  value,
  mono = true,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="eyebrow">{label}</dt>
      <dd className={`mt-0.5 ${mono ? "num" : ""} text-[var(--text)]`}>{value}</dd>
    </div>
  );
}

export function Case() {
  const { caseId } = useParams();
  const id = Number(caseId);

  const { data, isPending, error } = useQuery({
    queryKey: ["case", id],
    queryFn: () => getCase(id),
    // Poll only while a background investigation is in flight.
    refetchInterval: (query) =>
      query.state.data?.status === "investigating" ? 2000 : false,
  });

  if (isPending) {
    return (
      <div className="mx-auto max-w-[1500px] px-6 py-6">
        <Skeleton rows={8} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-[1500px] px-6 py-6">
        <Note kind="error" title="Could not load the case">
          {(error as Error).message}
          <div className="mt-3">
            <Link to="/queue" className="btn">
              Back to the queue
            </Link>
          </div>
        </Note>
      </div>
    );
  }

  const n = data.neighbourhood;

  return (
    <div className="mx-auto max-w-[1500px] px-6 py-6">
      <nav className="mb-4">
        <Link
          to="/queue"
          className="eyebrow hover:text-[var(--text-2)]"
        >
          ← Risk queue
        </Link>
      </nav>

      {/* --- Identity ---------------------------------------------------- */}
      <header className="border-b border-[var(--line)] pb-5">
        <div className="flex flex-wrap items-start justify-between gap-6">
          <div>
            <p className="eyebrow">Case {data.case_id}</p>
            <h1 className="num mt-1 text-[1.75rem] leading-none">{data.tx_id}</h1>
            <p className="mt-2 text-[var(--text-2)]">
              Batch {data.timestep} · rank{" "}
              <span className="num">{data.queue_rank ?? "—"}</span> · scored by{" "}
              <span className="num">{data.model_version}</span>
            </p>
          </div>

          <div className="flex items-end gap-8">
            <div>
              <p className="eyebrow">Risk score</p>
              <div className="mt-1 flex items-center gap-2.5">
                <RiskLadder score={data.risk_score} />
                <span className="num text-[1.75rem] leading-none">
                  {data.risk_score.toFixed(4)}
                </span>
              </div>
              <p className="mt-1 text-[11px] text-[var(--text-3)]">
                top {data.alert_budget ? `${(data.alert_budget * 100).toFixed(0)}%` : "—"} of
                its batch
              </p>
            </div>
            <div>
              <p className="eyebrow">Graph score</p>
              <p className="num mt-1 text-[1.75rem] leading-none text-[var(--model)]">
                {data.graph_score !== null ? data.graph_score.toFixed(4) : "—"}
              </p>
              <p className="mt-1 text-[11px] text-[var(--text-3)]">
                second opinion only
              </p>
            </div>
          </div>
        </div>
      </header>

      {/* --- Two columns: evidence, then interpretation ------------------- */}
      <div className="mt-5 grid grid-cols-1 items-start gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Panel title="Transaction context" meta="measured from the graph">
            <div className="panel-body">
              <div className="prov prov-measured">
                <ProvLabel origin="measured" />
                <dl className="mt-3 grid grid-cols-3 gap-x-5 gap-y-3 sm:grid-cols-6">
                  <Identity label="In" value={n.in_degree} />
                  <Identity label="Out" value={n.out_degree} />
                  <Identity label="Degree" value={n.total_degree} />
                  <Identity label="Parties" value={n.neighbour_count} />
                  <Identity label="Chain len" value={n.chain_length} />
                  <Identity label="Flagged" value={n.flagged_neighbours} />
                </dl>
              </div>

              <div className="mt-4 border-t border-[var(--line)] pt-4">
                <EgoGraph txId={data.tx_id} />
              </div>
            </div>
          </Panel>

          <Panel title="Observed evidence" meta="what Argus found">
            <div className="panel-body">
              <EvidenceList items={data.evidence} />
            </div>
          </Panel>

          {/* Reference material sits with the evidence: an analyst reads it to
              judge whether the typology claim holds, so it belongs next to
              what the claim is about rather than next to the claim. */}
          <TypologySources detail={data} />
        </div>

        {/* The action column stays in view while the evidence scrolls. */}
        <div className="space-y-4 xl:sticky xl:top-6">
          <Assessment detail={data} />
          <Review caseId={data.case_id} />
        </div>
      </div>

      <footer className="mt-6 border-t border-[var(--line)] pt-3">
        <p className="text-[11px] text-[var(--text-3)]">
          Ground-truth label{" "}
          <span className="num text-[var(--text-2)]">{data.label}</span> — present
          because this is a research dataset. No tool reads it and no evidence item
          is derived from it.
        </p>
      </footer>
    </div>
  );
}
