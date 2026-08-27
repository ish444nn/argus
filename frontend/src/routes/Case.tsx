import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getCase, type CaseDetail } from "../api/client";
import { EgoGraph } from "../components/EgoGraph";
import { EvidenceList } from "../components/Evidence";
import { Assessment, TypologySources } from "../components/Investigation";
import { Review } from "../components/Review";
import { Signals } from "../components/Signals";
import { Note, Panel, ProvLabel, Skeleton } from "../components/ui";

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

/** A dispatched investigation, and the case row as it stood when it went out. */
type Dispatched = { caseId: number; updatedAt: string };

/**
 * Is an investigation still in flight?
 *
 * The POST returns 202 as soon as the task is queued, which is before the
 * worker has picked it up and set the status -- so the status alone cannot
 * answer this, and a panel that trusted it flicked back to its button within a
 * frame and looked like it had done nothing. A run is finished only once the
 * case row has changed *and* settled somewhere other than `investigating`.
 */
function stillRunning(
  dispatched: Dispatched | null,
  caseId: number,
  detail: CaseDetail | undefined,
): boolean {
  if (!dispatched || dispatched.caseId !== caseId) return false;
  if (!detail) return true;
  return detail.updated_at === dispatched.updatedAt || detail.status === "investigating";
}

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

  const [dispatched, setDispatched] = useState<Dispatched | null>(null);

  const { data, isPending, error } = useQuery({
    queryKey: ["case", id],
    queryFn: () => getCase(id),
    // Poll while an investigation is in flight -- either one this page
    // dispatched, or one already running when the page opened.
    refetchInterval: (query) =>
      query.state.data?.status === "investigating" ||
      stillRunning(dispatched, id, query.state.data)
        ? 1500
        : false,
  });

  const running = stillRunning(dispatched, id, data);

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
      <header className="border-b border-[var(--line)] pb-4">
        <p className="eyebrow">Case {data.case_id}</p>
        <h1 className="num mt-1 text-[1.75rem] leading-none">{data.tx_id}</h1>
        <p className="mt-2 text-[var(--text-2)]">
          Batch <span className="num">{data.timestep}</span> · batch rank{" "}
          <span className="num">{data.queue_rank ?? "—"}</span> · scored by{" "}
          <span className="num">{data.model_version}</span>
        </p>
      </header>

      {/* --- The three signals -------------------------------------------- */}
      <div className="mt-4">
        <Signals detail={data} />
      </div>

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
          <Assessment
            detail={data}
            running={running}
            onDispatch={() =>
              setDispatched({ caseId: id, updatedAt: data.updated_at })
            }
            onDispatchFailed={() => setDispatched(null)}
          />
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
