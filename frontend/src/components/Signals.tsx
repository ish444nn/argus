import type { CaseDetail } from "../api/client";
import { Meter } from "./ui";

/**
 * The three signals, side by side.
 *
 * Argus produces three numbers and they mean genuinely different things. Shown
 * as a row of equivalent tiles they would read as three competing predictions,
 * which is exactly wrong, so each one states its source and what it decides:
 *
 *   Risk score          XGBoost              decides the queue
 *   Second opinion      GraphSAGE            decides nothing
 *   Evidence confidence deterministic evidence  decides nothing
 *
 * Only the first has an accent rail and a "decides the queue" line. The other
 * two are visually quieter on purpose — they inform the analyst, they do not
 * select the work.
 */

function Tile({
  label,
  value,
  source,
  role,
  meter,
  accent,
  detail,
  title,
}: {
  label: string;
  value: string;
  source: string;
  role: string;
  meter?: { value: number; colour: string };
  accent?: boolean;
  detail?: React.ReactNode;
  title?: string;
}) {
  return (
    <div
      className={`flex-1 border p-4 ${
        accent
          ? "border-[color-mix(in_oklab,var(--measured)_45%,var(--line))] bg-[color-mix(in_oklab,var(--measured)_7%,transparent)]"
          : "border-[var(--line)] bg-[var(--surface)]"
      }`}
      title={title}
    >
      <p className="eyebrow" style={accent ? { color: "var(--measured)" } : undefined}>
        {label}
      </p>
      <p className="num mt-1.5 text-[1.75rem] leading-none text-[var(--text)]">{value}</p>
      {meter && (
        <div className="mt-2">
          <Meter value={meter.value} colour={meter.colour} width={120} />
        </div>
      )}
      <p className="mt-2.5 border-t border-[var(--line)] pt-2 text-[11px] leading-snug text-[var(--text-3)]">
        <span className="text-[var(--text-2)]">{source}</span>
        <br />
        {role}
      </p>
      {detail && <div className="mt-1.5 text-[11px] text-[var(--text-3)]">{detail}</div>}
    </div>
  );
}

export function Signals({ detail }: { detail: CaseDetail }) {
  const confidence = detail.confidence ?? 0;

  // Counted from the evidence on the case rather than from the investigation
  // metadata, because a case has evidence — and therefore a confidence — long
  // before anyone presses "Run investigation", and the panel has to explain
  // the number at that point too.
  const contributingKinds = new Set(
    detail.evidence.filter((item) => item.weight > 0).map((item) => item.kind),
  ).size;

  return (
    <section aria-label="Case signals">
      <div className="flex flex-col gap-3 lg:flex-row">
        <Tile
          accent
          label="Risk score"
          value={detail.risk_score.toFixed(4)}
          source="XGBoost · primary scorer"
          role="Ranks the batch and decides which transactions become alerts."
          meter={{ value: detail.risk_score, colour: "var(--measured)" }}
          detail={
            detail.queue_rank ? (
              <>
                Batch rank <span className="num">{detail.queue_rank}</span> of batch{" "}
                <span className="num">{detail.timestep}</span>
                {detail.alert_budget !== null && (
                  <> · top {(detail.alert_budget * 100).toFixed(1)}%</>
                )}
              </>
            ) : undefined
          }
          title="The production risk score. This is the only signal that determines queue membership."
        />

        <Tile
          label="Second opinion"
          value={detail.graph_score !== null ? detail.graph_score.toFixed(4) : "—"}
          source="GraphSAGE · graph model"
          role="An independent view from a neighbourhood-aware model. Decides nothing."
          meter={
            detail.graph_score !== null
              ? { value: detail.graph_score, colour: "var(--model)" }
              : undefined
          }
          detail="Also produces the embeddings behind historical similarity."
          title="A second model's opinion, shown for comparison. It does not affect the queue or the evidence confidence."
        />

        <Tile
          label="Evidence confidence"
          value={confidence.toFixed(3)}
          source="Deterministic evidence"
          role="How strongly the evidence Argus gathered supports the case. Not a model score."
          meter={{ value: confidence, colour: "var(--cited)" }}
          detail={
            contributingKinds > 0
              ? `From ${contributingKinds} kind${contributingKinds === 1 ? "" : "s"} of evidence`
              : "No supporting evidence found."
          }
          title="Computed from the deterministic evidence on this case as soon as it is gathered — before any investigation runs — and independent of both model scores."
        />
      </div>
    </section>
  );
}
