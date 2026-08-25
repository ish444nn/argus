import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { getCitedSources, startInvestigation, type CaseDetail } from "../api/client";
import { Badge, Meter, Note, Panel, ProvLabel, Skeleton } from "./ui";

/**
 * The investigation.
 *
 * Two panels, and the split is the whole point: what a model wrote, and what
 * was quoted from a published source. Each wears its provenance rail, so a
 * reader never has to work out which they are looking at.
 *
 * Confidence sits with the model panel but is labelled as measured, because
 * it is computed from evidence rather than stated by the model — that
 * distinction is easy to lose and expensive to lose.
 */

const TYPOLOGY_LABELS: Record<string, string> = {
  structuring: "Structuring",
  funnelling: "Funnelling",
  layering: "Layering",
  mixing_or_obfuscation: "Mixing or obfuscation",
  mule_network: "Mule network",
  network_association: "Network association",
  no_clear_typology: "No clear typology",
};

/**
 * The corpus is written as hard-wrapped markdown, so quoting it verbatim drags
 * the source file's 76-column line breaks into a much narrower panel and the
 * text comes out ragged. Paragraphs (blank lines) are meaningful and kept;
 * single newlines are just the file's wrapping and are collapsed.
 */
function paragraphs(text: string): string[] {
  return text
    .split(/\n\s*\n/)
    .map((block) => block.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean);
}

const ACTION_TONE: Record<string, "bad" | "warn" | "neutral"> = {
  escalate: "bad",
  monitor: "warn",
  dismiss: "neutral",
};

export function Assessment({ detail }: { detail: CaseDetail }) {
  const queryClient = useQueryClient();
  const meta = detail.investigation_meta ?? {};
  const written = Boolean(detail.narrative);
  const running = detail.status === "investigating";
  const fromModel = detail.narrative_source === "llm";

  const run = useMutation({
    mutationFn: () => startInvestigation(detail.case_id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["case", detail.case_id] });
    },
  });

  if (detail.status === "failed" && !written) {
    return (
      <Panel title="Investigation">
        <div className="panel-body">
          <Note kind="error" title="The investigation failed">
            {detail.error ?? "No reason was recorded."}
            <div className="mt-3">
              <button className="btn" onClick={() => run.mutate()} disabled={run.isPending}>
                Try again
              </button>
            </div>
          </Note>
        </div>
      </Panel>
    );
  }

  if (!written) {
    return (
      <Panel title="Investigation">
        <div className="panel-body">
          {running ? (
            <div className="flex items-center gap-2.5 text-[var(--text-2)]">
              <span
                className="pulse-dot size-2 rounded-full bg-[var(--model)]"
                aria-hidden
              />
              <p role="status">
                Retrieving typology sources and writing the assessment…
              </p>
            </div>
          ) : (
            <Note title="Not investigated yet">
              The deterministic evidence is gathered. Running the investigation
              retrieves matching typology passages and writes a cited assessment.
              <div className="mt-3">
                <button
                  className="btn btn-primary"
                  onClick={() => run.mutate()}
                  disabled={run.isPending}
                >
                  {run.isPending ? "Starting…" : "Run investigation"}
                </button>
              </div>
              {run.error && (
                <p className="mt-2 text-[var(--bad)]" role="alert">
                  {(run.error as Error).message}
                </p>
              )}
            </Note>
          )}
        </div>
      </Panel>
    );
  }

  return (
    <div className="space-y-4">
      <Panel
        title="Assessment"
        actions={
          <button
            className="btn"
            onClick={() => run.mutate()}
            disabled={run.isPending || running}
            title="Re-run the investigation"
          >
            {run.isPending || running ? "Running" : "Re-run"}
          </button>
        }
      >
        <div className="panel-body space-y-4">
          {/* Confidence is measured, not model-stated — labelled as such. */}
          <div className="prov prov-measured">
            <ProvLabel origin="measured">Computed from evidence</ProvLabel>
            <div className="mt-2 flex flex-wrap items-end gap-x-8 gap-y-3">
              <div>
                <p className="eyebrow">Confidence</p>
                <div className="mt-1 flex items-center gap-2.5">
                  <span className="num text-[1.5rem] leading-none">
                    {(detail.confidence ?? 0).toFixed(3)}
                  </span>
                  <Meter
                    value={detail.confidence ?? 0}
                    width={90}
                    colour={
                      (detail.confidence ?? 0) >= 0.35 ? "var(--ok)" : "var(--warn)"
                    }
                  />
                </div>
              </div>
              <div>
                <p className="eyebrow">Queue tier</p>
                <p className="mt-1.5">
                  <Badge tone={detail.queue_tier === "primary" ? "measured" : "neutral"}>
                    {detail.queue_tier ?? "—"}
                  </Badge>
                </p>
              </div>
            </div>
            {meta.confidence_contributions && (
              <p className="num mt-2 text-[11px] text-[var(--text-3)]">
                {Object.entries(meta.confidence_contributions)
                  .map(([kind, value]) => `${kind.replace(/_/g, " ")} ${value.toFixed(3)}`)
                  .join("   ")}
              </p>
            )}
          </div>

          {/* What the model wrote. */}
          <div className={`prov ${fromModel ? "prov-model" : "prov-measured"}`}>
            <ProvLabel
              origin={fromModel ? "model" : "measured"}
              detail={
                fromModel
                  ? `${meta.provider ?? "model"} · ${meta.model ?? "?"}`
                  : "assembled by rule — no model output was used"
              }
            >
              {fromModel ? "Written by a model" : "Built from the evidence"}
            </ProvLabel>

            <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2">
              {detail.typology_assessment && (
                <p>
                  <span className="eyebrow">Likely typology</span>{" "}
                  <span className="ml-1.5 text-[var(--text)]">
                    {TYPOLOGY_LABELS[detail.typology_assessment] ??
                      detail.typology_assessment}
                  </span>
                </p>
              )}
              {detail.recommended_action && (
                <p className="flex items-center gap-2">
                  <span className="eyebrow">Suggests</span>
                  <Badge tone={ACTION_TONE[detail.recommended_action] ?? "neutral"}>
                    {detail.recommended_action}
                  </Badge>
                </p>
              )}
            </div>

            <p className="mt-3 max-w-[78ch] whitespace-pre-line leading-relaxed text-[var(--text)]">
              {detail.narrative}
            </p>

            <p className="mt-3 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
              {fromModel
                ? "Interpretation, not a finding. Each claim cites the evidence id or source it rests on; a response citing anything else is rejected before it is stored."
                : "No model narrative was used, so this was assembled from the evidence by rule."}
            </p>
          </div>
        </div>
      </Panel>

    </div>
  );
}

/**
 * The passages the report cites, quoted in full.
 *
 * Its own panel, and deliberately placed beside the evidence rather than the
 * assessment: this is reference material an analyst reads to judge whether the
 * typology claim holds, not part of the claim itself.
 */
export function TypologySources({ detail }: { detail: CaseDetail }) {
  const meta = detail.investigation_meta ?? {};
  const sources = useQuery({
    queryKey: ["sources", detail.case_id],
    queryFn: () => getCitedSources(detail.case_id),
    enabled: Boolean(detail.narrative),
  });

  if (!detail.narrative) return null;

  return (
      <Panel
        title="AML typology sources"
        meta="retrieved and quoted, never generated"
      >
        <div className="panel-body">
          {sources.isPending && <Skeleton rows={2} />}
          {sources.data?.length === 0 && (
            <Note title="No source cited">
              No typology passage matched this case&rsquo;s patterns, so the report
              makes no typology claim.
            </Note>
          )}
          <div className="space-y-4">
            {sources.data?.map((source) => (
              <article key={source.evidence_id} className="prov prov-cited">
                <ProvLabel
                  origin="cited"
                  detail={`match ${source.similarity.toFixed(3)}`}
                >
                  {source.typology_id}
                </ProvLabel>
                <h4 className="mt-1.5 text-[var(--text)]">
                  {source.title}
                  <span className="ml-2 font-normal text-[var(--text-3)]">
                    {source.section_heading}
                  </span>
                </h4>
                <blockquote className="mt-2 max-w-[78ch] space-y-2 text-[var(--text-2)]">
                  {paragraphs(source.text).map((para, i) => (
                    <p key={i}>{para}</p>
                  ))}
                </blockquote>
                <p className="mt-2 text-[11px] text-[var(--text-3)]">
                  {source.publisher}
                  {source.document && `, “${source.document}”`}
                  {source.year && `, ${source.year}`} ·{" "}
                  <a
                    href={source.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline decoration-dotted underline-offset-2 hover:text-[var(--text-2)]"
                  >
                    open source
                  </a>
                </p>
              </article>
            ))}
          </div>

          {(meta.retrieval_patterns?.length ?? 0) > 0 && (
            <p className="mt-4 border-t border-[var(--line)] pt-2 text-[11px] text-[var(--text-3)]">
              Retrieved by filtering the corpus to{" "}
              <span className="num">{meta.retrieval_patterns?.join(", ")}</span>, then
              ranking by similarity.
              {meta.attempts ? ` ${meta.attempts} generation attempt(s).` : ""}
              {meta.validation_errors?.length
                ? ` ${meta.validation_errors.length} response(s) rejected for citing sources that were not retrieved.`
                : ""}
            </p>
          )}
        </div>
      </Panel>
  );
}
