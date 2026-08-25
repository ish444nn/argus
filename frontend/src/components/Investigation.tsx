import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  getCitedSources,
  startInvestigation,
  type CaseDetail,
} from "../api/client";

/**
 * The investigation panel.
 *
 * The layout enforces the distinction the whole design rests on. Three
 * separately labelled blocks, in order of authority:
 *
 *   Observed          measured by Argus. Facts.
 *   AI interpretation written by a language model from those facts.
 *   External          retrieved from published typology guidance, quoted.
 *
 * A reader should never have to guess which of the three they are looking at,
 * so each is visually distinct and captioned with where it came from.
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

const ACTION_STYLES: Record<string, string> = {
  escalate: "bg-rose-500/15 text-rose-300 ring-rose-500/30",
  monitor: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
  dismiss: "bg-zinc-500/15 text-zinc-300 ring-zinc-500/30",
};

function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full ${
            value >= 0.35 ? "bg-emerald-400" : "bg-amber-400"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-xs text-zinc-300">{value.toFixed(3)}</span>
    </div>
  );
}

function SectionLabel({
  children,
  origin,
}: {
  children: React.ReactNode;
  origin: string;
}) {
  return (
    <div className="mb-2 flex items-baseline gap-2">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-400">
        {children}
      </h4>
      <span className="text-[11px] text-zinc-600">{origin}</span>
    </div>
  );
}

export function Investigation({ detail }: { detail: CaseDetail }) {
  const queryClient = useQueryClient();
  const meta = detail.investigation_meta ?? {};
  const investigated = Boolean(detail.narrative);

  const sources = useQuery({
    queryKey: ["sources", detail.case_id],
    queryFn: () => getCitedSources(detail.case_id),
    enabled: investigated,
  });

  const run = useMutation({
    mutationFn: () => startInvestigation(detail.case_id),
    onSuccess: () => {
      // The task is asynchronous; poll the case until the status settles.
      const poll = setInterval(() => {
        queryClient.invalidateQueries({ queryKey: ["case", detail.case_id] });
        queryClient.invalidateQueries({ queryKey: ["sources", detail.case_id] });
      }, 2000);
      setTimeout(() => clearInterval(poll), 30_000);
    },
  });

  if (!investigated) {
    return (
      <section className="mt-6">
        <SectionLabel origin="not yet run">Investigation</SectionLabel>
        <div className="rounded-lg border border-dashed border-white/10 p-4">
          <p className="text-sm text-zinc-400">
            This case has deterministic evidence but no written assessment yet.
          </p>
          <button
            onClick={() => run.mutate()}
            disabled={run.isPending || detail.status === "investigating"}
            className="mt-3 rounded bg-white/10 px-3 py-1.5 text-xs font-medium text-zinc-100 hover:bg-white/20 disabled:opacity-50"
          >
            {run.isPending || detail.status === "investigating"
              ? "Investigating…"
              : "Run investigation"}
          </button>
          {run.error && (
            <p className="mt-2 text-xs text-rose-400">
              {(run.error as Error).message}
            </p>
          )}
        </div>
      </section>
    );
  }

  const fromModel = detail.narrative_source === "llm";

  return (
    <section className="mt-6 space-y-5">
      {/* --- Assessment ------------------------------------------------- */}
      <div>
        <SectionLabel origin="computed from evidence, not model-reported">
          Risk assessment
        </SectionLabel>
        <div className="rounded-lg border border-white/10 bg-zinc-900/50 p-4">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <div>
              <p className="text-xs text-zinc-500">Confidence</p>
              <div className="mt-1">
                <ConfidenceBar value={detail.confidence ?? 0} />
              </div>
            </div>
            <div>
              <p className="text-xs text-zinc-500">Queue tier</p>
              <p className="mt-1 font-mono text-sm text-zinc-100">
                {detail.queue_tier ?? "—"}
              </p>
            </div>
            {detail.recommended_action && (
              <div>
                <p className="text-xs text-zinc-500">Suggested action</p>
                <span
                  className={`mt-1 inline-block rounded px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${
                    ACTION_STYLES[detail.recommended_action] ??
                    "bg-white/10 text-zinc-300 ring-white/20"
                  }`}
                >
                  {detail.recommended_action}
                </span>
              </div>
            )}
          </div>
          {meta.confidence_contributions && (
            <p className="mt-3 font-mono text-[11px] text-zinc-600">
              {Object.entries(meta.confidence_contributions)
                .map(([kind, value]) => `${kind} ${value.toFixed(3)}`)
                .join("  ·  ")}
            </p>
          )}
        </div>
      </div>

      {/* --- AI interpretation ------------------------------------------ */}
      <div>
        <SectionLabel
          origin={
            fromModel
              ? `written by ${meta.provider ?? "model"}/${meta.model ?? "?"}`
              : "assembled by rule — no model output was used"
          }
        >
          {fromModel ? "AI interpretation" : "Generated summary"}
        </SectionLabel>
        <div
          className={`rounded-lg border p-4 ${
            fromModel
              ? "border-sky-500/25 bg-sky-500/5"
              : "border-white/10 bg-zinc-900/50"
          }`}
        >
          {detail.typology_assessment && (
            <p className="mb-2">
              <span className="text-xs text-zinc-500">Likely typology: </span>
              <span className="text-sm font-medium text-zinc-100">
                {TYPOLOGY_LABELS[detail.typology_assessment] ??
                  detail.typology_assessment}
              </span>
            </p>
          )}
          <p className="whitespace-pre-line text-sm leading-relaxed text-zinc-200">
            {detail.narrative}
          </p>
          <p className="mt-3 border-t border-white/10 pt-2 text-[11px] text-zinc-600">
            {fromModel
              ? "Interpretation, not a finding. Every claim cites the evidence id or source it rests on; the analyst decides."
              : "No API key configured, so this was built from the evidence by rule rather than written by a model."}
          </p>
        </div>
      </div>

      {/* --- External typology knowledge -------------------------------- */}
      <div>
        <SectionLabel origin="retrieved and quoted, never generated">
          External typology sources
        </SectionLabel>
        {sources.isPending && (
          <p className="text-sm text-zinc-500">Loading sources…</p>
        )}
        {sources.data && sources.data.length === 0 && (
          <p className="rounded-lg border border-dashed border-white/10 p-3 text-sm text-zinc-500">
            No typology source was cited for this case.
          </p>
        )}
        <ul className="space-y-2">
          {sources.data?.map((source) => (
            <li
              key={source.evidence_id}
              className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-4"
            >
              <div className="flex flex-wrap items-baseline gap-2">
                <span className="text-sm font-medium text-zinc-100">
                  {source.title}
                </span>
                <span className="text-xs text-zinc-500">
                  {source.section_heading}
                </span>
                <span className="ml-auto font-mono text-[11px] text-zinc-600">
                  match {source.similarity.toFixed(3)}
                </span>
              </div>
              <blockquote className="mt-2 border-l-2 border-emerald-500/30 pl-3 text-sm leading-relaxed text-zinc-300">
                {source.text}
              </blockquote>
              <p className="mt-2 text-[11px] text-zinc-500">
                {source.publisher}
                {source.document && `, “${source.document}”`}
                {source.year && `, ${source.year}`} ·{" "}
                <a
                  href={source.source_url}
                  target="_blank"
                  rel="noreferrer"
                  className="underline decoration-dotted hover:text-zinc-300"
                >
                  source
                </a>
              </p>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-[11px] text-zinc-600">
        Retrieved for {(meta.retrieval_patterns ?? []).join(", ") || "no pattern"}
        {meta.attempts ? ` · ${meta.attempts} generation attempt(s)` : ""}
        {meta.validation_errors?.length
          ? ` · ${meta.validation_errors.length} citation error(s) rejected`
          : ""}
      </p>
    </section>
  );
}
