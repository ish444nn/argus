import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getReviews, recordReview } from "../api/client";
import { Badge, Note, Panel } from "./ui";

/**
 * The analyst decision.
 *
 * Three outcomes, matching the schema, and nothing more — this is a decision
 * log, not a case-management system.
 *
 * Decisions append rather than replace. Re-deciding a case adds to its
 * history, so what was concluded and when survives; the queue reads the most
 * recent entry.
 */

const DECISIONS = [
  {
    value: "confirmed",
    label: "Confirm",
    hint: "The evidence supports escalating this transaction.",
    tone: "bad" as const,
  },
  {
    value: "dismissed",
    label: "Dismiss",
    hint: "The evidence does not support escalation.",
    tone: "neutral" as const,
  },
  {
    value: "needs_more_evidence",
    label: "Needs more evidence",
    hint: "Cannot decide on what is here.",
    tone: "warn" as const,
  },
];

const DECISION_LABEL: Record<string, string> = {
  confirmed: "Confirmed",
  dismissed: "Dismissed",
  needs_more_evidence: "Needs more evidence",
};

function when(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

export function Review({ caseId }: { caseId: number }) {
  const queryClient = useQueryClient();
  const [note, setNote] = useState("");
  const [chosen, setChosen] = useState<string | null>(null);

  const reviews = useQuery({
    queryKey: ["reviews", caseId],
    queryFn: () => getReviews(caseId),
  });

  const submit = useMutation({
    mutationFn: (decision: string) => recordReview(caseId, decision, note),
    onSuccess: () => {
      setNote("");
      setChosen(null);
      queryClient.invalidateQueries({ queryKey: ["reviews", caseId] });
      queryClient.invalidateQueries({ queryKey: ["queue"] });
      queryClient.invalidateQueries({ queryKey: ["overview"] });
    },
  });

  const latest = reviews.data?.[0];

  return (
    <Panel
      title="Decision"
      meta={latest ? `last decided ${when(latest.created_at)}` : "not yet decided"}
    >
      <div className="panel-body space-y-4">
        {latest && (
          <div className="flex flex-wrap items-center gap-3 border-b border-[var(--line)] pb-3">
            <Badge
              tone={
                latest.decision === "confirmed"
                  ? "bad"
                  : latest.decision === "needs_more_evidence"
                    ? "warn"
                    : "neutral"
              }
            >
              {DECISION_LABEL[latest.decision] ?? latest.decision}
            </Badge>
            <span className="text-[12px] text-[var(--text-3)]">
              by {latest.analyst}
            </span>
            {latest.note && (
              <p className="w-full text-[var(--text-2)]">“{latest.note}”</p>
            )}
          </div>
        )}

        <fieldset>
          <legend className="eyebrow mb-2">
            {latest ? "Record a new decision" : "Record a decision"}
          </legend>
          <div className="space-y-1.5">
            {DECISIONS.map((option) => (
              <label
                key={option.value}
                className={`flex cursor-pointer items-start gap-2.5 border p-2.5 transition-colors ${
                  chosen === option.value
                    ? "border-[var(--measured)] bg-[var(--surface-2)]"
                    : "border-[var(--line)] hover:bg-[var(--surface-2)]"
                }`}
              >
                <input
                  type="radio"
                  name={`decision-${caseId}`}
                  value={option.value}
                  checked={chosen === option.value}
                  onChange={() => setChosen(option.value)}
                  className="mt-0.5 accent-[var(--measured)]"
                />
                <span className="min-w-0">
                  <span className="block text-[var(--text)]">{option.label}</span>
                  <span className="block text-[11px] text-[var(--text-3)]">
                    {option.hint}
                  </span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <div>
          <label htmlFor={`note-${caseId}`} className="eyebrow">
            Note <span className="normal-case tracking-normal">(optional)</span>
          </label>
          <textarea
            id={`note-${caseId}`}
            className="input mt-1.5 min-h-[64px] w-full resize-y"
            placeholder="What tipped the decision?"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>

        <div className="flex items-center gap-3">
          <button
            className="btn btn-primary"
            disabled={!chosen || submit.isPending}
            onClick={() => chosen && submit.mutate(chosen)}
          >
            {submit.isPending ? "Recording…" : "Record decision"}
          </button>
          {submit.isSuccess && (
            <span className="text-[12px] text-[var(--ok)]" role="status">
              Decision recorded.
            </span>
          )}
        </div>

        {submit.error && (
          <Note kind="error" title="Could not record the decision">
            {(submit.error as Error).message}
          </Note>
        )}

        {(reviews.data?.length ?? 0) > 1 && (
          <details className="border-t border-[var(--line)] pt-3">
            <summary className="eyebrow cursor-pointer">
              Earlier decisions ({(reviews.data?.length ?? 1) - 1})
            </summary>
            <ul className="mt-2 space-y-2">
              {reviews.data?.slice(1).map((review) => (
                <li key={review.review_id} className="text-[12px] text-[var(--text-3)]">
                  <span className="num">{when(review.created_at)}</span> —{" "}
                  {DECISION_LABEL[review.decision] ?? review.decision} by {review.analyst}
                  {review.note && <> · “{review.note}”</>}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </Panel>
  );
}
