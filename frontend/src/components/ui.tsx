/**
 * Shared primitives.
 *
 * Small on purpose — enough to stop the screens duplicating markup, not a
 * component library. Anything used once lives with the screen that uses it.
 *
 * The provenance components are the load-bearing ones: `Provenance` and
 * `ProvLabel` are how every block on every screen declares whether Argus
 * measured it, a model wrote it, or it was quoted from a source.
 */

import type { ReactNode } from "react";

export type Origin = "measured" | "model" | "cited";

const ORIGIN_TEXT: Record<Origin, string> = {
  measured: "Measured by Argus",
  model: "Written by a model",
  cited: "Quoted from a source",
};

const ORIGIN_COLOUR: Record<Origin, string> = {
  measured: "var(--measured)",
  model: "var(--model)",
  cited: "var(--cited)",
};

/** The left-edge rail. Line style differs per origin, not just colour. */
export function Provenance({
  origin,
  children,
  className = "",
}: {
  origin: Origin;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`prov prov-${origin} ${className}`}>{children}</div>
  );
}

/** The eyebrow that names an origin, with a matching line-style swatch. */
export function ProvLabel({
  origin,
  children,
  detail,
}: {
  origin: Origin;
  children?: ReactNode;
  detail?: string;
}) {
  return (
    <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span
        className={`eyebrow prov-label prov-label-${origin}`}
        style={{ color: ORIGIN_COLOUR[origin] }}
      >
        {children ?? ORIGIN_TEXT[origin]}
      </span>
      {detail && <span className="text-[11px] text-[var(--text-3)]">{detail}</span>}
    </div>
  );
}

export function Panel({
  title,
  meta,
  actions,
  children,
  id,
}: {
  title?: ReactNode;
  meta?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <section className="panel" id={id}>
      {(title || actions) && (
        <header className="panel-head">
          {title && <h2 className="eyebrow !text-[var(--text-2)]">{title}</h2>}
          {meta && <span className="text-[11px] text-[var(--text-3)]">{meta}</span>}
          {actions && <div className="ml-auto flex items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Badge({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "measured" | "model" | "cited" | "warn" | "bad" | "ok";
}) {
  const colour =
    tone === "neutral" ? "var(--text-3)" : `var(--${tone === "ok" ? "ok" : tone})`;
  return (
    <span
      className="badge"
      style={{
        color: colour,
        borderColor: `color-mix(in oklab, ${colour} 45%, transparent)`,
        background: `color-mix(in oklab, ${colour} 12%, transparent)`,
      }}
    >
      {children}
    </span>
  );
}

/**
 * Risk band as a four-step ladder.
 *
 * The number of filled steps carries the band, so it reads without colour —
 * which matters because risk is the one value an analyst scans for.
 */
export function RiskLadder({ score }: { score: number }) {
  const step = score >= 0.99 ? 4 : score >= 0.95 ? 3 : score >= 0.8 ? 2 : 1;
  const colour = `var(--risk-${step})`;
  const label = ["low", "elevated", "high", "severe"][step - 1];
  return (
    <span className="ladder" role="img" aria-label={`Risk band ${label}`}>
      {[1, 2, 3, 4].map((n) => (
        <span key={n} style={n <= step ? { background: colour } : undefined} />
      ))}
    </span>
  );
}

export function Meter({
  value,
  colour = "var(--measured)",
  width = 64,
}: {
  value: number;
  colour?: string;
  width?: number;
}) {
  return (
    <span className="meter inline-block align-middle" style={{ width }}>
      <i style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: colour }} />
    </span>
  );
}

/** A labelled figure. `hint` explains where the number comes from. */
export function Stat({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: string;
}) {
  return (
    <div className="border-t border-[var(--line)] pt-2">
      <dt className="eyebrow">{label}</dt>
      <dd
        className="num mt-1 text-[1.375rem] leading-none"
        style={tone ? { color: tone } : undefined}
      >
        {value}
      </dd>
      {hint && <p className="mt-1.5 text-[11px] text-[var(--text-3)]">{hint}</p>}
    </div>
  );
}

/** Empty, error and loading states share one shape so they read consistently. */
export function Note({
  kind = "empty",
  title,
  children,
  action,
}: {
  kind?: "empty" | "error";
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  const colour = kind === "error" ? "var(--bad)" : "var(--text-3)";
  return (
    <div
      className="border border-dashed p-4"
      style={{ borderColor: `color-mix(in oklab, ${colour} 40%, var(--line))` }}
      role={kind === "error" ? "alert" : undefined}
    >
      <p className="eyebrow" style={{ color: colour }}>
        {title}
      </p>
      {children && (
        <div className="mt-1.5 max-w-[62ch] text-[var(--text-2)]">{children}</div>
      )}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export function Skeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2" aria-hidden>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton h-6" style={{ opacity: 1 - i * 0.12 }} />
      ))}
    </div>
  );
}

/** A horizontal proportion bar. Used for distributions with real counts. */
export function Distribution({
  items,
  emptyLabel = "Nothing recorded yet",
}: {
  items: { label: string; count: number; colour?: string }[];
  emptyLabel?: string;
}) {
  const total = items.reduce((sum, item) => sum + item.count, 0);
  if (!total) {
    return <p className="text-[var(--text-3)]">{emptyLabel}</p>;
  }
  return (
    <div>
      <div className="flex h-2 w-full overflow-hidden rounded-[1px]">
        {items.map((item) =>
          item.count ? (
            <span
              key={item.label}
              style={{
                width: `${(item.count / total) * 100}%`,
                background: item.colour ?? "var(--line-2)",
              }}
              title={`${item.label}: ${item.count}`}
            />
          ) : null,
        )}
      </div>
      <dl className="mt-3 space-y-1.5">
        {items.map((item) => (
          <div key={item.label} className="flex items-baseline gap-2 text-[12px]">
            <span
              className="mt-[3px] size-2 shrink-0 rounded-[1px]"
              style={{ background: item.colour ?? "var(--line-2)" }}
              aria-hidden
            />
            <dt className="text-[var(--text-2)]">{item.label}</dt>
            <dd className="num ml-auto text-[var(--text)]">{item.count}</dd>
            <dd className="num w-11 text-right text-[var(--text-3)]">
              {total ? `${Math.round((item.count / total) * 100)}%` : "—"}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
