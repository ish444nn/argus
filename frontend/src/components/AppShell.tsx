import { useQuery } from "@tanstack/react-query";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth, getOverview } from "../api/client";
import { DEFAULT_BUDGET } from "../budget";

/**
 * The shell.
 *
 * A narrow persistent rail rather than a top bar: an analyst tool is used for
 * a whole shift, and vertical space is what the tables need. The rail also
 * keeps the queue one click away from anywhere, which is the movement that
 * happens most.
 *
 * The footer of the rail carries system state — provider and dependency
 * health — because "is the investigation layer actually running" is a
 * question the analyst needs answered without asking.
 */

function Mark() {
  /**
   * Eight watchers, one subject.
   *
   * A square frame — a case, a file, a screen — holding a 3×3 lattice. The
   * centre cell is the transaction under review and carries the only colour on
   * the mark; the eight around it are the watchers, deliberately muted. It is
   * the Argus story told in the one geometric form that still resolves at 22px,
   * and it doubles as a node-and-neighbourhood glyph, which is what the product
   * actually does.
   */
  const ring = [
    [6, 6],
    [12, 6],
    [18, 6],
    [6, 12],
    [18, 12],
    [6, 18],
    [12, 18],
    [18, 18],
  ];

  return (
    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
      <rect
        x="1.5"
        y="1.5"
        width="21"
        height="21"
        fill="none"
        stroke="var(--line-2)"
        strokeWidth="1.25"
      />
      {ring.map(([cx, cy]) => (
        <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="1.35" fill="var(--text-3)" />
      ))}
      <circle cx="12" cy="12" r="2.6" fill="var(--measured)" />
    </svg>
  );
}

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/queue", label: "Queue", end: false },
];

function SystemState() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });
  const overview = useQuery({
    // Same key the Overview route uses at its default budget, so the shell
    // shares that cache entry instead of issuing a second request.
    queryKey: ["overview", DEFAULT_BUDGET],
    queryFn: () => getOverview(DEFAULT_BUDGET),
    refetchInterval: 30_000,
  });

  const down = Object.entries(health.data?.dependencies ?? {})
    .filter(([, dep]) => dep.status === "error")
    .map(([name]) => name);
  const ok = health.data?.status === "ok";
  const provider = overview.data?.llm_provider;

  return (
    <div className="space-y-2 px-3 py-3">
      <div className="flex items-center gap-2" title={ok ? "All dependencies reachable" : `Unavailable: ${down.join(", ")}`}>
        <span
          className="size-1.5 shrink-0 rounded-full"
          style={{ background: ok ? "var(--ok)" : "var(--bad)" }}
          aria-hidden
        />
        <span className="eyebrow truncate">
          {health.isPending ? "checking" : ok ? "systems ok" : down.join(", ")}
        </span>
      </div>
      {provider && (
        <p className="text-[10px] leading-tight text-[var(--text-3)]">
          Narratives:{" "}
          <span style={{ color: provider === "gemini" ? "var(--model)" : "var(--text-2)" }}>
            {provider === "gemini" ? "Gemini" : "rule-built"}
          </span>
        </p>
      )}
    </div>
  );
}

export function AppShell() {
  return (
    <div className="flex h-full">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-50 focus:bg-[var(--surface-3)] focus:px-3 focus:py-2"
      >
        Skip to content
      </a>

      <nav
        aria-label="Sections"
        className="flex w-[170px] shrink-0 flex-col border-r border-[var(--line)] bg-[var(--surface)]"
      >
        <div className="flex items-center gap-2.5 border-b border-[var(--line)] px-3 py-3.5">
          <Mark />
          <div className="leading-none">
            <p className="font-cond text-[15px] font-semibold tracking-[0.14em]">ARGUS</p>
            <p className="mt-1 text-[10px] tracking-wide text-[var(--text-3)]">
              AML investigation
            </p>
          </div>
        </div>

        <ul className="flex-1 py-2">
          {NAV.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `block border-l-[3px] px-3 py-2 font-cond text-[12px] font-semibold uppercase tracking-[0.08em] transition-colors ${
                    isActive
                      ? "border-[var(--measured)] bg-[var(--surface-2)] text-[var(--text)]"
                      : "border-transparent text-[var(--text-3)] hover:bg-[var(--surface-2)] hover:text-[var(--text-2)]"
                  }`
                }
              >
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="border-t border-[var(--line)]">
          <SystemState />
        </div>
      </nav>

      <main id="main" className="min-w-0 flex-1 overflow-y-auto">
        <Outlet />
      </main>
    </div>
  );
}
