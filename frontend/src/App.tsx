import { useQuery } from "@tanstack/react-query";
import { getHealth, type DependencyStatus } from "./api/client";

const DEPENDENCY_LABELS: Record<string, string> = {
  api: "API",
  postgres: "PostgreSQL",
  pgvector: "pgvector",
  redis: "Redis",
};

function StatusDot({ status }: { status: DependencyStatus["status"] }) {
  return (
    <span
      className={`inline-block size-2.5 rounded-full ${
        status === "ok" ? "bg-emerald-400" : "bg-rose-500"
      }`}
      aria-hidden
    />
  );
}

function DependencyRow({ name, dep }: { name: string; dep: DependencyStatus }) {
  return (
    <li className="flex items-center justify-between gap-4 border-b border-white/5 px-4 py-3 last:border-0">
      <span className="flex items-center gap-3">
        <StatusDot status={dep.status} />
        <span className="text-sm font-medium text-zinc-100">
          {DEPENDENCY_LABELS[name] ?? name}
        </span>
      </span>
      <span className="text-right font-mono text-xs text-zinc-400">
        {dep.detail ?? dep.status}
      </span>
    </li>
  );
}

export default function App() {
  const { data, isPending, error } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 10_000,
  });

  return (
    <main className="min-h-full bg-zinc-950 px-6 py-16 text-zinc-100">
      <div className="mx-auto w-full max-w-lg">
        <h1 className="text-2xl font-semibold tracking-tight">Argus</h1>
        <p className="mt-1 text-sm text-zinc-400">
          Transaction risk assessment and evidence-grounded investigation.
        </p>

        <section className="mt-8 overflow-hidden rounded-xl border border-white/10 bg-zinc-900/60">
          <header className="flex items-center justify-between border-b border-white/10 px-4 py-3">
            <h2 className="text-sm font-semibold">System health</h2>
            {data && (
              <span className="font-mono text-xs text-zinc-500">
                v{data.version} &middot; {data.environment}
              </span>
            )}
          </header>

          {isPending && <p className="px-4 py-6 text-sm text-zinc-400">Checking dependencies…</p>}

          {error && (
            <p className="px-4 py-6 text-sm text-rose-400">
              Cannot reach the API. Is <code className="font-mono">docker compose up</code> running?
            </p>
          )}

          {data && (
            <ul>
              {Object.entries(data.dependencies).map(([name, dep]) => (
                <DependencyRow key={name} name={name} dep={dep} />
              ))}
            </ul>
          )}
        </section>

        {data && (
          <p className="mt-4 text-xs text-zinc-500">
            Overall status:{" "}
            <span className={data.status === "ok" ? "text-emerald-400" : "text-amber-400"}>
              {data.status}
            </span>
          </p>
        )}
      </div>
    </main>
  );
}
