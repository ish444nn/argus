import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { getHealth } from "./api/client";
import { CaseDetail } from "./components/CaseDetail";
import { Queue } from "./components/Queue";

function HealthPill() {
  const { data } = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    refetchInterval: 30_000,
  });

  if (!data) return null;
  const ok = data.status === "ok";
  const broken = Object.entries(data.dependencies)
    .filter(([, dep]) => dep.status === "error")
    .map(([name]) => name);

  return (
    <span
      title={ok ? "all dependencies healthy" : `unavailable: ${broken.join(", ")}`}
      className="flex items-center gap-1.5 text-xs text-zinc-500"
    >
      <span
        className={`inline-block size-1.5 rounded-full ${
          ok ? "bg-emerald-400" : "bg-rose-500"
        }`}
      />
      {ok ? "healthy" : broken.join(", ")}
      <span className="text-zinc-700">·</span>v{data.version}
    </span>
  );
}

export default function App() {
  const [caseId, setCaseId] = useState<number | null>(null);
  const [timestep, setTimestep] = useState<number | undefined>(undefined);

  return (
    <div className="flex h-full flex-col bg-zinc-950 text-zinc-100">
      <header className="flex items-center gap-3 border-b border-white/10 px-5 py-3">
        <h1 className="text-sm font-semibold tracking-tight">Argus</h1>
        <p className="hidden text-xs text-zinc-500 sm:block">
          Transaction risk assessment and evidence-grounded investigation
        </p>
        <div className="ml-auto">
          <HealthPill />
        </div>
      </header>

      <main className="grid flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[1fr_28rem]">
        <Queue
          selectedCaseId={caseId}
          onSelect={setCaseId}
          timestep={timestep}
          onTimestepChange={setTimestep}
        />
        {caseId !== null ? (
          <CaseDetail caseId={caseId} onClose={() => setCaseId(null)} />
        ) : (
          <aside className="hidden items-center justify-center border-l border-white/10 p-8 text-center text-sm text-zinc-600 lg:flex">
            Select a case to see its evidence.
          </aside>
        )}
      </main>
    </div>
  );
}
