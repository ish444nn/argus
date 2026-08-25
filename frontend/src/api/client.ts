/**
 * Thin typed fetch wrapper.
 *
 * Types are hand-written for now. From Phase 5 they are generated from the
 * API's OpenAPI schema so the client cannot drift from the server.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export type DependencyStatus = {
  status: "ok" | "error";
  detail: string | null;
};

export type HealthResponse = {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  dependencies: Record<string, DependencyStatus>;
};

export type QueueEntry = {
  case_id: number;
  tx_id: number;
  timestep: number;
  risk_score: number;
  queue_rank: number | null;
  graph_score: number | null;
  status: string;
  queue_tier: string | null;
  confidence: number | null;
  evidence_count: number;
  latest_decision: string | null;
  created_at: string;
};

export type QueuePage = {
  total: number;
  limit: number;
  offset: number;
  items: QueueEntry[];
};

export type EvidenceItem = {
  id: number;
  kind: string;
  summary: string;
  strength: number;
  weight: number;
  /** strength x weight — this item's share of the deterministic confidence. */
  contribution: number;
  neighbour_tx_id: number | null;
  neighbour_timestep: number | null;
  typology_reference_id: number | null;
  details: Record<string, unknown> | null;
  created_at: string;
};

export type Neighbourhood = {
  tx_id: number;
  timestep: number;
  in_degree: number;
  out_degree: number;
  total_degree: number;
  neighbour_count: number;
  same_batch_neighbours: number;
  flagged_neighbours: number;
  neighbour_mean_risk: number | null;
  chain_length: number;
};

export type CitedSource = {
  evidence_id: number;
  reference_id: number;
  typology_id: string;
  title: string;
  publisher: string;
  source_url: string;
  document: string | null;
  year: number | null;
  section_heading: string;
  text: string;
  patterns: string[];
  similarity: number;
  retrieved_for: string[];
};

export type InvestigationMeta = {
  provider?: string;
  model?: string;
  used_fallback?: boolean;
  attempts?: number;
  retrieval_patterns?: string[];
  retrieved_sources?: string[];
  validation_errors?: string[];
  confidence_contributions?: Record<string, number>;
};

export type CaseDetail = {
  case_id: number;
  tx_id: number;
  timestep: number;
  label: string;
  risk_score: number;
  model_version: string;
  queue_rank: number | null;
  graph_score: number | null;
  status: string;
  queue_tier: string | null;
  confidence: number | null;
  narrative: string | null;
  narrative_source: string | null;
  typology_assessment: string | null;
  recommended_action: string | null;
  investigation_meta: InvestigationMeta | null;
  batch_run_id: number | null;
  alert_budget: number | null;
  created_at: string;
  neighbourhood: Neighbourhood;
  evidence: EvidenceItem[];
};

export type BatchRun = {
  batch_run_id: number;
  timestep: number;
  status: string;
  model_version: string | null;
  alert_budget: number | null;
  scored_count: number;
  queued_count: number;
  investigated_count: number;
  failed_count: number;
  cases: number;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = ((await response.json()) as { detail?: string }).detail ?? detail;
    } catch {
      // Non-JSON error body; the status text is the best we have.
    }
    throw new Error(`${path} failed: ${detail}`);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getQueue(params: {
  timestep?: number;
  limit?: number;
  offset?: number;
  sortBy?: string;
  descending?: boolean;
}): Promise<QueuePage> {
  const search = new URLSearchParams();
  if (params.timestep !== undefined) search.set("timestep", String(params.timestep));
  search.set("limit", String(params.limit ?? 50));
  search.set("offset", String(params.offset ?? 0));
  search.set("sort_by", params.sortBy ?? "risk_score");
  search.set("descending", String(params.descending ?? true));
  return request<QueuePage>(`/api/queue?${search}`);
}

export function getCase(caseId: number): Promise<CaseDetail> {
  return request<CaseDetail>(`/api/cases/${caseId}`);
}

export function getCitedSources(caseId: number): Promise<CitedSource[]> {
  return request<CitedSource[]>(`/api/cases/${caseId}/sources`);
}

export function startInvestigation(
  caseId: number,
): Promise<{ task_id: string; provider: string }> {
  return request<{ task_id: string; provider: string }>(
    `/api/cases/${caseId}/investigate`,
    { method: "POST" },
  );
}

export function getBatches(): Promise<BatchRun[]> {
  return request<BatchRun[]>("/api/batches");
}

export function startReplay(timestep: number): Promise<{ task_id: string }> {
  return request<{ task_id: string }>(`/api/batches/${timestep}/replay`, {
    method: "POST",
  });
}
