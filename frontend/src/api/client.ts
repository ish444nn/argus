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

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
