/**
 * Typed fetch wrappers for every backend endpoint.
 * All calls go through the Vite proxy at /api → http://localhost:8000/api.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface SessionCreateParams {
  participant_id: string;
  session_number?: number;
  mode: "dev" | "behavioral" | "scanner";
  config_index: number;
}

export interface SessionOut {
  session_id: number;
  participant_id: string;
  mode: string;
  config_index: number;
}

export interface SessionDetail extends SessionOut {
  session_number: number;
  created_at: string;
  anchor_t_ms: number | null;
}

export interface TrialData {
  block: number;
  agent: string;
  trial_in_block: number;
  trial_global: number;
  participant_choice: number | null;
  agent_choice: number;
  outcome: string;
  points_delta: number;
  points_cumulative: number;
  rt_ms: number | null;
  onset_ms: number;
  iti_duration_ms: number;
  block_onset_ms: number;
  condition: string;
}

export interface BlockConfig {
  avatar_id: string;
  condition: string;
}

export interface RotationConfig {
  set: number;
  blocks: BlockConfig[];
}

export interface TrialOut extends TrialData {
  id: number;
  session_id: number;
}

export interface TriggerData {
  tr_number: number;
  t_ms: number;
}

export interface TriggerOut extends TriggerData {
  id: number;
  session_id: number;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${url} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function patch<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PATCH ${url} → ${res.status}`);
  return res.json() as Promise<T>;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} → ${res.status}`);
  return res.json() as Promise<T>;
}

// ── API functions ────────────────────────────────────────────────────────────

export async function createSession(
  params: SessionCreateParams,
): Promise<SessionOut> {
  return post<SessionOut>("/api/sessions", params);
}

export async function getSession(sessionId: number): Promise<SessionDetail> {
  return get<SessionDetail>(`/api/sessions/${sessionId}`);
}

export async function setAnchor(
  sessionId: number,
  tMs: number,
): Promise<SessionOut> {
  return patch<SessionOut>(`/api/sessions/${sessionId}/anchor`, {
    anchor_t_ms: tMs,
  });
}

export async function postTrial(
  sessionId: number,
  trial: TrialData,
): Promise<TrialOut> {
  return post<TrialOut>(`/api/sessions/${sessionId}/trials`, trial);
}

export async function getRotation(configIndex: number): Promise<RotationConfig> {
  return get<RotationConfig>(`/api/rotations/${configIndex}`);
}

export async function postTrigger(
  sessionId: number,
  trigger: TriggerData,
): Promise<TriggerOut> {
  return post<TriggerOut>(`/api/sessions/${sessionId}/triggers`, trigger);
}
