/**
 * Typed fetch wrappers for every backend endpoint.
 * All calls go through the Vite proxy at /api → http://localhost:8000/api.
 */

// ── Types ────────────────────────────────────────────────────────────────────

export interface SessionCreateParams {
  participant_id: string;
  session_number?: number;
  mode: "dev" | "behavioral" | "scanner";
  config_index?: number;  // optional; block order is now generated server-side from session_id
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
  feedback_onset_ms: number;
  iti_duration_ms: number;
  block_onset_ms: number;
  condition: string;
  level: number | null;
  is_noisy: boolean;
  noise_trigger: string | null;
  success_rate: number | null;
  noise_breaker: boolean;
  bot_attr_r: number;
  bot_attr_p: number;
  bot_attr_s: number;
  p_attr_r: number;
  p_attr_p: number;
  p_attr_s: number;
}

export interface BlockConfig {
  avatar_id: string;
  condition: string;
  level: number;
}

export interface SessionRotation {
  blocks: BlockConfig[];
}

export interface RegistrationOut {
  participant_id: string;
  friendly_avatar_id: string;
  neutral_avatar_id: string;
  registered_at: string;
}

export interface TrialOut extends TrialData {
  id: number;
  session_id: number;
}

/** @deprecated Use getSessionRotation instead */
export async function getRotation(_configIndex: number): Promise<never> {
  throw new Error("getRotation is deprecated — use getSessionRotation(sessionId)");
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

async function put<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`PUT ${url} → ${res.status}`);
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

export async function getSessionRotation(sessionId: number): Promise<SessionRotation> {
  return get<SessionRotation>(`/api/sessions/${sessionId}/rotation`);
}

export async function getRegistration(participantId: string): Promise<RegistrationOut> {
  return get<RegistrationOut>(`/api/participants/${encodeURIComponent(participantId)}/registration`);
}

export async function upsertRegistration(
  participantId: string,
  friendlyAvatarId: string,
  neutralAvatarId: string,
): Promise<RegistrationOut> {
  return put<RegistrationOut>(
    `/api/participants/${encodeURIComponent(participantId)}/registration`,
    { friendly_avatar_id: friendlyAvatarId, neutral_avatar_id: neutralAvatarId },
  );
}

export async function postTrigger(
  sessionId: number,
  trigger: TriggerData,
): Promise<TriggerOut> {
  return post<TriggerOut>(`/api/sessions/${sessionId}/triggers`, trigger);
}
