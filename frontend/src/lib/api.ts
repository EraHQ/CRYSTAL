// Tiny fetch wrapper. Single source of truth for "how do we talk to the
// admin API and the chat-completions endpoint."
//
// v2 reconciliation (frontend port, 2026-06): v2 reorganized the backend
// into endpoints/. A few admin routes changed shape or path vs v1; this
// file absorbs every one of those differences so the pages and types.ts
// stay unchanged. Each affected method fetches the raw v2 payload as `any`
// and returns the v1-shaped object the UI expects.
import type {
  AdminKeyResponse,
  AgentEventsResponse,
  AgentGapsResponse,
  AgentRunResponse,
  AgentTasksResponse,
  ChatCompletionResponse,
  ChatMessage,
  CreateCustomerRequest,
  CreateCustomerResponse,
  CrystalDetail,
  CrystalsListResponse,
  CustomersListResponse,
  QueryLogsListResponse,
  SessionCommandsResponse,
  SessionDependenciesResponse,
  SessionsListResponse,
} from "./types";

// Hosted-auth token provider (Accounts Phase C, 2026-07-06). When the
// AuthProvider has a Firebase session it registers a getter here; every
// request that doesn't carry its own Authorization gains the JWT. Self-
// host (no Firebase config) never registers one — zero behavior change.
let authTokenProvider: (() => Promise<string | null>) | null = null;
export function setAuthTokenProvider(
  fn: (() => Promise<string | null>) | null
) {
  authTokenProvider = fn;
}

class ApiError extends Error {
  constructor(
    public status: number,
    public statusText: string,
    public body: unknown,
    message?: string
  ) {
    super(message ?? `${status} ${statusText}`);
  }
}

async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  // Attach the hosted-session JWT unless the caller set its own bearer
  // (the playground's Key-A calls keep their key).
  if (!headers["Authorization"] && !headers["authorization"] && authTokenProvider) {
    try {
      const token = await authTokenProvider();
      if (token) headers["Authorization"] = `Bearer ${token}`;
    } catch {
      // A token fetch hiccup must not break unauthenticated routes.
    }
  }
  const res = await fetch(url, { ...init, headers });
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // Some 204s, etc. Ignore parse failures.
  }
  if (!res.ok) {
    throw new ApiError(res.status, res.statusText, body);
  }
  return body as T;
}

function qs(params: Record<string, string | number | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined) sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}

// K1 (2026-07-08): raw fetch with the console JWT for non-JSON bodies
// (multipart upload) — mirrors jsonFetch's auth without its Content-Type.
export async function authedFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    ...((init?.headers as Record<string, string>) ?? {}),
  };
  if (!headers["Authorization"] && authTokenProvider) {
    try {
      const token = await authTokenProvider();
      if (token) headers["Authorization"] = `Bearer ${token}`;
    } catch { /* unauthenticated routes keep working */ }
  }
  return fetch(url, { ...init, headers });
}

export const api = {
  // ── Hosted identity (Accounts Phase C) ─────────────────────
  me: () => jsonFetch<{
    kind: string; role: string; customer_id: string | null;
    user_id: string | null; email: string | null;
  }>("/v1/me"),

  signup: (body: {
    industry?: string; building?: string; experience?: string; model?: string;
  }) =>
    jsonFetch<{
      created: boolean; user_id: string; email: string; role: string;
      customer_id: string | null; api_key: string | null;
    }>("/v1/auth/signup", { method: "POST", body: JSON.stringify(body) }),

  updateOnboarding: (body: {
    industry?: string; building?: string; experience?: string;
  }) =>
    jsonFetch<any>("/v1/me/onboarding", {
      method: "POST", body: JSON.stringify(body),
    }),

  customerSpend: (customerId: string) =>
    jsonFetch<{
      customer_id: string; inference_mode: string;
      subscription_tier: string | null; totals: any;
      managed_month_to_date_micro_usd: number;
      managed_monthly_cap_micro_usd: number;
    }>(`/admin/api/customers/${encodeURIComponent(customerId)}/spend`),

  getCustomer: (customerId: string) =>
    jsonFetch<any>(`/v1/customers/${encodeURIComponent(customerId)}`),

  setModel: (customerId: string, modelId: string) =>
    jsonFetch<any>(
      `/v1/customers/${encodeURIComponent(customerId)}/model`,
      { method: "PATCH", body: JSON.stringify({ model_id: modelId }) }
    ),

  // S12: spend by ledger origin (console view).
  spendOrigins: (customerId: string, days?: number) =>
    jsonFetch<{ origins: any[] }>(
      `/v1/customers/${encodeURIComponent(customerId)}/spend/origins${qs({ days })}`
    ),

  listBudgets: (customerId: string) =>
    jsonFetch<{ budgets: any[] }>(
      `/v1/customers/${encodeURIComponent(customerId)}/budgets`
    ),

  upsertBudget: (customerId: string, fn: string, capMicroUsd: number) =>
    jsonFetch<any>(
      `/v1/customers/${encodeURIComponent(customerId)}/budgets/${encodeURIComponent(fn)}`,
      { method: "PUT", body: JSON.stringify({ cap_micro_usd: capMicroUsd }) }
    ),

  promoteGapToResearch: (gapId: string) =>
    jsonFetch<{ task_id: string; gap_id: string }>(
      `/v1/gaps/${encodeURIComponent(gapId)}/research`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  rotateApiKey: (customerId: string) =>
    jsonFetch<{ customer_id: string; api_key: string }>(
      `/v1/customers/${encodeURIComponent(customerId)}/api_key`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  setInferenceMode: (customerId: string, mode: "managed" | "byok") =>
    jsonFetch<any>(
      `/v1/customers/${encodeURIComponent(customerId)}/inference_mode`,
      { method: "PATCH", body: JSON.stringify({ inference_mode: mode }) }
    ),

  // v2: GET /admin/api/customers -> { customers, count }. Map to { items }.
  listCustomers: async (): Promise<CustomersListResponse> => {
    const body = await jsonFetch<any>("/admin/api/customers");
    return { items: body.customers ?? [] };
  },

  // POST /v1/customers -> { id, api_key, provider, model_id }. Onboarding.
  // Key A (api_key) is returned ONCE here. No Bearer needed to create.
  createCustomer: (body: CreateCustomerRequest): Promise<CreateCustomerResponse> =>
    jsonFetch<CreateCustomerResponse>("/v1/customers", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // v2: GET /admin/api/customers/{id}/crystals?offset&limit -> { total, crystals }.
  // Map crystals -> items; default diagnostic_tags (v2 omits it and
  // BankBrowser maps over it, so a missing value would crash the row).
  listCrystals: async (
    customerId: string,
    opts: { offset?: number; limit?: number } = {}
  ): Promise<CrystalsListResponse> => {
    const body = await jsonFetch<any>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/crystals${qs({
        offset: opts.offset,
        limit: opts.limit,
      })}`
    );
    return {
      total: body.total ?? 0,
      items: (body.crystals ?? []).map((c: any) => ({
        ...c,
        diagnostic_tags: c.diagnostic_tags ?? [],
      })),
    };
  },

  // v2: GET /admin/api/crystals/{cid} -> { crystal, facts } (no customer
  // segment in the path). Flatten `crystal`; default the diagnostic fields
  // v2 doesn't return so CrystalDetail stays satisfied.
  getCrystal: async (
    _customerId: string,
    crystalId: string
  ): Promise<CrystalDetail> => {
    const body = await jsonFetch<any>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}`
    );
    const c = body.crystal ?? {};
    return {
      ...c,
      diagnostic_tags: c.diagnostic_tags ?? [],
      keyword_fingerprint: c.keyword_fingerprint ?? [],
      cluster_tightness: c.cluster_tightness ?? null,
      facts: body.facts ?? [],
    } as CrystalDetail;
  },

  // v2: GET /admin/api/customers/{id}/query_logs?offset&limit -> { total, items }
  // (added in the frontend-port backend pass). Default matched_facts since
  // the Query Log + Chat pages index it.
  listQueryLogs: async (
    customerId: string,
    opts: { offset?: number; limit?: number } = {}
  ): Promise<QueryLogsListResponse> => {
    const body = await jsonFetch<any>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/query_logs${qs({
        offset: opts.offset,
        limit: opts.limit,
      })}`
    );
    return {
      total: body.total ?? 0,
      items: (body.items ?? []).map((q: any) => ({
        ...q,
        matched_facts: q.matched_facts ?? [],
      })),
    };
  },

  // v2: GET /admin/api/customers/{id}/admin_key -> { api_key }
  // (added in the frontend-port backend pass).
  getAdminKey: (customerId: string): Promise<AdminKeyResponse> =>
    jsonFetch<AdminKeyResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/admin_key`
    ),

  // Keyless admin chat proxy: POST /admin/api/customers/{id}/chat. Runs the
  // full chat pipeline for the customer resolved by path id — no Bearer
  // (API keys are hashed and unretrievable). Replaces getAdminKey +
  // chatCompletion for the playground. Same OpenAI-shaped response.
  adminChat: (
    customerId: string,
    body: { model: string; messages: ChatMessage[]; max_tokens?: number }
  ) =>
    jsonFetch<ChatCompletionResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/chat`,
      { method: "POST", body: JSON.stringify(body) }
    ),

  // Keyless admin agent run: POST /admin/api/customers/{id}/agent. Drives
  // CRYS (the agent) on the message history — the Inspector Chat page's
  // replacement for adminChat. (adminChat / the proxy stays a dev-only API
  // tool.) Anthropic Messages-shaped request; returns CRYS's run result
  // (final text + full trajectory + tool-call log).
  adminAgent: (
    customerId: string,
    body: {
      messages: { role: "user" | "assistant"; content: string }[];
      system?: string;
      max_tokens?: number;
      metadata?: { sequence_id?: string };
    }
  ) =>
    jsonFetch<AgentRunResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/agent`,
      { method: "POST", body: JSON.stringify(body) }
    ),

  // Keyless admin learn: POST /admin/api/customers/{id}/learn. Powers the
  // playground's thumbs-up / thumbs-down without the customer's Bearer key.
  adminLearn: (
    customerId: string,
    body: {
      prompt: string;
      response: string;
      outcome: "pass" | "fail";
      signal?: string;
    }
  ) =>
    jsonFetch<{
      crystals_written: number;
      reflection?: string;
      knowledge?: string;
      cached?: boolean;
      error?: string;
    }>(`/admin/api/customers/${encodeURIComponent(customerId)}/learn`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Unchanged in v2 — path + shape match.
  chatCompletion: (
    apiKey: string,
    body: { model: string; messages: ChatMessage[]; max_tokens?: number }
  ) =>
    jsonFetch<ChatCompletionResponse>("/v1/chat/completions", {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    }),

  store: (
    apiKey: string,
    body: { key: string; value: string; crystal_type?: string; pair_type?: string }
  ) =>
    jsonFetch<{ crystal_id: string; fact_id: string; sparse_key: string }>(
      "/v1/store",
      {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}` },
        body: JSON.stringify(body),
      }
    ),

  learn: (
    apiKey: string,
    body: {
      prompt: string;
      response: string;
      outcome: "pass" | "fail";
      signal?: string;
    }
  ) =>
    jsonFetch<{
      crystals_written: number;
      reflection?: string;
      knowledge?: string;
      cached?: boolean;
      error?: string;
    }>("/v1/learn", {
      method: "POST",
      headers: { Authorization: `Bearer ${apiKey}` },
      body: JSON.stringify(body),
    }),

  getStats: (apiKey: string) =>
    jsonFetch<{
      crystal_count: number;
      fact_count: number;
      cache_hit_eligible: number;
    }>("/v1/stats", {
      headers: { Authorization: `Bearer ${apiKey}` },
    }),

  updateUpstreamKey: (customerId: string, apiKeyRef: string) =>
    jsonFetch<{ updated: boolean }>(
      `/v1/customers/${encodeURIComponent(customerId)}/upstream_key`,
      {
        method: "PATCH",
        body: JSON.stringify({ api_key_ref: apiKeyRef }),
      }
    ),

  // v2 cognition admin routes are flat with ?customer_id= (not customer-
  // nested), the review queue is "push-queue", and list shapes use `count`.
  // Map count -> total so the Cognition page reads them unchanged.
  listReviewQueue: async (customerId: string) => {
    const body = await jsonFetch<any>(
      `/admin/api/push-queue${qs({ customer_id: customerId })}`
    );
    return { total: body.count ?? 0, items: (body.items ?? []) as any[] };
  },

  approveReviewItem: (customerId: string, itemId: string) =>
    jsonFetch<{ approved: boolean; crystal_id?: string }>(
      `/admin/api/push-queue/${encodeURIComponent(itemId)}/approve${qs({
        customer_id: customerId,
      })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  rejectReviewItem: (customerId: string, itemId: string) =>
    jsonFetch<{ rejected: boolean }>(
      `/admin/api/push-queue/${encodeURIComponent(itemId)}/reject${qs({
        customer_id: customerId,
      })}`,
      { method: "POST" }
    ),

  // K1: Knowledge surface off the deprecated admin_key (410 since
  // no-plaintext) — /v1 routes now accept the console session when
  // ?customer_id= is present (require_customer_or_console).
  listDocuments: (customerId: string) =>
    jsonFetch<{ total: number; documents: any[] }>(
      `/v1/documents${qs({ customer_id: customerId })}`
    ),

  crystallizeDocument: (customerId: string, docId: string) =>
    jsonFetch<any>(
      `/v1/documents/${encodeURIComponent(docId)}/crystallize${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  approveDocument: (customerId: string, docId: string, body: any) =>
    jsonFetch<any>(
      `/v1/documents/${encodeURIComponent(docId)}/approve${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify(body ?? {}) }
    ),

  deleteDocument: (customerId: string, docId: string) =>
    jsonFetch<any>(
      `/v1/documents/${encodeURIComponent(docId)}${qs({ customer_id: customerId })}`,
      { method: "DELETE" }
    ),

  uploadDocumentFile: (customerId: string, form: FormData) =>
    authedFetch(`/v1/documents/upload${qs({ customer_id: customerId })}`, {
      method: "POST", body: form,
    }),

  listSubscriptions: (customerId: string) =>
    jsonFetch<any>(`/v1/subscriptions${qs({ customer_id: customerId })}`),

  subscribeCrystalType: (customerId: string, typeId: string) =>
    jsonFetch<any>(`/v1/subscribe${qs({ customer_id: customerId })}`, {
      method: "POST", body: JSON.stringify({ crystal_type: typeId }),
    }),

  unsubscribeCrystalType: (customerId: string, typeId: string) =>
    jsonFetch<any>(
      `/v1/subscribe/${encodeURIComponent(typeId)}${qs({ customer_id: customerId })}`,
      { method: "DELETE" }
    ),

  listChatSessions: (customerId: string) =>
    jsonFetch<{ sessions: any[]; count: number }>(
      `/admin/api/chat/sessions${qs({ customer_id: customerId })}`
    ),

  getChatSession: (customerId: string, sequenceId: string) =>
    jsonFetch<{ sequence_id: string; turns: any[]; count: number }>(
      `/admin/api/chat/sessions/${encodeURIComponent(sequenceId)}${qs({ customer_id: customerId })}`
    ),

  dismissSubstrateObservation: (itemId: string) =>
    jsonFetch<{ id: string; status: string }>(
      `/admin/api/metacognition/substrate-observations/${encodeURIComponent(itemId)}/dismiss`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  dismissAllSubstrateObservations: (customerId: string) =>
    jsonFetch<{ dropped: number }>(
      `/admin/api/metacognition/substrate-observations/dismiss-all${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  // S11 (2026-07-09): response-quality critique stream (super-admin).
  listQualityObservations: (customerId: string, criticRole?: string) =>
    jsonFetch<{ total: number; observations: any[] }>(
      `/admin/api/metacognition/quality-observations${qs({ customer_id: customerId, critic_role: criticRole })}`
    ),

  groupedQualityObservations: (customerId: string) =>
    jsonFetch<{ total_groups: number; groups: any[] }>(
      `/admin/api/metacognition/quality-observations/grouped${qs({ customer_id: customerId })}`
    ),

  listSubstrateObservations: (customerId: string) =>
    jsonFetch<{ total: number; observations: any[] }>(
      `/admin/api/metacognition/substrate-observations${qs({ customer_id: customerId })}`
    ),

  groupedSubstrateObservations: (customerId: string) =>
    jsonFetch<{ total_groups: number; groups: any[] }>(
      `/admin/api/metacognition/substrate-observations/grouped${qs({ customer_id: customerId })}`
    ),

  listKnowledgeGaps: async (customerId: string) => {
    const body = await jsonFetch<any>(
      `/admin/api/knowledge-gaps${qs({ customer_id: customerId })}`
    );
    return {
      total: body.count ?? 0,
      items: (body.gaps ?? []).map((g: any) => ({
        ...g,
        filled_content: g.filled_snippet ?? g.filled_content ?? null,
      })) as any[],
    };
  },

  listCognitionTasks: async (customerId: string) => {
    const body = await jsonFetch<any>(
      `/admin/api/cognition-tasks${qs({ customer_id: customerId })}`
    );
    return { total: body.count ?? 0, items: (body.tasks ?? []) as any[] };
  },

  // --- Never-Idle Convergence: knowledge conflicts + the unified backlog ---

  // GET /admin/api/conflicts?customer_id=&status= -> { conflicts, count }.
  // Open conflicts by default (two facts that can't both be true).
  listConflicts: async (customerId: string, status = "open") => {
    const body = await jsonFetch<any>(
      `/admin/api/conflicts${qs({ customer_id: customerId, status })}`
    );
    return { total: body.count ?? 0, items: (body.conflicts ?? []) as any[] };
  },

  // GET /admin/api/backlog?customer_id= -> { items, count }. One ranked view
  // over every waiting-work queue (gaps, conflicts, tasks, review, verify).
  listBacklog: async (customerId: string) => {
    const body = await jsonFetch<any>(
      `/admin/api/backlog${qs({ customer_id: customerId })}`
    );
    return { total: body.count ?? 0, items: (body.items ?? []) as any[] };
  },

  // POST /admin/api/conflicts/scan?customer_id= -> { scan }. On-demand
  // contradiction scan (surfacing-only). Synchronous; can take a moment.
  scanConflicts: (customerId: string) =>
    jsonFetch<{ scan: any }>(
      `/admin/api/conflicts/scan${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  // POST /admin/api/conflicts/{id}/resolve -> { conflict }. The curation
  // gate: superseded/blacklisted deactivate the losing fact (loser 'a'|'b');
  // qualified keeps both; dismissed is a no-op.
  resolveConflict: (
    conflictId: string,
    resolution: string,
    loser?: "a" | "b"
  ) =>
    jsonFetch<{ conflict: any }>(
      `/admin/api/conflicts/${encodeURIComponent(conflictId)}/resolve`,
      {
        method: "POST",
        body: JSON.stringify({ resolution, loser: loser ?? null }),
      }
    ),

  // v2: GET /admin/api/sessions?customer_id= -> { sessions, count }. The
  // Foundation F4 session registry — live agents + their state for the
  // Activity view. Keyless admin (same posture as the other reads above).
  listSessions: (customerId: string): Promise<SessionsListResponse> =>
    jsonFetch<SessionsListResponse>(
      `/admin/api/sessions${qs({ customer_id: customerId })}`
    ),

  // GET /admin/api/sessions/{id}/dependencies?customer_id= -> { dependencies, count }.
  getSessionDependencies: (
    customerId: string,
    sessionId: string
  ): Promise<SessionDependenciesResponse> =>
    jsonFetch<SessionDependenciesResponse>(
      `/admin/api/sessions/${encodeURIComponent(sessionId)}/dependencies${qs({
        customer_id: customerId,
      })}`
    ),

  // GET /admin/api/sessions/{id}/commands?customer_id= -> { commands, count }.
  // The G2 control-plane commands for a session (approval decisions, terminates).
  getSessionCommands: (
    customerId: string,
    sessionId: string
  ): Promise<SessionCommandsResponse> =>
    jsonFetch<SessionCommandsResponse>(
      `/admin/api/sessions/${encodeURIComponent(sessionId)}/commands${qs({
        customer_id: customerId,
      })}`
    ),

  // v2: GET /admin/api/agents/events?session_id=&customer_id=&after_seq= ->
  // { events, count }. The Unify-Agents per-session activity stream (turns,
  // tool calls, subagents, crystals, gaps) the Agents timeline renders.
  getAgentEvents: (
    customerId: string,
    sessionId: string,
    afterSeq?: number
  ): Promise<AgentEventsResponse> =>
    jsonFetch<AgentEventsResponse>(
      `/admin/api/agents/events${qs({
        customer_id: customerId,
        session_id: sessionId,
        after_seq: afterSeq,
      })}`
    ),

  // GET /admin/api/agents/tasks?customer_id= -> { tasks, count }. The daemon
  // background queue.
  listAgentTasks: (customerId: string): Promise<AgentTasksResponse> =>
    jsonFetch<AgentTasksResponse>(
      `/admin/api/agents/tasks${qs({ customer_id: customerId })}`
    ),

  // GET /admin/api/agents/gaps?customer_id= -> { gaps, count }. Agent-run
  // gaps (terminal background failures, retryable).
  listAgentGaps: (customerId: string): Promise<AgentGapsResponse> =>
    jsonFetch<AgentGapsResponse>(
      `/admin/api/agents/gaps${qs({ customer_id: customerId })}`
    ),
};

export { ApiError };
