// Tiny fetch wrapper. Single source of truth for "how do we talk to the
// admin API and the chat-completions endpoint."
//
// v2 reconciliation (frontend port, 2026-06): v2 reorganized the backend
// into endpoints/. A few admin routes changed shape or path vs v1; this
// file absorbs every one of those differences so the pages and types.ts
// stay unchanged. Each affected method fetches the raw v2 payload as `any`
// and returns the v1-shaped object the UI expects.
import type {
  AgentEventsResponse,
  AgentGapsResponse,
  AgentRunResponse,
  AgentTask,
  AgentTasksResponse,
  CreateScheduleRequest,
  ScheduleWriteResponse,
  SchedulesResponse,
  CreateCustomerRequest,
  CreateCustomerResponse,
  CrystalDetail,
  CrystalsListResponse,
  Assumption,
  BacklogItem,
  CognitionRun,
  CognitionRunDetail,
  CurationEvent,
  KnowledgeConflict,
  CognitionTask,
  CustomersListResponse,
  DailyCost,
  DocumentReview,
  KnowledgeGap,
  LedgerEntry,
  OperatorRow,
  SpendBudget,
  SchemaMapping,
  SchemaPreviewItem,
  SourceSchemaItem,
  Watch,
  WatchActivity,
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

// The humane-message extractor for failed responses — EXPORTED so the
// contract test (errorMessage.test.ts) exercises exactly what jsonFetch
// runs. LIVE-FOUND 2026-09-24: this API wraps HTTPException in the
// OpenAI error envelope {error:{message,...}} — there is NO top-level
// detail. Read BOTH shapes so the mapping works against this API and
// any plain-FastAPI deployment. The server side of this contract is
// pinned in tests/test_error_envelope_contract.py.
export function errorMessageFrom(body: unknown): string | undefined {
  const b = body as {
    detail?: unknown;
    error?: { message?: unknown };
  } | null;
  if (typeof b?.detail === "string") return b.detail;
  if (typeof b?.error?.message === "string") return b.error.message;
  return undefined;
}

export async function jsonFetch<T>(url: string, init?: RequestInit): Promise<T> {
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
    throw await apiErrorFrom(res, body);
  }
  return body as T;
}

// RC-16 (2026-10-08): ONE error path for every non-ok response, whatever
// made the request. Announces a plan wall (X-Plan-Wall) exactly as v110
// did, and for every other status surfaces the server's humane message
// (detail / error.message) as the ApiError message when there is one.
// Before this, only 402/403 got the message; a 400 or 500 rendered as
// "400 Bad Request" or raw JSON, and multipart uploads (authedFetch)
// got no handling at all, so their walls were silent.
async function apiErrorFrom(res: Response, body: unknown): Promise<ApiError> {
  announceWall(res, body);
  return new ApiError(res.status, res.statusText, body, errorMessageFrom(body) ?? undefined);
}

// The message a page should SHOW for any thrown error: the server's
// humane text when it sent one, else a short status line, never
// "[object Object]" and never raw JSON.
export function errorMessage(e: unknown, fallback = "Something went wrong."): string {
  if (e instanceof ApiError) {
    const detail = errorMessageFrom(e.body);
    if (detail) return detail;
    return `${fallback} (${e.status}${e.statusText ? " " + e.statusText : ""})`;
  }
  if (e instanceof Error && e.message) return e.message;
  return fallback;
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
//
// Console Refine PR 2 (docs/CONSOLE_UX_AUDIT.md 4.3 item 1, bug 5): a
// non-ok response goes through the SAME plan-wall dispatch as jsonFetch
// before the Response is handed back. Thirty callers read res.ok
// themselves, which is why this still returns the Response instead of
// throwing — but a 402 with X-Plan-Wall now opens the wall from every
// one of them (uploads, tier changes, deletes, fact edits, watches,
// review, export/import) instead of only from the JSON routes.
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
  const res = await fetch(url, { ...init, headers });
  if (!res.ok) {
    announceWall(res, await readErrorBody(res));
  }
  return res;
}

async function readErrorBody(res: Response): Promise<unknown> {
  try {
    return await res.clone().json();
  } catch {
    return null; // non-JSON error body: the status line is all we have
  }
}

// The message a page should SHOW for a non-ok Response it got back from
// authedFetch (Refine PR 2, 4.3 item 2): the server's humane text when
// it sent one, otherwise the caller's fallback with the status. Never
// raw JSON, never "status statusText" on its own.
export async function responseMessage(res: Response, fallback: string): Promise<string> {
  const detail = errorMessageFrom(await readErrorBody(res));
  return detail ?? `${fallback} (${res.status})`;
}

function announceWall(res: Response, body: unknown): void {
  dispatchWall(res.headers.get("X-Plan-Wall"), res.status, body);
}

function dispatchWall(wall: string | null, status: number, body: unknown): void {
  if (!wall) return;
  window.dispatchEvent(
    new CustomEvent("crystal:plan-wall", {
      detail: { code: wall, message: errorMessageFrom(body) ?? "", status },
    }),
  );
}

// Console Refine PR 5 (5.3 item 3): a multipart upload with progress.
// fetch() cannot report upload progress, so this is the one XHR in the
// console; it carries the same bearer, the same plan-wall dispatch and
// the same ApiError shape as jsonFetch, and `signal` aborts it.
export async function uploadWithProgress(
  url: string, form: FormData,
  opts: { onProgress?: (fraction: number) => void; signal?: AbortSignal } = {},
): Promise<unknown> {
  let token: string | null = null;
  if (authTokenProvider) {
    try { token = await authTokenProvider(); } catch { /* unauthenticated routes keep working */ }
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && opts.onProgress) opts.onProgress(e.total ? e.loaded / e.total : 0);
    };
    const parse = (): unknown => { try { return JSON.parse(xhr.responseText); } catch { return null; } };
    xhr.onload = () => {
      const body = parse();
      if (xhr.status >= 200 && xhr.status < 300) { resolve(body); return; }
      dispatchWall(xhr.getResponseHeader("X-Plan-Wall"), xhr.status, body);
      reject(new ApiError(xhr.status, xhr.statusText, body, errorMessageFrom(body) ?? undefined));
    };
    xhr.onerror = () => reject(new ApiError(0, "network", null, "The upload could not reach the server."));
    xhr.onabort = () => reject(new ApiError(0, "aborted", null, "Upload cancelled."));
    if (opts.signal) {
      if (opts.signal.aborted) { xhr.abort(); return; }
      opts.signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(form);
  });
}

// RC-16 (2026-10-08): the throwing variant for callers that want the
// SAME walls and humane messages as jsonFetch on a non-JSON request
// (multipart upload). authedFetch itself keeps returning the Response
// on error because thirty-two callers handle res.ok themselves.
export async function authedFetchOrThrow(url: string, init?: RequestInit): Promise<Response> {
  const res = await authedFetch(url, init);
  if (!res.ok) {
    // authedFetch already announced any wall; build the error only.
    const body = await readErrorBody(res);
    throw new ApiError(res.status, res.statusText, body, errorMessageFrom(body) ?? undefined);
  }
  return res;
}

export const api = {
  // ── Hosted identity (Accounts Phase C) ─────────────────────
  me: () => jsonFetch<{
    kind: string; role: string; customer_id: string | null;
    user_id: string | null; email: string | null;
    // L2-S4=B: the money state (hosted sessions only; null elsewhere).
    subscription_tier?: string | null; trial_expires_at?: string | null;
  }>("/v1/me"),

  // ── Billing (L2-S4=B, 2026-09-08) ────────────────────────
  billingCheckout: (
    successUrl: string,
    cancelUrl: string,
    plan: "starter" | "scale" = "starter",
  ) =>
    jsonFetch<{ checkout_url: string; session_id: string }>(
      "/v1/billing/checkout",
      {
        method: "POST",
        body: JSON.stringify({
          success_url: successUrl, cancel_url: cancelUrl, plan,
        }),
      },
    ),

  billingPortal: (returnUrl: string) =>
    jsonFetch<{ portal_url: string }>("/v1/billing/portal", {
      method: "POST",
      body: JSON.stringify({ return_url: returnUrl }),
    }),

  signup: (body: {
    industry?: string; building?: string; experience?: string; model?: string;
    // T2b: the wizard's capture.
    operator_name?: string; tools?: string[];
  }) =>
    jsonFetch<{
      created: boolean; user_id: string; email: string; role: string;
      customer_id: string | null; api_key: string | null;
      operator_id?: string;
    }>("/v1/auth/signup", { method: "POST", body: JSON.stringify(body) }),

  // T2b: the connect screen's polling target.
  onboardingStatus: () =>
    jsonFetch<{ connected: boolean; last_seen_at: string | null;
                first_recall_at?: string | null; ai_tools: string[] }>("/v1/onboarding/status"),

  // L2-S5b: the OAuth consent page's two calls.
  oauthClientInfo: (clientId: string) =>
    jsonFetch<{ client_id: string; client_name: string;
                redirect_hosts: string[]; grant?: string; scopes?: string[] }>(
      `/v1/oauth/client/${encodeURIComponent(clientId)}`),
  oauthApprove: (body: {
    client_id: string; redirect_uri: string; state: string;
    code_challenge: string; scopes: string; explicit_redirect: boolean;
    resource?: string;
  }) =>
    jsonFetch<{ redirect_to: string }>("/v1/oauth/approve", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // T2b: the About-you step — a real document through the extraction
  // pipeline (auto_crystallize births the first crystals immediately).
  createDocumentText: (customerId: string, body: {
    text: string; label: string; scope?: string; auto_crystallize?: boolean;
  }) =>
    jsonFetch<{ id: string }>(
      `/v1/documents?customer_id=${encodeURIComponent(customerId)}`,
      { method: "POST", body: JSON.stringify(body) },
    ),

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

  // Q87: PATCH /v1/customers/{id}/background_model { model_id } — the
  // background-curation pick; "" clears it (the platform tier map applies).
  setBackgroundModel: (customerId: string, modelId: string) =>
    jsonFetch<{ updated: boolean; customer_id: string; background_model_id: string | null }>(
      `/v1/customers/${encodeURIComponent(customerId)}/background_model`,
      { method: "PATCH", body: JSON.stringify({ model_id: modelId }) }
    ),

  // S12: spend by ledger origin (console view).
  spendOrigins: (customerId: string, days?: number) =>
    jsonFetch<{ origins: any[] }>(
      `/v1/customers/${encodeURIComponent(customerId)}/spend/origins${qs({ days })}`
    ),

  // {budgets: SpendBudget[], shadow_critic: {...}}; Settings reads both,
  // the Cognition "Background research" switch reads the auto_research row.
  listBudgets: (customerId: string) =>
    jsonFetch<{ budgets: SpendBudget[]; shadow_critic?: any }>(
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

  // Assumptions funnel F3 (Q5=A): tri-state explore toggle. true/false
  // are explicit; null reverts to the deployment default (explore ON).
  setAssumptionsExplore: (customerId: string, explore: boolean | null) =>
    jsonFetch<{ customer_id: string; assumptions_explore: boolean | null }>(
      `/v1/customers/${encodeURIComponent(customerId)}/assumptions_explore`,
      {
        method: "PATCH",
        body: JSON.stringify({ assumptions_explore: explore }),
      }
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

  // v2: GET /admin/api/customers/{id}/crystals?offset&limit&q&tier ->
  // { total, bank, crystals }. Console Refine PR 4 (5.1): `q` and `tier`
  // filter on the server, so a search reaches crystals past the loaded
  // page; `bank` is the whole-bank overview. Map crystals -> items;
  // default diagnostic_tags (v2 omits it and the page maps over it).
  listCrystals: async (
    customerId: string,
    opts: { offset?: number; limit?: number; q?: string; tier?: string; documentId?: string } = {}
  ): Promise<CrystalsListResponse> => {
    const body = await jsonFetch<any>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/crystals${qs({
        offset: opts.offset,
        limit: opts.limit,
        q: opts.q?.trim() ? opts.q.trim() : undefined,
        tier: opts.tier || undefined,
        document_id: opts.documentId || undefined,
      })}`
    );
    const bank = body.bank ?? {};
    return {
      total: body.total ?? 0,
      items: (body.crystals ?? []).map((c: any) => ({
        ...c,
        diagnostic_tags: c.diagnostic_tags ?? [],
      })),
      bank: {
        crystals: bank.crystals ?? body.total ?? 0,
        facts: bank.facts ?? 0,
        updated_at: bank.updated_at ?? null,
        tier_counts: bank.tier_counts ?? {},
      },
    };
  },

  // Console Refine PR 4 (5.1 items 6 and 7): the reader's writes, bound
  // here so the page has no raw fetches. All three are long-standing
  // admin routes (Gate D4a tier; fact supersede/retire with the ledger).
  setCrystalTier: (crystalId: string, tier: string) =>
    jsonFetch<{ crystal_id: string; tier: string }>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}/tier`,
      { method: "PATCH", body: JSON.stringify({ tier }) }
    ),
  supersedeFact: (crystalId: string, factId: string, text: string) =>
    jsonFetch<unknown>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}/facts/${encodeURIComponent(factId)}/supersede`,
      { method: "POST", body: JSON.stringify({ text }) }
    ),
  retireFact: (crystalId: string, factId: string) =>
    jsonFetch<{ retired: string }>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}/facts/${encodeURIComponent(factId)}/retire`,
      { method: "POST", body: JSON.stringify({}) }
    ),
  // GET /admin/api/crystals/{id}/ledger -> { ledger }: the append-only
  // History tab (before -> after with actor and time).
  getCrystalLedger: async (crystalId: string): Promise<LedgerEntry[]> => {
    const body = await jsonFetch<any>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}/ledger`
    );
    return (body.ledger ?? []) as LedgerEntry[];
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

  // Keyless admin agent run: POST /admin/api/customers/{id}/agent. Drives
  // CRYS (the agent) on the message history — the Inspector Chat page's
  // keyless entrypoint. Anthropic Messages-shaped request; returns CRYS's
  // run result (final text + full trajectory + tool-call log).
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

  // Block 2 slice 1 (2026-07-21): streaming agent run over the same
  // admin route with `stream: true` — the response is SSE with
  // agent-native named events (memory/src/crystal_cache/agent/events.py
  // is the vocabulary's one home). fetch + manual parse because
  // EventSource can't POST. `onEvent` fires per frame; resolves with the
  // terminal run_completed payload's `result` — the IDENTICAL dict
  // adminAgent returns — or throws (transport failure / `error` frame /
  // stream end without a terminal), letting the caller fall back to
  // adminAgent so the playground never loses a turn.
  adminAgentStream: async (
    customerId: string,
    body: {
      messages: { role: "user" | "assistant"; content: string }[];
      system?: string;
      max_tokens?: number;
      metadata?: { sequence_id?: string };
    },
    onEvent: (event: string, payload: any) => void
  ): Promise<AgentRunResponse> => {
    const res = await authedFetch(
      `/admin/api/customers/${encodeURIComponent(customerId)}/agent`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ ...body, stream: true }),
      }
    );
    if (!res.ok || !res.body) {
      let errBody: unknown = null;
      try { errBody = await res.json(); } catch { /* non-JSON error */ }
      throw new ApiError(res.status, res.statusText, errBody);
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let terminal: AgentRunResponse | null = null;
    let streamError: string | null = null;
    const handleFrame = (frame: string) => {
      let event = "message";
      const dataLines: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) return;
      let payload: any = null;
      try { payload = JSON.parse(dataLines.join("\n")); } catch { return; }
      onEvent(event, payload);
      if (event === "run_completed") terminal = payload.result as AgentRunResponse;
      else if (event === "error") streamError = payload.error || "agent stream error";
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        handleFrame(buf.slice(0, idx));
        buf = buf.slice(idx + 2);
      }
    }
    if (streamError) throw new Error(streamError);
    if (!terminal) throw new Error("agent stream ended without run_completed");
    return terminal;
  },

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

  uploadDocumentFile: (
    customerId: string, form: FormData,
    opts: { onProgress?: (fraction: number) => void; signal?: AbortSignal } = {},
  ) =>
    // RC-16: throws on a non-ok response (walls announce the modal; other
    // failures carry the server's message), like every JSON call.
    // Refine PR 5: reports upload progress and can be cancelled.
    uploadWithProgress(`/v1/documents/upload${qs({ customer_id: customerId })}`, form, opts),

  // GET /v1/documents/{id}/review -> the extracted items and chunks a
  // curator approves (Refine PR 5 review drawer).
  getDocumentReview: (customerId: string, docId: string) =>
    jsonFetch<DocumentReview>(
      `/v1/documents/${encodeURIComponent(docId)}/review${qs({ customer_id: customerId })}`
    ),

  // ── Watched repos (Gate M; Refine PR 5 bindings) ──────────
  listWatches: async (customerId: string): Promise<Watch[]> => {
    const body = await jsonFetch<{ watches: Watch[] }>(`/admin/api/watches${qs({ customer_id: customerId })}`);
    return body.watches ?? [];
  },
  createWatch: (customerId: string, body: {
    scheme: "git"; source_name: string; config: { repo: string; branch: string };
    review_mode: "auto" | "gated"; cadence_minutes: number; token?: string;
  }) =>
    jsonFetch<Watch>(`/admin/api/watches${qs({ customer_id: customerId })}`, {
      method: "POST", body: JSON.stringify(body),
    }),
  // PATCH: { action: "sync_now" } | { status: "active" | "paused" } |
  // { token } (Refine PR 5: replace a revoked credential in place).
  updateWatch: (customerId: string, watchId: string, body: { action?: "sync_now"; status?: "active" | "paused"; token?: string }) =>
    jsonFetch<Watch>(`/admin/api/watches/${encodeURIComponent(watchId)}${qs({ customer_id: customerId })}`, {
      method: "PATCH", body: JSON.stringify(body),
    }),
  deleteWatch: (customerId: string, watchId: string) =>
    jsonFetch<unknown>(`/admin/api/watches/${encodeURIComponent(watchId)}${qs({ customer_id: customerId })}`, {
      method: "DELETE",
    }),
  watchActivity: async (customerId: string, watchId: string): Promise<WatchActivity[]> => {
    const body = await jsonFetch<{ activity: WatchActivity[] }>(
      `/admin/api/watches/${encodeURIComponent(watchId)}/activity${qs({ customer_id: customerId })}`
    );
    return body.activity ?? [];
  },

  // ── Data shapes (Gate G3; Refine PR 5 bindings) ───────────
  listSourceSchemas: async (customerId: string): Promise<SourceSchemaItem[]> => {
    const body = await jsonFetch<{ schemas: SourceSchemaItem[] }>(`/admin/api/source-schemas${qs({ customer_id: customerId })}`);
    return body.schemas ?? [];
  },
  previewSourceSchema: (customerId: string, body: { mapping: SchemaMapping; sample: unknown[]; label: string }) =>
    jsonFetch<{ items: SchemaPreviewItem[] }>(`/admin/api/source-schemas/preview${qs({ customer_id: customerId })}`, {
      method: "POST", body: JSON.stringify(body),
    }),
  decideSourceSchema: (customerId: string, schemaId: string, action: "approve" | "reject") =>
    jsonFetch<unknown>(`/admin/api/source-schemas/${encodeURIComponent(schemaId)}/${action}${qs({ customer_id: customerId })}`, {
      method: "POST", body: JSON.stringify({}),
    }),
  saveSourceSchemaMapping: (customerId: string, schemaId: string, mapping: SchemaMapping) =>
    jsonFetch<unknown>(`/admin/api/source-schemas/${encodeURIComponent(schemaId)}/mapping${qs({ customer_id: customerId })}`, {
      method: "PUT", body: JSON.stringify({ mapping }),
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

  // Refine PR 13 (5.10): the Undo of dismiss — status back to deferred.
  restoreSubstrateObservation: (itemId: string) =>
    jsonFetch<{ id: string; status: string }>(
      `/admin/api/metacognition/substrate-observations/${encodeURIComponent(itemId)}/restore`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  dismissAllSubstrateObservations: (customerId: string) =>
    jsonFetch<{ dropped: number }>(
      `/admin/api/metacognition/substrate-observations/dismiss-all${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),

  // S11 (2026-07-09): response-quality critique stream (super-admin).
  groupedQualityObservations: (customerId: string) =>
    jsonFetch<{ total_groups: number; groups: any[] }>(
      `/admin/api/metacognition/quality-observations/grouped${qs({ customer_id: customerId })}`
    ),

  listSubstrateObservations: (customerId: string, limit = 200) =>
    jsonFetch<{ total: number; observations: any[] }>(
      `/admin/api/metacognition/substrate-observations${qs({ customer_id: customerId, limit })}`
    ),

  // Refine PR 13 (5.10): the flat quality stream, for "{n} flags this week".
  listQualityObservations: (customerId: string, since?: string, limit = 200) =>
    jsonFetch<{ total: number; observations: any[] }>(
      `/admin/api/metacognition/quality-observations${qs({ customer_id: customerId, since, limit })}`
    ),

  groupedSubstrateObservations: (customerId: string) =>
    jsonFetch<{ total_groups: number; groups: any[] }>(
      `/admin/api/metacognition/substrate-observations/grouped${qs({ customer_id: customerId })}`
    ),

  dailyCost: async (customerId: string): Promise<DailyCost> => {
    return jsonFetch<DailyCost>(`/admin/api/cost/daily${qs({ customer_id: customerId })}`);
  },

  // Refine PR 6 (bug 7.1): ONE shape for the gaps list. Cognition.tsx and
  // the tracker used the same query key with two shapes, so whichever
  // resolved first won and the Gaps rail was always empty.
  listKnowledgeGaps: async (customerId: string): Promise<{ total: number; items: KnowledgeGap[]; cycle_cap: number }> => {
    const body = await jsonFetch<any>(
      `/admin/api/knowledge-gaps${qs({ customer_id: customerId })}`
    );
    return {
      total: body.count ?? 0,
      cycle_cap: body.cycle_cap ?? 3,
      items: (body.gaps ?? []) as KnowledgeGap[],
    };
  },
  // POST /v1/gaps/{id}/answer | dismiss | reopen (Refine PR 6, 5.4 item 5).
  answerGap: (gapId: string, text: string) =>
    jsonFetch<{ gap_id: string; crystal_id: string; status: string }>(
      `/v1/gaps/${encodeURIComponent(gapId)}/answer`, { method: "POST", body: JSON.stringify({ text }) }),
  dismissGap: (gapId: string) =>
    jsonFetch<{ gap_id: string; status: string }>(`/v1/gaps/${encodeURIComponent(gapId)}/dismiss`, { method: "POST", body: "{}" }),
  reopenGap: (gapId: string) =>
    jsonFetch<{ gap_id: string; status: string }>(`/v1/gaps/${encodeURIComponent(gapId)}/reopen`, { method: "POST", body: "{}" }),

  listCognitionTasks: async (customerId: string, opts: { status?: string; limit?: number } = {}): Promise<{ total: number; items: CognitionTask[] }> => {
    const body = await jsonFetch<any>(
      `/admin/api/cognition-tasks${qs({ customer_id: customerId, status: opts.status, limit: opts.limit })}`
    );
    return { total: body.count ?? 0, items: (body.tasks ?? []) as CognitionTask[] };
  },

  // ── Cognition runs (cognition/api.py; Refine PR 6 bindings) ─────
  listCognitionRuns: async (customerId: string): Promise<CognitionRun[]> => {
    const body = await jsonFetch<{ total: number; environments: CognitionRun[] }>(
      `/admin/api/cognition/environments${qs({ customer_id: customerId })}`);
    return body.environments ?? [];
  },
  getCognitionRun: (runId: string) =>
    jsonFetch<CognitionRunDetail>(`/admin/api/cognition/environments/${encodeURIComponent(runId)}`),
  // The server owns liveness: a requeue on a LIVE run 409s "still alive".
  requeueCognitionTask: (taskId: string) =>
    jsonFetch<unknown>(`/admin/api/cognition/tasks/${encodeURIComponent(taskId)}/requeue`, { method: "POST", body: "{}" }),
  // A live run stops at its next step boundary (cancelled: "requested");
  // an orphan is finalized at once (cancelled: true).
  cancelCognitionTask: (taskId: string) =>
    jsonFetch<{ cancelled: boolean | "requested"; note?: string }>(
      `/admin/api/cognition/tasks/${encodeURIComponent(taskId)}/cancel`, { method: "POST", body: "{}" }),


  // --- Never-Idle Convergence: knowledge conflicts + the unified backlog ---

  // GET /admin/api/conflicts?customer_id=&status= -> { conflicts, count }.
  // Open conflicts by default (two facts that can't both be true).
  listConflicts: async (customerId: string, status = "open"): Promise<{ total: number; items: KnowledgeConflict[] }> => {
    const body = await jsonFetch<any>(
      `/admin/api/conflicts${qs({ customer_id: customerId, status })}`
    );
    return { total: body.count ?? 0, items: (body.conflicts ?? []) as KnowledgeConflict[] };
  },

  // GET /admin/api/backlog?customer_id= -> { items, count }. One ranked view
  // over every waiting-work queue (gaps, conflicts, tasks, review, verify).
  listBacklog: async (customerId: string): Promise<{ total: number; items: BacklogItem[] }> => {
    const body = await jsonFetch<any>(
      `/admin/api/backlog${qs({ customer_id: customerId })}`
    );
    return { total: body.count ?? 0, items: (body.items ?? []) as BacklogItem[] };
  },

  // POST /admin/api/conflicts/{id}/reopen -> { conflict }: the Undo of a
  // resolution (Refine PR 7); restores a deactivated loser and removes a
  // blacklist mark the conflict wrote.
  reopenConflict: (customerId: string, conflictId: string) =>
    jsonFetch<{ conflict: KnowledgeConflict }>(
      `/admin/api/conflicts/${encodeURIComponent(conflictId)}/reopen${qs({ customer_id: customerId })}`,
      { method: "POST", body: "{}" }
    ),

  // POST /admin/api/conflicts/{id}/explain -> { conflict } (Refine PR 7b):
  // one small model call writes why the claims contradict and a suggested
  // verb onto a conflict that has none. 409 for a duplicate, 502 when the
  // model's answer was unusable (nothing stored).
  explainConflict: (customerId: string, conflictId: string) =>
    jsonFetch<{ conflict: KnowledgeConflict }>(
      `/admin/api/conflicts/${encodeURIComponent(conflictId)}/explain${qs({ customer_id: customerId })}`,
      { method: "POST", body: "{}" }
    ),

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
    customerId: string,
    conflictId: string,
    resolution: string,
    loser?: "a" | "b"
  ) =>
    jsonFetch<{ conflict: any }>(
      `/admin/api/conflicts/${encodeURIComponent(conflictId)}/resolve${qs({ customer_id: customerId })}`,
      {
        method: "POST",
        body: JSON.stringify({ resolution, loser: loser ?? null }),
      }
    ),

  // Assumptions slice 5 (2026-08-05). GET /admin/api/assumptions
  // ?customer_id= -> { assumptions, count }: every assumption crystal
  // (pending + invalidated) with parsed confidence/provenance and live
  // parents hydrated via chains.
  listAssumptions: async (customerId: string): Promise<{ total: number; items: Assumption[] }> => {
    const body = await jsonFetch<any>(
      `/admin/api/assumptions${qs({ customer_id: customerId })}`
    );
    return {
      total: body.count ?? 0,
      items: (body.assumptions ?? []) as Assumption[],
    };
  },

  // POST /admin/api/assumptions/{id}/approve -> clears the recall gate
  // (the ratified promotion act); tier stays a signal the curator can
  // set separately.
  approveAssumption: (crystalId: string) =>
    jsonFetch<{ crystal_id: string; recall_gated: boolean }>(
      `/admin/api/assumptions/${encodeURIComponent(crystalId)}/approve`,
      { method: "POST" }
    ),

  // C4: POST /admin/api/assumptions/{id}/verify -> queues a research
  // task that hunts evidence for OR against the inference; the result
  // meets the assumption through the C3 scans.
  verifyAssumption: (crystalId: string) =>
    jsonFetch<{ crystal_id: string; task_id: string }>(
      `/admin/api/assumptions/${encodeURIComponent(crystalId)}/verify`,
      { method: "POST" }
    ),

  // v110: respond to an assumption; the model proposes revisions that
  // join the thread. GET the thread any member belongs to.
  respondAssumption: (crystalId: string, text: string) =>
    jsonFetch<{ root_id: string; created: string[]; thread: any[]; count: number }>(
      `/admin/api/assumptions/${encodeURIComponent(crystalId)}/respond`,
      { method: "POST", body: JSON.stringify({ text }) }
    ),

  assumptionThread: (crystalId: string) =>
    jsonFetch<{ root_id: string; thread: any[]; count: number }>(
      `/admin/api/assumptions/${encodeURIComponent(crystalId)}/thread`
    ),

  // Team (Refine PR 9): the operators the agent answers AS. Admin routes
  // under /admin/api/customers/{id}/ are path-scoped to the owning tenant.
  listOperators: (customerId: string) =>
    jsonFetch<{ operators: OperatorRow[] }>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/operators`
    ),
  seedOperator: (customerId: string, body: { name: string; role: string; facts: string[] }) =>
    jsonFetch<{ operator_id: string; display_name: string; facts_written: number }>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/operators/seed`,
      { method: "POST", body: JSON.stringify(body) }
    ),
  // PATCH: rename, append identity facts, and (PR 9) status
  // active | suspended. Suspension keeps the row; Undo reactivates.
  updateOperator: (customerId: string, operatorId: string,
                   body: { display_name?: string; facts?: string[]; status?: "active" | "suspended" }) =>
    jsonFetch<{ operator_id: string; display_name: string; facts_written: number; status?: string }>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/operators/${encodeURIComponent(operatorId)}`,
      { method: "PATCH", body: JSON.stringify(body) }
    ),

  // Your data (2026-10-03): erase, delete, restore. Owner only; the
  // server requires a fresh sign-in (X-Step-Up) and the typed word.
  eraseMyMemories: () =>
    jsonFetch<{ erased: boolean; rows: Record<string, number> }>(
      "/v1/me/erase",
      { method: "POST", body: JSON.stringify({ confirm: "ERASE" }) }
    ),
  deleteMyAccount: () =>
    jsonFetch<{ scheduled: boolean; purge_after: string | null; subscriptions_cancelled: number }>(
      "/v1/me/delete",
      { method: "POST", body: JSON.stringify({ confirm: "DELETE" }) }
    ),
  restoreMyAccount: () =>
    jsonFetch<{ restored: boolean }>("/v1/me/restore", { method: "POST" }),

  // DELETE /admin/api/crystals/{id} -> the curator delete (Gate D4a
  // route, long-standing; first client binding added for the
  // Assumptions surface).
  deleteCrystal: (crystalId: string) =>
    jsonFetch<{ crystal_id: string; deleted: boolean }>(
      `/admin/api/crystals/${encodeURIComponent(crystalId)}`,
      { method: "DELETE" }
    ),

  // C2 Q3=A: GET /admin/api/curation/activity?customer_id= ->
  // { events, count }: newest-first self-curation witness stream
  // (assumptions written/approved/invalidated/deleted, gaps
  // filled/reopened).
  listCurationActivity: async (customerId: string, limit = 50, eventType?: string): Promise<{ total: number; events: CurationEvent[] }> => {
    const body = await jsonFetch<any>(
      `/admin/api/curation/activity${qs({ customer_id: customerId, limit, event_type: eventType })}`
    );
    return {
      total: body.count ?? 0,
      events: (body.events ?? []) as CurationEvent[],
    };
  },

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

  // Scheduled runs (launch feature, 2026-10-18): tenant-pathed admin routes.
  listSchedules: (customerId: string): Promise<SchedulesResponse> =>
    jsonFetch<SchedulesResponse>(`/admin/api/customers/${encodeURIComponent(customerId)}/schedules`),
  createSchedule: (customerId: string, body: CreateScheduleRequest): Promise<ScheduleWriteResponse> =>
    jsonFetch<ScheduleWriteResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/schedules`,
      { method: "POST", body: JSON.stringify(body) }
    ),
  setScheduleStatus: (customerId: string, scheduleId: string, status: "active" | "paused"): Promise<ScheduleWriteResponse> =>
    jsonFetch<ScheduleWriteResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/schedules/${encodeURIComponent(scheduleId)}`,
      { method: "PATCH", body: JSON.stringify({ status }) }
    ),
  // "" clears the webhook; setting one on a schedule without a secret mints one (returned once).
  setScheduleDelivery: (customerId: string, scheduleId: string, deliverUrl: string): Promise<ScheduleWriteResponse> =>
    jsonFetch<ScheduleWriteResponse>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/schedules/${encodeURIComponent(scheduleId)}`,
      { method: "PATCH", body: JSON.stringify({ deliver_url: deliverUrl }) }
    ),
  deleteSchedule: (customerId: string, scheduleId: string): Promise<{ deleted: boolean }> =>
    jsonFetch<{ deleted: boolean }>(
      `/admin/api/customers/${encodeURIComponent(customerId)}/schedules/${encodeURIComponent(scheduleId)}`,
      { method: "DELETE" }
    ),

  // Refine PR 13 (5.11): POST /admin/api/agents/tasks/{id}/retry?customer_id=
  // -> { task }. Re-enqueues a FAILED daemon task as a fresh queued row.
  retryAgentTask: (customerId: string, taskId: string): Promise<{ task: AgentTask }> =>
    jsonFetch<{ task: AgentTask }>(
      `/admin/api/agents/tasks/${encodeURIComponent(taskId)}/retry${qs({ customer_id: customerId })}`,
      { method: "POST", body: JSON.stringify({}) }
    ),
};

export { ApiError };
