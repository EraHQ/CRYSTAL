// Shapes returned by /admin/api/* — mirror src/crystal_cache/ingress/admin.py.
// Keep these manually in sync. If the count of fields ever exceeds what's
// reasonable to maintain by hand, generate from the OpenAPI schema.

export interface CustomerSummary {
  id: string;
  provider: string;
  model_id: string;
  created_at: string;
  crystal_count: number;
}

export interface CustomersListResponse {
  items: CustomerSummary[];
}

export interface CrystalSummary {
  id: string;
  summary_text: string | null;
  fact_count: number;
  quality_tier: string;
  diagnostic_tags: string[];
  created_at: string;
  // P3 bank readability: representative sparse key + classification
  // fields. The list route attaches these so the bank renders a human
  // breadcrumb + title and groups agent-made crystals.
  crystal_type?: string;
  build_method?: string | null;
  headline_key?: string | null;
  headline_claim?: string | null;
  headline_source_kind?: string | null;
  // Gate D6: path identity for file crystals (labels + grouping).
  source_path?: string | null;
  source_uri?: string | null;
  // Console Refine PR 4 (5.1): sum of the crystal's facts' hit_count —
  // the Map keeps the 80 most recalled; cards say nothing about it.
  recall_hits?: number;
  last_activity?: string | null;
}

// Console Refine PR 4 (5.1): whole-bank numbers for the header and the
// trust chips; rides every list page regardless of the filter.
export interface BankOverview {
  crystals: number;
  facts: number;
  updated_at: string | null;
  tier_counts: Record<string, number>;
}

export interface CrystalsListResponse {
  total: number;
  items: CrystalSummary[];
  bank: BankOverview;
}

export interface FactSummary {
  id: string;
  claim_text: string;
  pair_type: string;
  source_kind: string;
  prompt_text: string;
  answer_value: string | null;
  created_at: string;
  // Console Refine PR 4 (5.1 item 7): "Recalled {n} times"; distinct
  // source_doc_ids count as the crystal's sources.
  hit_count?: number;
  source_doc_id?: string | null;
  citation?: string | null;
}

export interface CrystalDetail {
  id: string;
  customer_id: string;
  summary_text: string | null;
  fact_count: number;
  quality_tier: string;
  diagnostic_tags: string[];
  keyword_fingerprint: string[];
  cluster_tightness: number | null;
  build_method: string;
  parent_crystal_id: string | null;
  created_at: string;
  last_activity: string;
  facts: FactSummary[];
  // Console Refine PR 4: identity fields so /bank/{id} renders without
  // a list fetch (same names as CrystalSummary).
  crystal_type?: string;
  source_path?: string | null;
  source_uri?: string | null;
  headline_key?: string | null;
  headline_source_kind?: string | null;
  recall_hits?: number;
}

// ── Knowledge (Refine PR 5) ──────────────────────────────────────────

// One row of GET /v1/documents (endpoints/documents.py _doc_to_response).
// status: pending | review | approved | crystallizing | crystallized | error.
export interface DocumentItem {
  id: string;
  label: string;
  status: string;
  char_count: number;
  crystals_written: number;
  items_extracted: number;
  content_chunks_count?: number;
  error_message?: string | null;
  detected_type?: string | null;
  created_at: string;
  crystallized_at?: string | null;
}

export interface ReviewChunk {
  index: number;
  label: string;
  text: string;
  char_count: number;
  // Gate D3: the describer's judgment (or the curator's); steers retrieval.
  description?: string | null;
  // Gate D4: the C2 screen's chunk-time findings.
  injection_hits?: string[] | null;
}

export interface ReviewItem {
  key: string;
  sparse_key?: string;
  value: string;
  type: string;
}

// GET /v1/documents/{id}/review
export interface DocumentReview {
  document_id: string;
  label: string;
  status: string;
  detected_type: string;
  confirmed_type: string | null;
  content_chunks: ReviewChunk[];
  extracted_items: ReviewItem[];
  char_count: number;
  items_extracted: number;
  comprehension?: { imports: { module: string; resolved_path: string | null }[] } | null;
}

// GET /admin/api/watches (endpoints/admin.py _watch_payload + sync_state).
export interface Watch {
  id: string;
  scheme: string;
  source_name: string;
  config: { repo?: string; branch?: string; folder_id?: string; folder_name?: string; connection_id?: string };
  cadence_minutes: number;
  review_mode: string;
  status: string;
  has_token: boolean;
  last_state: { head?: string } | null;
  last_checked_at: string | null;
  last_error: string | null;
  sync_state?: string;
  inflight?: number;
}

export interface WatchActivity {
  event_type: string;
  label: string;
  payload: Record<string, unknown> | null;
  created_at: string | null;
}

// GET /admin/api/source-schemas (Gate G3).
export interface SchemaMapping {
  version?: number;
  roles?: Record<string, string>;
  subject?: string | null;
  domain?: string | null;
}

export interface SourceSchemaItem {
  id: string;
  schema_hash: string;
  status: string;
  mapping: SchemaMapping;
  sample: unknown[];
  label: string | null;
  parked_count: number;
  created_at: string | null;
  updated_at: string | null;
}

export interface SchemaPreviewItem {
  key: string;
  sparse_key: string;
  value: string;
  type: string;
  citation: string;
}

// ── Cognition (Refine PR 6) ─────────────────────────────────────────

// One row of GET /admin/api/cognition/environments (cognition/api.py;
// the stored summary of a run). cost_usd is for the admin view only;
// capacity_share_pct is the tenant-facing figure (percent of the daily
// allowance, null without a tier).
export interface CognitionRun {
  id: string;
  customer_id: string;
  status: string;
  trigger_type: string;
  trigger_id?: string;
  cycle?: number;
  cycle_cap?: number;
  goal_title: string;
  attempts: number;
  max_attempts: number;
  step_count: number;
  steps_complete: number;
  cost_usd: number;
  tokens_used: number;
  created_at: string;
  completed_at?: string | null;
  open_critiques?: number;
  capacity_share_pct?: number | null;
}

export interface CriterionEval {
  criterion: string;
  status: string;
  evidence: string;
  possibly_infeasible?: boolean;
}

export interface CognitionStep {
  step_id: number;
  action: string;
  status: string;
  model_used: string;
  duration_ms: number;
  output: any;
  error: string | null;
}

// GET /admin/api/cognition/environments/{id} (env.to_dict()).
export interface CognitionRunDetail {
  id: string;
  customer_id: string;
  status: string;
  trigger_type: string;
  trigger_id?: string;
  cycle?: number;
  cycle_cap?: number;
  attempts: number;
  max_attempts: number;
  cost_usd: number;
  tokens_used: number;
  created_at: string;
  capacity_share_pct?: number | null;
  goal: {
    title: string;
    description: string;
    acceptance_criteria: string[];
    amendments?: Array<{ attempt: number; index: number; original: string; amended: string; evidence: string }>;
  } | null;
  plan: {
    reasoning: string;
    steps: Array<{ id: number; action: string; description: string; parallel_group: string | null; model: string }>;
    suggested_key: string;
    retry_route?: string;
  } | null;
  steps: Record<string, CognitionStep>;
  deliverables: Record<string, string>;
  validation: {
    approved: boolean;
    score: number;
    reasoning: string;
    criteria_evaluation: CriterionEval[];
    issues: string[];
  } | null;
  rejection_log: Array<{ attempt: number; reasoning: string; score: number }>;
  events?: Array<Record<string, any>>;
  attempt_history?: Array<{
    attempt: number;
    plan: { reasoning?: string; retry_route?: string; steps?: Array<{ id: number; action: string; description?: string }> } | null;
    steps: Record<string, { step_id?: number; action: string; status: string; duration_ms?: number; model_used?: string; output?: any; error?: string | null }>;
    deliverable: string;
    validation: { approved: boolean; score: number; reasoning?: string; issues?: string[] };
  }>;
}

// GET /admin/api/knowledge-gaps (endpoints/admin.py). status: open |
// filled | closed | retrying | needs_operator; disposition: researchable |
// workable | needs_document | cycles_exhausted.
export interface KnowledgeGap {
  id: string;
  customer_id: string;
  domain: string | null;
  subject: string | null;
  missing: string;
  priority: string;
  status: string;
  disposition: string | null;
  source: string;
  filled_by_crystal_id: string | null;
  filled_snippet: string | null;
  full_key?: string | null;
  triggering_query?: string | null;
  run_count: number;
  last_run_id: string | null;
  last_run_status: string | null;
  created_at: string;
  resolved_at: string | null;
}

// GET /admin/api/push-queue: facts the model was not sure enough to save.
export interface PushQueueItem {
  id: string;
  key: string;
  value: string;
  confidence: number;
  status: string;
  crystal_id?: string | null;
  created_at: string;
}

// GET /admin/api/cognition-tasks
export interface CognitionTask {
  id: string;
  task_type: string;
  status: string;
  priority: string;
  payload: Record<string, any> | null;
  result?: Record<string, any> | null;
  result_crystal_id?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
}

// GET /admin/api/cost/daily (admin dollars; the tenant card uses shares).
export interface DailyCost {
  day_start: string;
  families: { origin: string; cost_micro_usd: number; calls: number }[];
  total_micro_usd: number;
  budget_per_customer_usd: number | null;
}

// GET /v1/customers/{id}/budgets -> budgets[]
export interface SpendBudget {
  function: string;
  cap_micro_usd: number;
  period: string;
  // Present on per-operator rows; the tenant-wide row has none.
  operator_id?: string | null;
}

// ── Conflicts and Assumptions (Refine PR 7) ─────────────────────────

// One side of a conflict (endpoints/admin.py admin_list_conflicts `facts`).
export interface ConflictFact {
  learned_at: string | null;
  source_kind: string | null;
  source_path: string | null;
  crystal_id: string | null;
  active: boolean;
}

// What the scanner would do about a contradiction (Refine PR 7b): written
// by a second small call after a CONTRADICTS verdict, or by /explain.
// resolution: qualified | superseded | blacklisted | dismissed; loser is
// the side to drop for superseded / blacklisted, null otherwise.
export interface ConflictRecommendation {
  resolution: string;
  loser: "a" | "b" | null;
  reason: string;
}

// GET /admin/api/conflicts (models/knowledge_conflict.py + facts).
// detector: contradiction_scan | dedup_scan; status: open | resolved |
// dismissed; resolution: qualified | superseded | blacklisted | dismissed.
export interface KnowledgeConflict {
  id: string;
  customer_id: string;
  fact_a_id: string;
  fact_b_id: string;
  crystal_a_id: string | null;
  crystal_b_id: string | null;
  subject: string | null;
  claim_a: string;
  claim_b: string;
  provenance_a: string | null;
  provenance_b: string | null;
  detector: string;
  status: string;
  resolution: string | null;
  pair_key: string;
  // Refine PR 7b: "why it looks like a contradiction" and the suggested
  // verb; null until explained (duplicates, older rows, a failed call).
  reasoning?: string | null;
  recommendation?: ConflictRecommendation | null;
  created_at: string;
  resolved_at: string | null;
  facts?: { a: ConflictFact | null; b: ConflictFact | null };
}

// GET /admin/api/customers/{id}/operators (Refine PR 9 Team tab).
export interface OperatorRow {
  id: string;
  display_name: string;
  role: string;
  status: string;
  has_identity: boolean;
}

// GET /admin/api/backlog: kind gap | conflict | cognition_task |
// agent_task | push_review | verification.
export interface BacklogItem {
  kind: string;
  id: string;
  subject: string;
  status: string;
  priority_score: number;
  created_at: string | null;
}

// GET /admin/api/assumptions (metadata_store_assumption_ext
// list_assumption_crystals + parse_assumption_tags + parents).
export interface Assumption {
  id: string;
  statement: string | null;
  quality_tier: string;
  recall_gated: boolean;
  diagnostic_tags: string[];
  parent_crystal_id: string | null;
  created_at: string;
  confidence: number | null;
  gap_id: string | null;
  invalidated_parents: string[];
  thread_id: string | null;
  responds_to: string | null;
  iteration: number | null;
  replaces_parent: boolean;
  superseded_by: string | null;
  rationale: string | null;
  // 2026-10-21: label (summary, else the source file, else the first key)
  // and up to three claims, so the page quotes the evidence in text.
  parents: Array<{ id: string; summary_text: string | null; label?: string | null; claims?: string[] }>;
  // Thread read only.
  operator_response?: string | null;
}

// GET /admin/api/curation/activity
export interface CurationEvent {
  id: string;
  subject_id: string | null;
  event_type: string;
  label: string;
  payload: Record<string, any> | null;
  created_at: string | null;
}

// GET /admin/api/crystals/{id}/ledger — one append-only row per fact
// edit or retirement (the reader's History tab).
export interface LedgerEntry {
  id: string;
  op: string;
  actor: string;
  fact_id: string;
  successor_fact_id: string | null;
  before_prompt: string | null;
  before_text: string | null;
  after_text: string | null;
  created_at: string | null;
}

export type MatchType = "high" | "medium" | "low" | "none";

export interface QueryLogSummary {
  id: string;
  timestamp: string;
  query_text: string;
  match_type: MatchType;
  injection_method: string;
  matched_facts: string[];
  response_text: string | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  // S12: prompt-caching split (agent turns).
  cache_read_tokens?: number | null;
  cache_creation_tokens?: number | null;
  prompt_token_overhead: number | null;
  shadow_ran: boolean;
  shadow_delta: number | null;
  concept_top_config: string | null;
  concept_top_score: number | null;
  concept_payload: Record<string, unknown> | null;
  latency_ms: number | null;
  // Refine PR 11: which tool asked (User-Agent family, "console", or null
  // on older rows) and the memories that answered, as text.
  client?: string | null;
  // M3 follow-up: the model that answered (agent rows; null on
  // recall-only rows and before the column).
  model?: string | null;
  matched_memories?: { id: string; text: string }[];
}

export interface QueryLogsListResponse {
  total: number;
  items: QueryLogSummary[];
}

// /v1/chat/completions — only the slice we use in the playground.
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface ChatCompletionResponse {
  id: string;
  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// POST /admin/api/customers/{id}/agent (and /v1/agent/messages) — CRYS's run
// result. The Inspector Chat drives CRYS, not the proxy; this mirrors
// Agent.run()'s dict (see endpoints/agent.py run_agent_messages). The
// trajectory is Anthropic Messages-shaped: content is a string or a list of
// typed blocks (text / tool_use / tool_result). P4c renders the blocks.
export interface AgentToolCall {
  iteration: number;
  tool_name: string;
  tool_use_id: string;
  input: Record<string, unknown>;
  output: unknown; // dict or string (JSON-serialized at the tool boundary)
  is_error: boolean;
  duration_ms?: number;
}

// P4d — the output shape of the create_document tool (agent/tools/
// artifacts.py). The model writes `content`; the tool packages it for
// download. The Inspector Chat narrows a tool_call's `output` to this and
// renders a DocumentCard (download + preview) instead of raw JSON. `bytes`
// is the UTF-8 byte length of `content`.
export interface DocumentArtifact {
  type: "document";
  filename: string;
  format: string; // "md" | "txt" | "html"
  mime: string;
  title: string;
  content: string;
  bytes: number;
}

export type AgentContentBlock =
  | { type: "text"; text: string }
  | { type: "tool_use"; id: string; name: string; input: Record<string, unknown> }
  | { type: "tool_result"; tool_use_id: string; content: string; is_error?: boolean }
  | { type: string; [key: string]: unknown }; // forward-compat for unknown blocks

export interface AgentTrajectoryMessage {
  role: "user" | "assistant";
  content: string | AgentContentBlock[];
}

export interface AgentRunResponse {
  id: string;
  model: string;
  messages: AgentTrajectoryMessage[];
  final_text: string;
  stop_reason: string;
  iterations: number;
  prompt_tokens: number;
  completion_tokens: number;
  tool_calls: AgentToolCall[];
  mcr?: Record<string, unknown> | null;
}

// POST /v1/customers — onboarding. Mirrors ingress/schema.py
// CreateCustomerRequest / CreateCustomerResponse.
export interface CreateCustomerRequest {
  provider: "openai" | "anthropic" | "self_hosted";
  model_id: string;
  api_key_ref: string; // upstream provider key (Key B)
  base_url?: string; // required when provider === "self_hosted"
  injection_preference?: "text" | "hidden_state" | "none";
  shadow_sample_rate?: number;
}

export interface CreateCustomerResponse {
  id: string;
  api_key: string; // Key A — the Crystal Cache key, shown once
  provider: string;
  model_id: string;
}

// /admin/api/sessions* — the Foundation F4 session registry (live agents),
// plus the G2 control commands. Mirrors agent_sessions / session_dependencies
// / control_commands. `effective_status` and `is_stale` are derived per row
// by the registry (a stale non-terminal session reads "crashed").
// Scheduled runs (launch feature, 2026-10-18).
export interface ScheduledRun {
  id: string;
  customer_id: string;
  kind: "brief" | "research" | string;
  title: string;
  prompt: string;
  cadence: string;
  at_minute: number | null;
  weekday: number | null;
  every_hours: number | null;
  run_once_at: string | null;
  tz: string;
  next_run_at: string | null;
  status: "active" | "paused" | "parked" | "done" | string;
  series_failures: number;
  last_task_id: string | null;
  last_run_at: string | null;
  last_outcome: "queued" | "completed" | "failed" | "skipped_capacity" | string | null;
  last_result_summary: string | null;
  last_crystal_id: string | null;
  deliver_url: string | null;
  delivers: boolean;
  last_delivery: string | null;
  last_delivery_at: string | null;
  when: string;
  created_at: string | null;
  updated_at: string | null;
}
export interface SchedulesResponse { schedules: ScheduledRun[]; count: number; active: number; max_active: number }

// Activity trace (2026-10-19): endpoints/activity.py.
export interface ModelCall {
  id: string;
  session_id: string | null;
  parent_session_id: string | null;
  origin: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cache_creation_tokens: number;
  cache_read_tokens: number;
  cost_micro_usd?: number; // platform admins only
  created_at: string | null;
}
export interface ToolCall {
  id: string;
  surface: string;
  client: string | null;
  tool: string;
  args: Record<string, unknown> | null;
  ok: boolean;
  code: string | null;
  result_summary: string | null;
  duration_ms: number | null;
  query_log_id: string | null;
  created_at: string | null;
}
export interface QueryTrace {
  query_log_id: string;
  sequence_id: string | null;
  turn_index: number | null;
  model: string | null;
  trace: {
    id: string;
    tool_calls: AgentToolCall[];
    events: Record<string, unknown>[];
    crystals_used: string[];
    inferences: Record<string, unknown>[];
    borders_crossed: Record<string, unknown>[];
    gaps_felt: Record<string, unknown>[];
    created_at: string | null;
  } | null;
  model_calls: ModelCall[];
  tool_calls: ToolCall[];
}
export interface CreateScheduleRequest {
  kind: "brief" | "research"; cadence: string; tz: string; time?: string; weekday?: string;
  every_hours?: number; run_at?: string; question?: string; title?: string; deliver_url?: string;
}
// Edit (2026-10-20): the definition fields of CreateScheduleRequest; what is
// not sent keeps its current value. The webhook has its own editor.
export type EditScheduleRequest = Partial<Omit<CreateScheduleRequest, "deliver_url">>;
// deliver_secret is present only when a webhook was set and is never returned again.
export interface ScheduleWriteResponse { schedule: ScheduledRun; deliver_secret?: string }
export interface RunNowResponse { task_id: string; schedule: ScheduledRun }

export interface AgentSession {
  session_id: string;
  team_id: string;
  operator_id: string | null;
  host: string | null;
  pid: number | null;
  project_dir: string | null;
  model: string | null;
  status: string;
  effective_status: string;
  is_stale: boolean;
  current_action: string | null;
  awaiting_payload: Record<string, unknown> | null;
  parent_session_id: string | null;
  started_at: string | null;
  last_heartbeat_at: string | null;
  cost_usd_cumulative: number | null;
}

export interface SessionDependency {
  dependency_id: string;
  session_id: string;
  kind: string; // mcp_server | subprocess | browser | queued_task | pip_env
  descriptor: string;
  pid: number | null;
  status: string; // active | exited | orphaned
  spawned_at: string | null;
}

export interface ControlCommand {
  id: string;
  session_id: string;
  request_id: string;
  command_type: string; // approval_decision | terminate | terminate_dependency
  decision: string | null;
  dependency_id: string | null;
  status: string; // pending | consumed | voided
  created_at: string | null;
  consumed_at: string | null;
}

export interface SessionsListResponse {
  sessions: AgentSession[];
  count: number;
}

export interface SessionDependenciesResponse {
  dependencies: SessionDependency[];
  count: number;
}

export interface SessionCommandsResponse {
  commands: ControlCommand[];
  count: number;
}

// /admin/api/agents/* — the Unify-Agents surfaces. agent_events is the
// per-session activity stream (turns / tool calls / subagents / crystals /
// gaps); agent_tasks is the daemon queue; agent-run gaps are terminal
// background failures. Mirrors agent_events / agent_tasks / knowledge_gaps
// (source='agent_run').
export interface AgentEvent {
  id: string;
  session_id: string;
  team_id: string | null;
  seq: number;
  turn_index: number | null;
  parent_session_id: string | null;
  event_type: string;
  phase: string | null;
  label: string;
  payload: Record<string, unknown> | null;
  status: string | null;
  duration_ms: number | null;
  tokens_input: number | null;
  tokens_output: number | null;
  cost_micro_usd: number | null;
  created_at: string | null;
}

export interface AgentTask {
  id: string;
  customer_id: string;
  project_dir: string | null;
  task: string;
  branch: string | null;
  status: string; // queued | running | done | failed
  source: string; // cli | agent | gap_retry | ...
  run_at: string | null;
  recur_seconds: number | null;
  parent_task_id: string | null;
  series_failures: number | null;
  report: string | null;
  error: string | null;
  log_path: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface AgentGap {
  id: string;
  customer_id: string;
  subject: string;
  missing: string;
  status: string; // open | retrying | filled | needs_operator
  created_at: string | null;
  resolved_at: string | null;
  filled_by_crystal_id: string | null;
}

export interface AgentEventsResponse {
  events: AgentEvent[];
  count: number;
}

export interface AgentTasksResponse {
  tasks: AgentTask[];
  count: number;
}

export interface AgentGapsResponse {
  gaps: AgentGap[];
  count: number;
}

// The Mind tab (2026-10-21): everything the scene draws, from
// GET /admin/api/customers/{id}/mind. Positions are scene units
// (about 50 = the bank's radius), laid out server-side.
export type MindTier = "trusted" | "reviewed" | "new" | "wrong";
export interface MindRegion { id: number; name: string; center: [number, number, number]; facts: number; clusters: number; trusted: number; hits: number }
export interface MindFact { id: string; crystal_id: string; region: number; pos: [number, number, number]; tier: MindTier; hits: number; key: string; claim: string; source_kind: string | null; doc_id: string | null; learned_at: string | null }
export interface MindSource { id: string; label: string; region: number; facts: number; learned_at: string | null }
export interface MindBond { a: string; b: string; a_pos: [number, number, number]; b_pos: [number, number, number]; region_a: number | null; region_b: number | null; type: string | null; weight: number }
export interface MindConflictSide { fact_id: string; claim: string; pos: [number, number, number] | null; region: number | null }
export interface MindConflict { id: string; kind: "contradiction" | "duplicate"; subject: string | null; status: string; resolution: string | null; a: MindConflictSide; b: MindConflictSide; reasoning: string | null; recommendation: ConflictRecommendation | null; found_at: string | null }
export interface MindAssumptionParent { crystal_id: string; label: string; pos: [number, number, number] | null; region: number | null }
export interface MindAssumption { id: string; statement: string; confidence: number | null; rationale: string | null; state: "waiting" | "approved" | "invalidated"; parents: MindAssumptionParent[]; formed_at: string | null }
export interface MindGap { id: string; missing: string; subject: string | null; priority: string; source: string; disposition: string | null; region: number | null; opened_at: string | null }
export interface MindRun { id: string; type: string; origin: string | null; title: string; status: string; outcome: string | null; summary: string; started_at: string | null; finished_at: string | null; result_crystal_id: string | null; region: number | null; landing: [number, number, number] | null }
export interface MindRecall { id: string; question: string; at: string | null; client: string | null; model: string | null; match: string; latency_ms: number | null; region: number | null; touched: string[] }
export interface MindEvent { id: string; type: string; label: string | null; subject_id: string | null; at: string }
export interface MindLane { name: string; hours: Array<{ calls: number; tokens: number }> }
export interface MindScene {
  generated_at: string;
  window: { since: string; until: string; hours: number };
  totals: { facts: number; clusters: number; regions: number; sources: number; contradictions: number; duplicates: number; assumptions_waiting: number; gaps: number; runs: number; recalls: number };
  regions: MindRegion[];
  facts: MindFact[];
  sources: MindSource[];
  bonds: MindBond[];
  conflicts: MindConflict[];
  assumptions: MindAssumption[];
  gaps: MindGap[];
  runs: MindRun[];
  recalls: MindRecall[];
  events: MindEvent[];
  lanes: MindLane[];
}

// Hackathon item 1 (2026-10-22): every text model Token Factory lists,
// for the Settings picker. No dollars: a cost band and the context length.
export interface CatalogModelCaps {
  json_schema: boolean;
  tools: boolean;
  error?: string | null;
  probed_at?: string;
}
export interface CatalogModel {
  id: string;
  name: string;
  context_length: number;
  band: "light" | "standard" | "heavy";
  fast: boolean;
  nvidia: boolean;
  caps: CatalogModelCaps | null;
}
export interface ModelCatalog {
  enabled: boolean;
  models: CatalogModel[];
}

// Item 2 (2026-10-22): the providers a tenant's own key can belong to.
export type UpstreamProvider = "anthropic" | "openai" | "nebius" | "self_hosted";
