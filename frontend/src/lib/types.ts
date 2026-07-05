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
}

export interface CrystalsListResponse {
  total: number;
  items: CrystalSummary[];
}

export interface FactSummary {
  id: string;
  claim_text: string;
  pair_type: string;
  source_kind: string;
  prompt_text: string;
  answer_value: string | null;
  created_at: string;
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
  prompt_token_overhead: number | null;
  shadow_ran: boolean;
  shadow_delta: number | null;
  concept_top_config: string | null;
  concept_top_score: number | null;
  concept_payload: Record<string, unknown> | null;
  latency_ms: number | null;
}

export interface QueryLogsListResponse {
  total: number;
  items: QueryLogSummary[];
}

export interface AdminKeyResponse {
  api_key: string;
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
