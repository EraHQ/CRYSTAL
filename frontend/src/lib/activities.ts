// Spend origins (llm_calls.origin) in the words a tenant recognizes.
// Two levels: the five ACTIVITIES the capacity card groups by (anything
// new lands in "Other" rather than vanishing), and the finer ORIGIN_TEXT
// the Activity page's Model calls view shows per call.
export const ACTIVITY_OF: Record<string, string> = {
  direct: "Answering your tools", agent: "Answering your tools", tool_loop: "Answering your tools", inline_research: "Answering your tools",
  code_descriptions: "Reading files", sparse_keys: "Reading files", schema_inference: "Reading files",
  shadow_critic: "Curation", shadow_eval: "Curation", self_critique: "Curation", scan_dedup: "Curation", scan_contradiction: "Curation",
  scan_gap_discovery: "Curation", failure_learning: "Curation", meta_reflection: "Curation", outbound_review: "Curation",
  cognition: "Research", precondition_verify: "Research",
  assumptions: "Assumptions", assumption_verification: "Assumptions",
};
export const ACTIVITIES = ["Answering your tools", "Reading files", "Curation", "Research", "Assumptions", "Other"];
export const activityOf = (origin: string): string => ACTIVITY_OF[origin] ?? "Other";

export const ORIGIN_TEXT: Record<string, string> = {
  direct: "Chat answer", agent: "Chat answer", tool_loop: "Tool loop", inline_research: "Inline research",
  code_descriptions: "Describing code", sparse_keys: "Memory keys", schema_inference: "Reading a source schema",
  shadow_critic: "Shadow critic", shadow_eval: "Shadow evaluation", self_critique: "Self-critique", scan_dedup: "Duplicate scan",
  scan_contradiction: "Conflict scan", scan_gap_discovery: "Gap discovery", failure_learning: "Learning from a failure",
  meta_reflection: "Reflection", outbound_review: "Reviewing an outbound save",
  cognition: "Research run", precondition_verify: "Verifying a precondition",
  assumptions: "Assumptions sweep", assumption_verification: "Verifying an assumption",
  extraction: "Extracting facts", crystallization: "Crystallizing a document", interactive: "Interactive",
};
export const originText = (origin: string): string => ORIGIN_TEXT[origin] ?? origin.replace(/_/g, " ");
