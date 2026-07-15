// Evidence Bench — critique API + shared types (Q2B frontend,
// ratified 2026-07-15). target_path addresses the run's anatomy:
// "run", "criterion:2", "step:3", "step:3/tool_call:1",
// "step:2/finding:4", "attempt:1", "verdict", "deliverable".
import { authedFetch } from "@/lib/api";

export interface Critique {
  id: string;
  run_id: string;
  customer_id: string;
  trigger_id: string | null;
  target_path: string;
  author: string;
  text: string;
  status: "open" | "resolved";
  created_at: string | null;
  resolved_at: string | null;
}

export async function fetchCritiques(envId: string): Promise<Critique[]> {
  const res = await authedFetch(
    `/admin/api/cognition/environments/${encodeURIComponent(envId)}/critiques`);
  if (!res.ok) throw new Error(`${res.status}`);
  const data = await res.json();
  return data.critiques ?? [];
}

export async function postCritique(
  envId: string, targetPath: string, text: string,
): Promise<Critique> {
  const res = await authedFetch(
    `/admin/api/cognition/environments/${encodeURIComponent(envId)}/critiques`,
    {
      method: "POST",
      body: JSON.stringify({ target_path: targetPath, text }),
    });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

export async function patchCritique(
  critiqueId: string, status: "open" | "resolved",
): Promise<Critique> {
  const res = await authedFetch(
    `/admin/api/cognition/critiques/${encodeURIComponent(critiqueId)}`,
    { method: "PATCH", body: JSON.stringify({ status }) });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

/** Critiques whose target sits at or under a node path — a step node
 * owns its tool calls' and findings' critiques for count roll-up. */
export function critiquesUnder(all: Critique[], path: string): Critique[] {
  if (path === "critiques") return all;
  return all.filter(
    (c) => c.target_path === path || c.target_path.startsWith(path + "/"));
}

export function openCount(all: Critique[], path: string): number {
  return critiquesUnder(all, path).filter((c) => c.status === "open").length;
}
