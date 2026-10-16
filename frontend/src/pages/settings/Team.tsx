// Settings › Team (Refine PR 9, 5.8 item 7). Operators are who the agent
// answers AS; identity facts entered here are pinned into every
// conversation. Add and Edit open a Drawer; Remove is a suspension (the
// row survives) with Undo. The list route carries no email or fact
// count, so rows show the role, the status and whether an identity is
// seeded.
import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UserPlus } from "lucide-react";
import { api, errorMessage } from "@/lib/api";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { Drawer, EmptyState, ErrorBanner, Skeleton, StatusPill, useToast } from "@/components/ui";
import type { OperatorRow } from "@/lib/types";
import { BTN_PRIMARY, BTN_QUIET, BTN_SECONDARY, INPUT, Section } from "./shared";

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  return parts.slice(0, 2).map((p) => p[0]!.toUpperCase()).join("") || "?";
}

const ROLES = ["admin", "operator", "viewer"] as const;

export function Team() {
  const { selectedCustomerId } = useSelectedCustomer();
  const qc = useQueryClient();
  const toast = useToast();
  const [editing, setEditing] = useState<OperatorRow | "new" | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState<string>("admin");
  const [facts, setFacts] = useState("");
  const [error, setError] = useState<string | null>(null);

  const operators = useQuery({ queryKey: ["operators", selectedCustomerId], queryFn: () => api.listOperators(selectedCustomerId!), enabled: !!selectedCustomerId });
  const refresh = () => qc.invalidateQueries({ queryKey: ["operators", selectedCustomerId] });
  const factLines = () => facts.split("\n").map((f) => f.trim()).filter(Boolean);

  const open = (target: OperatorRow | "new") => {
    setEditing(target);
    setName(target === "new" ? "" : target.display_name);
    setRole(target === "new" ? "admin" : target.role);
    setFacts("");
    setError(null);
  };
  const close = () => setEditing(null);

  const save = useMutation({
    mutationFn: async () => {
      if (editing === "new") return api.seedOperator(selectedCustomerId!, { name: name.trim(), role, facts: factLines() });
      return api.updateOperator(selectedCustomerId!, (editing as OperatorRow).id, { display_name: name.trim(), facts: factLines() });
    },
    onSuccess: (out) => {
      setError(null);
      toast(editing === "new" ? `${out.display_name} added${out.facts_written ? ` with ${out.facts_written} identity fact${out.facts_written === 1 ? "" : "s"}` : ""}.`
        : `${out.display_name} updated${out.facts_written ? `, ${out.facts_written} fact${out.facts_written === 1 ? "" : "s"} added` : ""}.`);
      close(); refresh();
    },
    onError: (e) => setError(errorMessage(e, "Couldn't save this operator.")),
  });
  const setStatus = useMutation({
    mutationFn: ({ id, status }: { id: string; status: "active" | "suspended" }) => api.updateOperator(selectedCustomerId!, id, { status }),
    onSuccess: (_o, { id, status }) => {
      setError(null);
      const row = operators.data?.operators.find((x) => x.id === id);
      if (status === "suspended") {
        toast(`${row?.display_name ?? "Operator"} removed from the team.`, { action: { label: "Undo", onClick: () => setStatus.mutate({ id, status: "active" }) }, duration: 6000 });
      } else {
        toast(`${row?.display_name ?? "Operator"} is back on the team.`);
      }
      refresh();
    },
    onError: (e) => setError(errorMessage(e, "Couldn't change this operator.")),
  });

  const rows = operators.data?.operators ?? [];
  const busy = save.isPending || setStatus.isPending;

  return (
    <div className="space-y-4">
      {error && !editing && <ErrorBanner title="That didn't go through" message={error} />}
      <Section title="Team" help="Who the agent answers as. Identity facts are recalled in every conversation; with one active operator, “me” and “my” resolve to them.">
        <div className="mb-3">
          <button type="button" onClick={() => open("new")} className={BTN_PRIMARY}><UserPlus aria-hidden className="h-3.5 w-3.5" /> Add operator</button>
        </div>
        {operators.isLoading && <Skeleton rows={2} />}
        {!operators.isLoading && rows.length === 0 && (
          <EmptyState icon={UserPlus} title="No operators yet" help="Add yourself so the agent can answer as you." />
        )}
        <ul className="divide-y divide-line">
          {rows.map((op) => (
            <li key={op.id} className="flex flex-wrap items-center gap-3 py-2.5">
              <span aria-hidden className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[12px] font-semibold text-accent">{initials(op.display_name)}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13.5px] font-medium text-fg">{op.display_name}</span>
                <span className="block text-[12px] text-fg-3">{op.has_identity ? "Identity seeded" : "No identity yet"}</span>
              </span>
              <StatusPill variant="neutral" raw={op.role}>{op.role}</StatusPill>
              {op.status !== "active" && <StatusPill variant="warn" raw={op.status}>Removed</StatusPill>}
              <button type="button" disabled={busy} onClick={() => open(op)} className={BTN_SECONDARY}>Edit</button>
              {op.status === "active" ? (
                <button type="button" disabled={busy} onClick={() => setStatus.mutate({ id: op.id, status: "suspended" })} className={BTN_QUIET + " hover:text-bad"}>Remove</button>
              ) : (
                <button type="button" disabled={busy} onClick={() => setStatus.mutate({ id: op.id, status: "active" })} className={BTN_QUIET}>Put back</button>
              )}
            </li>
          ))}
        </ul>
      </Section>

      <Drawer open={editing !== null} onClose={close} title={editing === "new" ? "Add operator" : "Edit operator"}
        footer={<>
          <button type="button" onClick={close} className={BTN_SECONDARY}>Cancel</button>
          <button type="button" disabled={busy || !name.trim()} onClick={() => save.mutate()} className={BTN_PRIMARY}>{save.isPending ? "Saving…" : editing === "new" ? "Add operator" : "Save"}</button>
        </>}>
        <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); if (name.trim()) save.mutate(); }}>
          {error && <ErrorBanner title="That didn't go through" message={error} />}
          <label className="block text-[12.5px] font-medium text-fg-2">Name
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Anthony" className={INPUT + " mt-1"} />
          </label>
          {editing === "new" && (
            <label className="block text-[12.5px] font-medium text-fg-2">Role
              <select value={role} onChange={(e) => setRole(e.target.value)} className={INPUT + " mt-1"}>
                {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </label>
          )}
          <label className="block text-[12.5px] font-medium text-fg-2">Identity facts, one per line
            <textarea value={facts} onChange={(e) => setFacts(e.target.value)} rows={4} placeholder={"Owner of Wren & Sparrow\nHandles supplier relations and pricing"} className={INPUT + " mt-1"} />
          </label>
          <p className="text-[12px] text-fg-3">Crystal recalls these when {name.trim() || "they"} asks something.{editing !== "new" && " New lines are added to what is already saved."}</p>
        </form>
      </Drawer>
    </div>
  );
}
