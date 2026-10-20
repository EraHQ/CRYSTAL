// Activity — Inspect (Q90=A, 2026-10-19). The home for discoverability
// and auditability: everything that happened with this memory, where it
// went, what it produced. Three views: Questions (every question a tool
// or the console asked, and what answered it), Runs (every cognition
// run: scheduled, research, assumption checks, tool-triggered), and
// Background (what the curators changed on their own). Not a replacement
// for Chat, Cognition or Scheduled runs: those are where work happens;
// this is where it is found. The view lives in the URL (?view=) so a
// page can link straight to it; a run's pipeline opens as the same
// drawer Cognition uses, at /activity/run/:runId.
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { History } from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useSelectedCustomer } from "@/lib/selected-customer";
import { EmptyState, PageHeader, Segmented } from "@/components/ui";
import { QueryLogPanel } from "./QueryLog";
import { RunsLog } from "@/components/activity/RunsLog";
import { BackgroundLog } from "@/components/activity/BackgroundLog";
import { ToolCalls } from "@/components/activity/ToolCalls";
import { ModelCalls } from "@/components/activity/ModelCalls";
import { RunDrawer } from "./CognitionTracker";

export type ActivityView = "questions" | "runs" | "tools" | "models" | "background";
const VIEWS: { value: ActivityView; label: string }[] = [
  { value: "questions", label: "Questions" },
  { value: "runs", label: "Runs" },
  { value: "tools", label: "Tool calls" },
  { value: "models", label: "Model calls" },
  { value: "background", label: "Background" },
];
export const isActivityView = (v: string | null): v is ActivityView => VIEWS.some((o) => o.value === v);

export function Activity() {
  const { selectedCustomerId } = useSelectedCustomer();
  const { me } = useAuth();
  const isPlatformAdmin = me?.role === "platform_admin" || me?.kind === "platform_admin_key";
  const [params, setParams] = useSearchParams();
  const { runId } = useParams<{ runId: string }>();
  const navigate = useNavigate();
  const raw = params.get("view");
  // A run drawer belongs to the Runs view.
  const view: ActivityView = runId ? "runs" : isActivityView(raw) ? raw : "questions";
  const setView = (v: ActivityView) => { setParams(v === "questions" ? {} : { view: v }, { replace: true }); };

  if (!selectedCustomerId) {
    return (
      <>
        <PageHeader title="Activity" />
        <EmptyState icon={History} title="No customer selected" help="Pick a customer above to see what happened with their memory." />
      </>
    );
  }

  return (
    <div>
      <PageHeader title="Activity" meta="Everything that happened with this memory: what was asked, what ran, every tool and model call, what the curators changed, and where it went." />
      <Segmented<ActivityView> label="View" value={view} onChange={setView} className="mb-4 flex-wrap" options={VIEWS} />
      {view === "questions" && <QueryLogPanel customerId={selectedCustomerId} admin={isPlatformAdmin} />}
      {view === "runs" && <RunsLog customerId={selectedCustomerId} onPipeline={(id) => navigate(`/activity/run/${encodeURIComponent(id)}`)} />}
      {view === "tools" && <ToolCalls customerId={selectedCustomerId} />}
      {view === "models" && <ModelCalls customerId={selectedCustomerId} admin={isPlatformAdmin} />}
      {view === "background" && <BackgroundLog customerId={selectedCustomerId} />}
      {runId && <RunDrawer runId={runId} onClose={() => navigate("/activity?view=runs")} />}
    </div>
  );
}
