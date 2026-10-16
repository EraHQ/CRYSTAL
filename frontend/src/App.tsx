import { useState } from "react";
import { NavLink, Route, Routes, Navigate, useLocation } from "react-router-dom";
import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { AuthProvider, useAuth } from "@/lib/auth";
import { api, errorMessage } from "@/lib/api";
import { BankBrowser } from "@/pages/BankBrowser";
import { ChatPlayground } from "@/pages/ChatPlayground";
import { Cognition } from "@/pages/Cognition";
import { KnowledgeManager } from "@/pages/KnowledgeManager";
import { QueryLog } from "@/pages/QueryLog";
import { Onboard } from "@/pages/Onboard";
import { Agents } from "@/pages/Agents";
import { Conflicts } from "@/pages/Conflicts";
import { Assumptions } from "@/pages/Assumptions";
import { Critiques } from "@/pages/Critiques";
import { Login } from "@/pages/Login";
import { OnboardingSetup } from "@/pages/OnboardingSetup";
import { OAuthConsent } from "@/pages/OAuthConsent";
import { SettingsApi } from "@/pages/SettingsApi";
import { Billing } from "@/pages/Billing";
import { SelfCurationBanner } from "@/components/SelfCurationBanner";
import { GettingStarted } from "@/components/GettingStarted";
import { PlanWallModal } from "@/components/PlanWallModal";
import { Sidebar } from "@/components/shell/Sidebar";
import { BottomBar } from "@/components/shell/BottomBar";
import { CrystalMark } from "@/components/shell/CrystalMark";
import { ConfirmInline } from "@/components/ui";
import { cn } from "@/lib/utils";

// Deletion banner (4.2): warn-soft surface, body text in fg, the two
// actions the locked state allows. "Export my data" goes to Your data in
// Settings; "Cancel deletion" confirms inline and restores the account.
function DeletionBanner() {
  const { me, refreshMe } = useAuth();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!me?.deletion_scheduled_at) return null;
  const when = me.purge_after ? new Date(me.purge_after).toLocaleDateString() : "the scheduled date";
  return (
    <div role="status" className="border-b border-warn/40 bg-warn-soft px-4 py-2 text-[12.5px] text-fg min-[720px]:px-6">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="min-w-0 flex-1">
          This account is scheduled for deletion on {when}. Only export and restore work until then.
        </span>
        <NavLink to="/settings" className="rounded-md border border-line px-2.5 py-1 font-medium text-fg-2 hover:bg-surface hover:text-fg">
          Export my data
        </NavLink>
        <button type="button" onClick={() => setConfirming(true)} disabled={confirming || busy}
          className="rounded-md bg-accent px-2.5 py-1 font-semibold text-on-accent hover:bg-accent-hover disabled:opacity-50">
          Cancel deletion
        </button>
      </div>
      <ConfirmInline
        open={confirming}
        className="mt-2 border-line bg-surface"
        message="Cancel the scheduled deletion and unlock this account?"
        confirmLabel="Keep my account"
        busy={busy}
        onCancel={() => setConfirming(false)}
        onConfirm={async () => {
          setBusy(true);
          try {
            await api.restoreMyAccount();
            await refreshMe();
            setConfirming(false);
          } catch (e) {
            setError(errorMessage(e, "Could not cancel the deletion."));
          } finally {
            setBusy(false);
          }
        }}
      />
      {error && <p role="alert" className="mt-1 text-[12px] text-bad">{error}</p>}
    </div>
  );
}

function Console() {
  const location = useLocation();
  const { status, me, email, signOut } = useAuth();
  const isChat = location.pathname.startsWith("/playground");

  // Hosted tenant mode: a signed-in owner is PINNED to their tenant,
  // no picker, no cross-tenant nav. Platform admins (and self-host,
  // where auth is disabled) keep the full surface.
  const isTenant = status === "signedIn" && me?.role === "owner";
  const pinnedId = isTenant ? me?.customer_id ?? null : null;

  return (
    <SelectedCustomerProvider pinnedId={pinnedId}>
      <div className="flex h-screen overflow-hidden">
        <Sidebar me={me} email={email} isTenant={isTenant} signedIn={status === "signedIn"} onSignOut={() => void signOut()} />

        {/* Content. The phone bar is fixed, so the scrolling column keeps
            room for it underneath (4.1 items 6 and 7). */}
        <main className={cn(
          "min-w-0 flex-1 pb-[calc(56px+env(safe-area-inset-bottom))] min-[720px]:pb-0",
          isChat ? "flex flex-col overflow-hidden" : "overflow-y-auto",
        )}>
          <PlanWallModal />
          <DeletionBanner />
          {!isTenant && <SelfCurationBanner />}
          {isChat ? (
            <>
              {isTenant && <GettingStarted collapsed />}
              {/* Chat sizes itself to this box (it was h-screen, which
                  overflowed once a banner sat above it). */}
              <div className="min-h-0 flex-1">
                <Routes>
                  <Route path="/playground" element={<ChatPlayground />} />
                </Routes>
              </div>
            </>
          ) : (
            <div className="mx-auto max-w-[1200px] px-4 py-[18px] min-[720px]:px-8 min-[720px]:py-7">
              {isTenant && <GettingStarted />}
              <Routes>
                <Route path="/" element={<BankBrowser />} />
                {/* Refine PR 4 (5.1 item 5): the reader is a drawer with
                    its own URL over the same list, so a link opens it
                    directly and closing keeps the list's filters. */}
                <Route path="/bank/:crystalId" element={<BankBrowser />} />
                <Route path="/knowledge" element={<KnowledgeManager />} />
                {/* Refine PR 5 (5.3 item 6): the review drawer has its own URL. */}
                <Route path="/knowledge/review/:docId" element={<KnowledgeManager />} />
                <Route path="/cognition" element={<Cognition />} />
                <Route path="/conflicts" element={<Conflicts />} />
                <Route path="/assumptions" element={<Assumptions />} />
                {!isTenant && <Route path="/critiques" element={<Critiques />} />}
                <Route path="/queries" element={<QueryLog />} />
                <Route path="/settings" element={<SettingsApi />} />
                <Route path="/billing" element={<Billing />} />
                {!isTenant && <Route path="/agents" element={<Agents />} />}
                {!isTenant && <Route path="/onboard" element={<Onboard />} />}
                <Route path="*" element={<Navigate to="/playground" replace />} />
              </Routes>
            </div>
          )}
        </main>
        <BottomBar />
      </div>
    </SelectedCustomerProvider>
  );
}

// The auth gate (Accounts Phase C). Presence-as-switch: without Firebase
// config the status is 'disabled' and the console renders exactly as it
// always has (self-host / local dev). With config: session → console,
// no session → Login, session-without-account → Onboarding.
function Gate() {
  const { status } = useAuth();
  if (status === "loading") {
    return (
      <div className="flex h-screen items-center justify-center bg-canvas">
        <CrystalMark className="h-8 w-8 animate-pulse" />
      </div>
    );
  }
  if (status === "signedOut") return <Login />;
  if (status === "needsSignup") return <OnboardingSetup />;
  // L2-S5b: Claude's connector flow lands here — intercepted before the
  // console shell. Signed-out users hit Login above with the URL (and
  // every OAuth param) intact; new users complete the wizard first.
  if (window.location.pathname.endsWith("/oauth/consent")) return <OAuthConsent />;
  return <Console />;
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
