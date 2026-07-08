import { NavLink, Route, Routes, Navigate, useLocation } from "react-router-dom";
import {
  Database, MessageSquare, ListOrdered, BookOpen, Brain,
  UserPlus, Activity, Scale, Settings as SettingsIcon, LogOut,
  MessageSquareWarning,
} from "lucide-react";
import { CustomerSelector } from "@/components/CustomerSelector";
import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { AuthProvider, useAuth } from "@/lib/auth";
import { BankBrowser } from "@/pages/BankBrowser";
import { ChatPlayground } from "@/pages/ChatPlayground";
import { Cognition } from "@/pages/Cognition";
import { KnowledgeManager } from "@/pages/KnowledgeManager";
import { QueryLog } from "@/pages/QueryLog";
import { Onboard } from "@/pages/Onboard";
import { Agents } from "@/pages/Agents";
import { Conflicts } from "@/pages/Conflicts";
import { Critiques } from "@/pages/Critiques";
import { Login } from "@/pages/Login";
import { OnboardingSetup } from "@/pages/OnboardingSetup";
import { SettingsApi } from "@/pages/SettingsApi";
import { cn } from "@/lib/utils";

// The sidebar destinations. `adminOnly` marks the cross-tenant / platform
// surfaces (Accounts Phase C): tenant principals see everything else,
// pinned to their own tenant by the backend guard AND the pinned
// SelectedCustomerProvider below. Onboard (manual customer minting) is
// superseded by self-signup for tenants; Agents pends its tenant-scoping
// audit (plan: page split held loosely).
const NAV_ITEMS = [
  { to: "/playground", label: "Chat", icon: MessageSquare, end: false, adminOnly: false },
  { to: "/knowledge", label: "Knowledge", icon: BookOpen, end: false, adminOnly: false },
  { to: "/", label: "Crystal Bank", icon: Database, end: true, adminOnly: false },
  { to: "/cognition", label: "Cognition", icon: Brain, end: false, adminOnly: false },
  { to: "/conflicts", label: "Conflicts", icon: Scale, end: false, adminOnly: false },
  { to: "/critiques", label: "Critiques", icon: MessageSquareWarning, end: false, adminOnly: true },
  { to: "/queries", label: "Logs", icon: ListOrdered, end: false, adminOnly: false },
  { to: "/agents", label: "Agents", icon: Activity, end: false, adminOnly: true },
  { to: "/onboard", label: "Onboard", icon: UserPlus, end: false, adminOnly: true },
  { to: "/settings", label: "Settings", icon: SettingsIcon, end: false, adminOnly: false },
];

function CrystalMark({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none">
      <path d="M10 2L17 7V13L10 18L3 13V7L10 2Z" fill="#6f72f7" />
      <path d="M10 2L17 7L10 10L3 7L10 2Z" fill="#8487fb" />
      <path d="M10 10L17 7V13L10 18V10Z" fill="#5d60ee" />
    </svg>
  );
}

function Console() {
  const location = useLocation();
  const { status, me, email, signOut } = useAuth();
  const isChat = location.pathname.startsWith("/playground");

  // Hosted tenant mode: a signed-in owner is PINNED to their tenant —
  // no picker, no cross-tenant nav. Platform admins (and self-host,
  // where auth is disabled) keep the full surface.
  const isTenant = status === "signedIn" && me?.role === "owner";
  const pinnedId = isTenant ? me?.customer_id ?? null : null;
  const items = NAV_ITEMS.filter((i) => !(isTenant && i.adminOnly));

  return (
    <SelectedCustomerProvider pinnedId={pinnedId}>
      <div className="flex h-screen overflow-hidden">
        {/* ── Sidebar ── */}
        <aside className="flex w-[230px] shrink-0 flex-col border-r border-gray-200 bg-[#10131d]">
          {/* Brand */}
          <div className="flex items-center gap-2.5 px-5 pt-5 pb-4">
            <div className="flex h-8 w-8 items-center justify-center rounded-xl bg-brand-50 shadow-glow">
              <CrystalMark className="h-5 w-5" />
            </div>
            <div className="leading-tight">
              <span className="block text-[14px] font-semibold tracking-tight text-gray-900">
                Crystal Cache
              </span>
              <span className="block text-[10px] font-medium uppercase tracking-[0.14em] text-gray-400">
                Inspector
              </span>
            </div>
          </div>

          <div className="crystal-divider mx-4 mb-3" />

          {/* Nav */}
          <nav className="flex-1 space-y-0.5 px-3">
            {items.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  cn(
                    "group relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                    isActive
                      ? "bg-brand-50 text-brand-700"
                      : "text-gray-500 hover:bg-gray-50 hover:text-gray-800"
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    {isActive && (
                      <span className="absolute left-0 top-1/2 h-4 w-0.5 -translate-y-1/2 rounded-full bg-prism-accent" />
                    )}
                    <item.icon
                      className={cn(
                        "h-4 w-4 transition-colors",
                        isActive ? "text-brand-400" : "text-gray-400 group-hover:text-gray-600"
                      )}
                    />
                    {item.label}
                  </>
                )}
              </NavLink>
            ))}
          </nav>

          {/* Customer / session */}
          <div className="space-y-2 px-4 pb-4">
            <div className="crystal-divider mb-3" />
            {isTenant ? (
              <>
                <p className="px-1 text-[10px] font-medium uppercase tracking-[0.14em] text-gray-400">
                  Workspace
                </p>
                <p className="truncate px-1 text-[12px] text-gray-300">{email}</p>
                <button
                  onClick={() => void signOut()}
                  className="flex w-full items-center gap-2 rounded-lg px-1 py-1.5 text-[12px] font-medium text-gray-500 transition hover:text-gray-200"
                >
                  <LogOut className="h-3.5 w-3.5" />
                  Sign out
                </button>
              </>
            ) : (
              <>
                <p className="px-1 text-[10px] font-medium uppercase tracking-[0.14em] text-gray-400">
                  Customer
                </p>
                <CustomerSelector />
                {status === "signedIn" && (
                  <button
                    onClick={() => void signOut()}
                    className="flex w-full items-center gap-2 rounded-lg px-1 py-1.5 text-[12px] font-medium text-gray-500 transition hover:text-gray-200"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                    Sign out ({email})
                  </button>
                )}
              </>
            )}
          </div>
        </aside>

        {/* ── Main ── */}
        <main className={cn("min-w-0 flex-1", isChat ? "overflow-hidden" : "overflow-y-auto")}>
          {isChat ? (
            <Routes>
              <Route path="/playground" element={<ChatPlayground />} />
            </Routes>
          ) : (
            <div className="mx-auto max-w-[1200px] px-8 py-8">
              <Routes>
                <Route path="/" element={<BankBrowser />} />
                <Route path="/knowledge" element={<KnowledgeManager />} />
                <Route path="/cognition" element={<Cognition />} />
                <Route path="/conflicts" element={<Conflicts />} />
                {!isTenant && <Route path="/critiques" element={<Critiques />} />}
                <Route path="/queries" element={<QueryLog />} />
                <Route path="/settings" element={<SettingsApi />} />
                {!isTenant && <Route path="/agents" element={<Agents />} />}
                {!isTenant && <Route path="/onboard" element={<Onboard />} />}
                <Route path="*" element={<Navigate to="/playground" replace />} />
              </Routes>
            </div>
          )}
        </main>
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
      <div className="flex h-screen items-center justify-center bg-[#0b0e17]">
        <CrystalMark className="h-8 w-8 animate-pulse" />
      </div>
    );
  }
  if (status === "signedOut") return <Login />;
  if (status === "needsSignup") return <OnboardingSetup />;
  return <Console />;
}

export default function App() {
  return (
    <AuthProvider>
      <Gate />
    </AuthProvider>
  );
}
