import { NavLink, Route, Routes, Navigate, useLocation } from "react-router-dom";
import { Database, MessageSquare, ListOrdered, BookOpen, Brain, UserPlus, Activity, Scale } from "lucide-react";
import { CustomerSelector } from "@/components/CustomerSelector";
import { SelectedCustomerProvider } from "@/lib/selected-customer";
import { BankBrowser } from "@/pages/BankBrowser";
import { ChatPlayground } from "@/pages/ChatPlayground";
import { Cognition } from "@/pages/Cognition";
import { KnowledgeManager } from "@/pages/KnowledgeManager";
import { QueryLog } from "@/pages/QueryLog";
import { Onboard } from "@/pages/Onboard";
import { Agents } from "@/pages/Agents";
import { Conflicts } from "@/pages/Conflicts";
import { cn } from "@/lib/utils";

// Eight destinations in the sidebar (the modern chat-app shell). Agents (the
// CRYS window — live sessions, turn-by-turn activity, the background queue)
// is the rename + expansion of the former Activity tab. Conflicts is the
// Never-Idle Convergence surface (contradictions the bank found + the backlog).
const NAV_ITEMS = [
  { to: "/playground", label: "Chat", icon: MessageSquare, end: false },
  { to: "/knowledge", label: "Knowledge", icon: BookOpen, end: false },
  { to: "/", label: "Crystal Bank", icon: Database, end: true },
  { to: "/cognition", label: "Cognition", icon: Brain, end: false },
  { to: "/conflicts", label: "Conflicts", icon: Scale, end: false },
  { to: "/queries", label: "Logs", icon: ListOrdered, end: false },
  { to: "/agents", label: "Agents", icon: Activity, end: false },
  { to: "/onboard", label: "Onboard", icon: UserPlus, end: false },
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

export default function App() {
  const location = useLocation();
  const isChat = location.pathname.startsWith("/playground");

  return (
    <SelectedCustomerProvider>
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
            {NAV_ITEMS.map((item) => (
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

          {/* Customer */}
          <div className="space-y-2 px-4 pb-4">
            <div className="crystal-divider mb-3" />
            <p className="px-1 text-[10px] font-medium uppercase tracking-[0.14em] text-gray-400">
              Customer
            </p>
            <CustomerSelector />
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
                <Route path="/queries" element={<QueryLog />} />
                <Route path="/agents" element={<Agents />} />
                <Route path="/onboard" element={<Onboard />} />
                <Route path="*" element={<Navigate to="/playground" replace />} />
              </Routes>
            </div>
          )}
        </main>
      </div>
    </SelectedCustomerProvider>
  );
}
