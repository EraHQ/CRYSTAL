// First-run onboarding (Accounts Phase C). A valid session with no
// account lands here: three signal fields → POST /v1/auth/signup →
// managed tenant provisioned → Key A revealed EXACTLY ONCE (copy it or
// lose it — hashed at rest) → enter the console.
import { useState } from "react";
import { Check, Copy } from "lucide-react";
import { api } from "@/lib/api";
import { useAuth } from "@/lib/auth";

const INDUSTRIES = [
  "Software / SaaS", "Healthcare", "Finance", "E-commerce",
  "Education", "Legal", "Other",
];
const MODELS = [
  { value: "claude-sonnet-5", label: "Sonnet", note: "Balanced — great default" },
  { value: "claude-haiku-4-5", label: "Haiku", note: "Fastest, most economical" },
  { value: "claude-opus-4-8", label: "Opus", note: "Deepest reasoning" },
];
const EXPERIENCE = [
  { value: "new", label: "New to AI agents" },
  { value: "some", label: "Built a few things" },
  { value: "pro", label: "Ship AI systems professionally" },
];

export function OnboardingSetup() {
  const { email, refreshMe, signOut } = useAuth();
  const [industry, setIndustry] = useState("");
  const [building, setBuilding] = useState("");
  const [experience, setExperience] = useState("");
  const [model, setModel] = useState("claude-sonnet-5");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const out = await api.signup({ industry, building, experience, model });
      if (out.api_key) {
        setApiKey(out.api_key); // the one-time reveal screen
      } else {
        await refreshMe(); // idempotent re-signup / admin bootstrap
      }
    } catch {
      setError("Could not create your workspace. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  const copyKey = async () => {
    if (!apiKey) return;
    await navigator.clipboard.writeText(apiKey);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  if (apiKey) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#0b0e17]">
        <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#10131d] p-8">
          <h1 className="mb-1 text-[17px] font-semibold text-white">
            Your workspace is ready
          </h1>
          <p className="mb-5 text-[13px] leading-relaxed text-gray-400">
            This is your CRYSTAL API key for SDK and API access. It is shown{" "}
            <span className="font-semibold text-gray-200">only this once</span>{" "}
            — we store a hash, not the key. You can chat in the console
            without it; you need it only for programmatic access.
          </p>
          <div className="mb-5 flex items-center gap-2 rounded-lg border border-white/10 bg-[#0b0e17] px-3 py-2.5">
            <code className="min-w-0 flex-1 truncate text-[12px] text-emerald-400">
              {apiKey}
            </code>
            <button
              onClick={() => void copyKey()}
              className="shrink-0 rounded-md p-1.5 text-gray-400 transition hover:bg-white/10 hover:text-white"
              title="Copy"
            >
              {copied ? <Check className="h-4 w-4 text-emerald-400" /> : <Copy className="h-4 w-4" />}
            </button>
          </div>
          <button
            onClick={() => void refreshMe()}
            className="w-full rounded-lg bg-[#6f72f7] px-4 py-2.5 text-[13px] font-semibold text-white transition hover:bg-[#5d60ee]"
          >
            Enter the console
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen items-center justify-center bg-[#0b0e17]">
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-[#10131d] p-8">
        <h1 className="mb-1 text-[17px] font-semibold text-white">
          Set up your workspace
        </h1>
        <p className="mb-6 text-[13px] text-gray-400">
          Signed in as <span className="text-gray-200">{email}</span>{" "}
          <button
            onClick={() => void signOut()}
            className="text-[#8487fb] hover:underline"
          >
            (switch)
          </button>
        </p>

        <form onSubmit={submit} className="space-y-4">
          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-gray-300">
              What industry are you in?
            </label>
            <select
              value={industry}
              onChange={(e) => setIndustry(e.target.value)}
              required
              className="w-full rounded-lg border border-white/10 bg-[#0b0e17] px-3 py-2.5 text-[13px] text-gray-200 outline-none focus:border-[#6f72f7]"
            >
              <option value="" disabled>Choose one…</option>
              {INDUSTRIES.map((i) => (
                <option key={i} value={i}>{i}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-gray-300">
              What are you building?
            </label>
            <textarea
              value={building}
              onChange={(e) => setBuilding(e.target.value)}
              rows={2}
              placeholder="A support agent that remembers every customer…"
              className="w-full resize-none rounded-lg border border-white/10 bg-[#0b0e17] px-3 py-2.5 text-[13px] text-gray-200 placeholder-gray-600 outline-none focus:border-[#6f72f7]"
            />
          </div>

          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-gray-300">
              Experience with AI agents
            </label>
            <div className="space-y-1.5">
              {EXPERIENCE.map((opt) => (
                <label
                  key={opt.value}
                  className="flex cursor-pointer items-center gap-2.5 rounded-lg border border-white/10 px-3 py-2 text-[13px] text-gray-300 transition hover:border-white/25"
                >
                  <input
                    type="radio"
                    name="experience"
                    value={opt.value}
                    checked={experience === opt.value}
                    onChange={() => setExperience(opt.value)}
                    className="accent-[#6f72f7]"
                  />
                  {opt.label}
                </label>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-[12px] font-medium text-gray-300">
              Model
            </label>
            <div className="grid grid-cols-3 gap-1.5">
              {MODELS.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => setModel(m.value)}
                  className={
                    model === m.value
                      ? "rounded-lg border border-[#6f72f7] bg-[#6f72f7]/15 px-2 py-2 text-left"
                      : "rounded-lg border border-white/10 px-2 py-2 text-left transition hover:border-white/25"
                  }
                >
                  <span className="block text-[12.5px] font-semibold text-gray-200">
                    {m.label}
                  </span>
                  <span className="block text-[10.5px] leading-tight text-gray-500">
                    {m.note}
                  </span>
                </button>
              ))}
            </div>
            <p className="mt-1 text-[11px] text-gray-500">
              You can change this any time in Settings.
            </p>
          </div>

          {error && (
            <p className="rounded-lg bg-red-500/10 px-3 py-2 text-[12px] text-red-400">
              {error}
            </p>
          )}
          <button
            type="submit"
            disabled={busy}
            className="w-full rounded-lg bg-[#6f72f7] px-4 py-2.5 text-[13px] font-semibold text-white transition hover:bg-[#5d60ee] disabled:opacity-50"
          >
            {busy ? "Creating your workspace…" : "Create workspace"}
          </button>
        </form>
      </div>
    </div>
  );
}
