import { FormEvent, ReactNode, useId, useState } from "react";
import { cn } from "@/lib/utils";

// Part 3: for destructive ACCOUNT actions. The consequence line, an
// input that must equal `word` (ERASE, DELETE, REGENERATE), a password
// field when the route needs step-up, and a button that stays disabled
// until both are valid.
export function TypedConfirm({ word, message, confirmLabel, onConfirm, onCancel, requirePassword = false, busy = false, error, className }: {
  word: string;
  message: ReactNode;
  confirmLabel: string;
  onConfirm: (password?: string) => void;
  onCancel?: () => void;
  requirePassword?: boolean;
  busy?: boolean;
  error?: string;
  className?: string;
}) {
  const [typed, setTyped] = useState("");
  const [password, setPassword] = useState("");
  const id = useId();
  const valid = typed === word && (!requirePassword || password.length > 0);
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (valid && !busy) onConfirm(requirePassword ? password : undefined);
  };
  return (
    <form onSubmit={submit} className={cn("mt-3 rounded-lg border border-bad/40 bg-bad-soft p-3 text-[13px]", className)}>
      <p className="text-fg">{message}</p>
      <label htmlFor={`${id}-word`} className="mt-3 block text-[12px] font-medium text-fg-2">
        Type <span className="font-mono font-semibold text-fg">{word}</span> to continue
      </label>
      <input id={`${id}-word`} value={typed} onChange={(e) => setTyped(e.target.value)} autoComplete="off" spellCheck={false}
        className="mt-1 w-full rounded-md border border-line bg-inset px-2.5 py-1.5 font-mono text-[13px] text-fg outline-none focus:border-accent" />
      {requirePassword && (
        <>
          <label htmlFor={`${id}-pw`} className="mt-3 block text-[12px] font-medium text-fg-2">Your password</label>
          <input id={`${id}-pw`} type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password"
            className="mt-1 w-full rounded-md border border-line bg-inset px-2.5 py-1.5 text-[13px] text-fg outline-none focus:border-accent" />
        </>
      )}
      {error && <p role="alert" className="mt-2 text-[12.5px] text-bad">{error}</p>}
      <div className="mt-3 flex items-center justify-end gap-2">
        {onCancel && (
          <button type="button" onClick={onCancel} disabled={busy}
            className="rounded-md border border-line px-2.5 py-1 text-[12.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50">
            Cancel
          </button>
        )}
        <button type="submit" disabled={!valid || busy}
          className="rounded-md bg-bad px-3 py-1 text-[12.5px] font-semibold text-canvas hover:bg-bad/90 disabled:cursor-not-allowed disabled:opacity-40">
          {busy ? "Working…" : confirmLabel}
        </button>
      </div>
    </form>
  );
}
