import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

export interface ToastOptions {
  /** e.g. { label: "Undo", onClick } */
  action?: { label: string; onClick: () => void };
  /** ms; default 2500 */
  duration?: number;
}

interface ToastItem extends ToastOptions { id: number; message: string }

const ToastContext = createContext<((message: string, opts?: ToastOptions) => void) | null>(null);

// Part 3: bottom-center, auto-hide after 2.5 s, optional Undo, announced
// through role="status". One provider at the app root; pages call
// useToast()("Fact retired", { action: { label: "Undo", onClick } }).
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);
  const seq = useRef(0);
  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((x) => x.id !== id)), []);
  const toast = useCallback((message: string, opts: ToastOptions = {}) => {
    const id = ++seq.current;
    setItems((xs) => [...xs.slice(-2), { id, message, ...opts }]);
  }, []);
  const value = useMemo(() => toast, [toast]);
  return (
    <ToastContext.Provider value={value}>
      {children}
      <div aria-live="polite" className="pointer-events-none fixed inset-x-0 bottom-4 z-[60] flex flex-col items-center gap-2 px-4">
        {items.map((t) => <ToastView key={t.id} item={t} onDone={() => dismiss(t.id)} />)}
      </div>
    </ToastContext.Provider>
  );
}

function ToastView({ item, onDone }: { item: ToastItem; onDone: () => void }) {
  useEffect(() => {
    const h = window.setTimeout(onDone, item.duration ?? 2500);
    return () => window.clearTimeout(h);
  }, [item, onDone]);
  return (
    <div role="status" className="pointer-events-auto flex items-center gap-3 rounded-lg border border-line bg-surface px-3.5 py-2 text-[13px] text-fg shadow-pop">
      <span>{item.message}</span>
      {item.action && (
        <button type="button" onClick={() => { item.action?.onClick(); onDone(); }}
          className="rounded-md px-1.5 py-0.5 text-[12.5px] font-semibold text-accent hover:bg-accent-soft">
          {item.action.label}
        </button>
      )}
    </div>
  );
}

export function useToast() {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error("useToast needs a <ToastProvider> above it (src/main.tsx)");
  return ctx;
}
