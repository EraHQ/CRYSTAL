import { AlertCircle, RotateCw } from "lucide-react";
import { errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

// Part 3 / 4.3: every page error goes through here. The message is the
// server's humane text (errorMessage reads the envelope), never raw
// JSON or "status statusText"; Retry appears only when the caller can
// actually retry.
export function ErrorBanner({ title = "Something went wrong", message, error, onRetry, className }: {
  title?: string;
  message?: string;
  error?: unknown;
  onRetry?: () => void;
  className?: string;
}) {
  const text = message ?? (error !== undefined ? errorMessage(error) : undefined);
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-lg border border-bad/40 bg-bad-soft px-4 py-3 text-[13px]", className)}>
      <AlertCircle aria-hidden className="mt-0.5 h-4 w-4 shrink-0 text-bad" />
      <div className="min-w-0 flex-1">
        <p className="font-medium text-bad">{title}</p>
        {text && <p className="mt-0.5 break-words text-fg-2">{text}</p>}
      </div>
      {onRetry && (
        <button type="button" onClick={onRetry}
          className="inline-flex shrink-0 items-center gap-1 rounded-md border border-line px-2 py-1 text-[12px] font-medium text-fg-2 hover:bg-inset hover:text-fg">
          <RotateCw aria-hidden className="h-3 w-3" /> Retry
        </button>
      )}
    </div>
  );
}
