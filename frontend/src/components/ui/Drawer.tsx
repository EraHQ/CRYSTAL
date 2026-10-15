import { ReactNode, useId, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useFocusTrap } from "./focus";
import { IconButton } from "./IconButton";

// Part 3: right-side panel, 440px, full width under 720px. Focus is
// trapped inside, Escape and the backdrop close it, and focus returns
// to the opener. Its URL (deep links) is the caller's: the page routes
// to /thing/:id and renders <Drawer open>.
export function Drawer({ open, onClose, title, children, footer, className }: {
  open: boolean;
  onClose: () => void;
  title: ReactNode;
  children: ReactNode;
  footer?: ReactNode;
  className?: string;
}) {
  const panel = useRef<HTMLDivElement>(null);
  const id = useId();
  useFocusTrap(open, panel, onClose);
  if (!open) return null;
  return createPortal(
    <div className="fixed inset-0 z-50 flex justify-end">
      <div aria-hidden className="absolute inset-0 bg-canvas/70" onClick={onClose} />
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} tabIndex={-1}
        className={cn("relative flex h-full w-full flex-col border-l border-line bg-surface shadow-pop outline-none min-[720px]:w-[440px]", className)}>
        <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
          <h2 id={`${id}-title`} className="min-w-0 truncate text-[15px] font-semibold text-fg">{title}</h2>
          <IconButton label="Close" icon={X} onClick={onClose} />
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">{children}</div>
        {footer && <div className="border-t border-line px-4 py-3">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}
