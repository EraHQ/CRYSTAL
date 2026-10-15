import { ReactNode, useId, useRef } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useFocusTrap } from "./focus";
import { IconButton } from "./IconButton";

// Part 3: centered, 460px, same focus rules as Drawer. Used by the plan
// wall and the key reveal.
export function Modal({ open, onClose, title, children, footer, className }: {
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div aria-hidden className="absolute inset-0 bg-canvas/70" onClick={onClose} />
      <div ref={panel} role="dialog" aria-modal="true" aria-labelledby={`${id}-title`} tabIndex={-1}
        className={cn("relative w-full max-w-[460px] rounded-2xl border border-line bg-surface shadow-pop outline-none", className)}>
        <div className="flex items-center justify-between gap-3 px-5 pt-4">
          <h2 id={`${id}-title`} className="min-w-0 text-[16px] font-semibold text-fg">{title}</h2>
          <IconButton label="Close" icon={X} onClick={onClose} />
        </div>
        <div className="px-5 py-4 text-[13.5px] text-fg-2">{children}</div>
        {footer && <div className="flex flex-wrap items-center justify-end gap-2 px-5 pb-4">{footer}</div>}
      </div>
    </div>,
    document.body,
  );
}
