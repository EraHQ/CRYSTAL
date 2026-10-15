// Focus management shared by Drawer and Modal (docs/CONSOLE_UX_AUDIT.md
// Part 3): trap Tab inside the panel, close on Escape, and hand focus
// back to whatever opened it when the panel closes.
import { RefObject, useEffect, useRef } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
  'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

export function useFocusTrap(open: boolean, panel: RefObject<HTMLElement>, onClose: () => void) {
  const opener = useRef<Element | null>(null);
  // The latest onClose, read at event time: callers pass inline arrows,
  // and re-running the effect on every render would steal focus from
  // whatever the user is typing into inside the panel.
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement;
    const node = panel.current;
    if (!node) return;
    const first = node.querySelector<HTMLElement>(FOCUSABLE);
    (first ?? node).focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        close.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE)).filter((el) => el.offsetParent !== null || el === document.activeElement);
      if (items.length === 0) {
        e.preventDefault();
        node.focus();
        return;
      }
      const firstEl = items[0];
      const lastEl = items[items.length - 1];
      if (e.shiftKey && document.activeElement === firstEl) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    };
    node.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      node.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      const back = opener.current as HTMLElement | null;
      if (back && typeof back.focus === "function") back.focus();
    };
  }, [open, panel]);
}
