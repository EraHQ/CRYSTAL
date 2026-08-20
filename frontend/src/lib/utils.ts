import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

// Standard cn helper. Used by every component that conditionally
// composes Tailwind classes.
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Pretty-print a number of bytes / tokens / milliseconds. Returns "—"
// for null so the table cells render consistently.
export function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  return n.toLocaleString();
}

export function fmtSigned(n: number | null | undefined): string {
  if (n === null || n === undefined) return "—";
  if (n > 0) return `+${n.toLocaleString()}`;
  return n.toLocaleString();
}

export function fmtDateTime(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

// Truncate a string to maxLen chars, adding ellipsis if truncated.
export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}
