// Shared pieces for the three entry pages (Console Refine PR 12, 5.13 to
// 5.15): the card on the canvas, the brand row, the provider marks, and
// the legal footer. Terms and Privacy links appear only once their URLs
// exist (Q66: none yet; set LEGAL below).
import type { ReactNode } from "react";
import { CrystalMark } from "@/components/shell/CrystalMark";
import { cn } from "@/lib/utils";

// Fill these in when the pages exist; the footer links appear by itself.
export const LEGAL: { terms: string | null; privacy: string | null } = { terms: null, privacy: null };

export const BTN_PRIMARY = "inline-flex w-full items-center justify-center gap-2 rounded-md bg-accent px-4 py-2.5 text-[13.5px] font-medium text-on-accent hover:bg-accent-hover disabled:opacity-50";
export const BTN_SECONDARY = "inline-flex w-full items-center justify-center gap-2 rounded-md border border-line px-4 py-2.5 text-[13.5px] font-medium text-fg-2 hover:bg-inset hover:text-fg disabled:opacity-50";
export const INPUT = "w-full rounded-md border border-line bg-inset px-3 py-2.5 text-[13.5px] text-fg outline-none placeholder:text-fg-3 focus:border-accent";

export function EntryShell({ wide = false, children }: { wide?: boolean; children: ReactNode }) {
  return (
    <div className="flex min-h-screen justify-center overflow-y-auto bg-canvas px-4 py-8">
      <div className={cn("m-auto w-full rounded-2xl border border-line bg-surface p-6 min-[720px]:p-8", wide ? "max-w-2xl" : "max-w-md")}>
        {children}
      </div>
    </div>
  );
}

export function Brand({ sub }: { sub?: string }) {
  return (
    <div className="mb-5 flex items-center gap-2.5">
      <CrystalMark className="h-7 w-7" />
      <div>
        <p className="text-[16px] font-semibold tracking-tight text-fg">Crystal</p>
        {sub && <p className="text-[12px] text-fg-3">{sub}</p>}
      </div>
    </div>
  );
}

export function LegalFooter() {
  const hasLinks = !!(LEGAL.terms && LEGAL.privacy);
  return (
    <p className="mt-5 text-center text-[11.5px] leading-relaxed text-fg-3">
      {hasLinks && (
        <>By continuing you agree to the <a href={LEGAL.terms!} className="text-fg-2 underline hover:text-fg">Terms</a> and <a href={LEGAL.privacy!} className="text-fg-2 underline hover:text-fg">Privacy Policy</a>. </>
      )}
      Your memories stay yours: export or delete them any time.
    </p>
  );
}

export function GoogleMark() {
  return (
    <svg aria-hidden className="h-4 w-4" viewBox="0 0 48 48">
      <path fill="#FFC107" d="M43.6 20.1H42V20H24v8h11.3C33.7 32.7 29.2 36 24 36c-6.6 0-12-5.4-12-12s5.4-12 12-12c3.1 0 5.9 1.2 8 3l5.7-5.7C34.3 6.1 29.4 4 24 4 13 4 4 13 4 24s9 20 20 20 20-9 20-20c0-1.3-.1-2.6-.4-3.9z"/>
      <path fill="#FF3D00" d="M6.3 14.7l6.6 4.8C14.6 15.1 18.9 12 24 12c3.1 0 5.9 1.2 8 3l5.7-5.7C34.3 6.1 29.4 4 24 4 16.3 4 9.7 8.3 6.3 14.7z"/>
      <path fill="#4CAF50" d="M24 44c5.2 0 9.9-2 13.4-5.2l-6.2-5.2C29.2 35.1 26.7 36 24 36c-5.2 0-9.6-3.3-11.3-8l-6.5 5C9.5 39.6 16.2 44 24 44z"/>
      <path fill="#1976D2" d="M43.6 20.1H42V20H24v8h11.3c-.8 2.2-2.2 4.2-4.1 5.6l6.2 5.2C41 35.4 44 30.2 44 24c0-1.3-.1-2.6-.4-3.9z"/>
    </svg>
  );
}

export function GitHubMark() {
  return (
    <svg aria-hidden className="h-4 w-4" viewBox="0 0 16 16" fill="currentColor">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
    </svg>
  );
}
