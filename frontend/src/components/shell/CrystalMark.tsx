// The gem. Fills come from the accent tokens so the mark follows the
// theme (4.1 item 5: brand is the gem plus "Crystal", nothing else).
export function CrystalMark({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="none" aria-hidden>
      <path d="M10 2L17 7V13L10 18L3 13V7L10 2Z" style={{ fill: "rgb(var(--ac))" }} />
      <path d="M10 2L17 7L10 10L3 7L10 2Z" style={{ fill: "rgb(var(--ac2))" }} />
      <path d="M10 10L17 7V13L10 18V10Z" style={{ fill: "rgb(var(--ac) / 0.75)" }} />
    </svg>
  );
}
