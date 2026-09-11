import type { HTMLAttributes, PropsWithChildren } from "react";

import clsx from "clsx";

type Tone = "neutral" | "live" | "bullish" | "bearish" | "warning";

const toneClasses: Record<Tone, string> = {
  neutral: "border-white/10 bg-white/6 text-[var(--muted)]",
  live: "border-[rgba(30,216,193,0.24)] bg-[rgba(30,216,193,0.12)] text-[var(--accent)]",
  bullish: "border-green-500/20 bg-green-500/12 text-green-300",
  bearish: "border-red-500/20 bg-red-500/12 text-red-300",
  warning: "border-amber-500/20 bg-amber-500/12 text-amber-300",
};

export function Badge({
  children,
  className,
  tone = "neutral",
  ...props
}: PropsWithChildren<HTMLAttributes<HTMLDivElement>> & { tone?: Tone }) {
  return (
    <div
      className={clsx(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1.5 font-mono text-[11px] uppercase tracking-[0.22em]",
        toneClasses[tone],
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

