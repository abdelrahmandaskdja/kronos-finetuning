"use client";

import dynamic from "next/dynamic";

const LiveCandleChart = dynamic(
  () => import("@/components/terminal/live-candle-chart").then((mod) => mod.LiveCandleChart),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full min-h-[34rem] items-center justify-center bg-black">
        <div className="rounded-3xl border border-white/10 bg-black/30 px-6 py-5 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[var(--muted)]">Chart bootstrap</p>
          <p className="mt-3 text-lg font-medium text-white">Loading chart surface</p>
        </div>
      </div>
    ),
  },
);

export function TradingViewChart({
  symbol,
  interval,
}: {
  symbol: string;
  interval: string;
}) {
  return <LiveCandleChart symbol={symbol} interval={interval} />;
}
