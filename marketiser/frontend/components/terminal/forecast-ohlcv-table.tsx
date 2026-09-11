"use client";

import type { ForecastPoint } from "@/lib/types";
import { formatCompact, formatPrice, formatTime } from "@/lib/utils";

function candleDirection(point: ForecastPoint) {
  const open = point.open ?? point.price;
  const close = point.close ?? point.price;

  if (close > open) return "up";
  if (close < open) return "down";
  return "flat";
}

function candleMovePercent(point: ForecastPoint) {
  const open = point.open ?? point.price;
  const close = point.close ?? point.price;

  if (!open) return 0;
  return ((close - open) / open) * 100;
}

export function ForecastOhlcvTable({
  points,
  title = "Forecast OHLCV",
  emptyLabel = "Run a prediction to inspect the forecast bars.",
}: {
  points: ForecastPoint[];
  title?: string;
  emptyLabel?: string;
}) {
  if (!points.length) {
    return (
      <div className="rounded-2xl border border-dashed border-white/10 bg-black/14 p-4 text-sm text-[var(--muted)]">
        {emptyLabel}
      </div>
    );
  }

  return (
    <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <p className="text-sm font-medium text-white">{title}</p>
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">{points.length} bars</p>
      </div>
      <div className="overflow-x-auto">
        <table className="min-w-full border-separate border-spacing-y-2 text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
              <th className="pr-4">Bar</th>
              <th className="pr-4">Time</th>
              <th className="pr-4">Direction</th>
              <th className="pr-4">Move</th>
              <th className="pr-4">Open</th>
              <th className="pr-4">High</th>
              <th className="pr-4">Low</th>
              <th className="pr-4">Close</th>
              <th className="pr-0">Volume</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => {
              const direction = candleDirection(point);
              const movePercent = candleMovePercent(point);
              const directionClass =
                direction === "up"
                  ? "border-emerald-500/20 bg-emerald-500/10 text-emerald-300"
                  : direction === "down"
                    ? "border-rose-500/20 bg-rose-500/10 text-rose-300"
                    : "border-white/10 bg-white/5 text-white";

              return (
                <tr key={`${point.timestamp}-${point.index}`} className="text-white">
                  <td className="pr-4 font-mono text-[12px] text-[var(--muted)]">{point.index}</td>
                  <td className="pr-4 whitespace-nowrap">{formatTime(point.timestamp)}</td>
                  <td className="pr-4">
                    <span
                      className={`inline-flex min-w-[4.75rem] justify-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${directionClass}`}
                    >
                      {direction}
                    </span>
                  </td>
                  <td
                    className={`pr-4 font-medium ${
                      movePercent > 0 ? "text-emerald-300" : movePercent < 0 ? "text-rose-300" : "text-white"
                    }`}
                  >
                    {movePercent >= 0 ? "+" : ""}
                    {movePercent.toFixed(2)}%
                  </td>
                  <td className="pr-4">{formatPrice(point.open ?? point.price)}</td>
                  <td className="pr-4">{formatPrice(point.high ?? point.price)}</td>
                  <td className="pr-4">{formatPrice(point.low ?? point.price)}</td>
                  <td className="pr-4 font-medium">{formatPrice(point.close ?? point.price)}</td>
                  <td className="pr-0">{formatCompact(point.volume ?? 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
