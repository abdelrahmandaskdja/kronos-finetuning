"use client";

import { Star } from "lucide-react";

import { Card } from "@/components/ui/card";
import { ASSET_META } from "@/lib/constants";
import { formatCompact, formatPercent, formatPrice } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

export function WatchlistPanel() {
  const watchlist = useTerminalStore((state) => state.watchlist);
  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const favorites = useTerminalStore((state) => state.favorites);
  const setSelectedSymbol = useTerminalStore((state) => state.setSelectedSymbol);
  const toggleFavorite = useTerminalStore((state) => state.toggleFavorite);

  const rows = [...watchlist].sort((left, right) => {
    const leftScore = favorites.includes(left.symbol) ? 1 : 0;
    const rightScore = favorites.includes(right.symbol) ? 1 : 0;
    return rightScore - leftScore || left.symbol.localeCompare(right.symbol);
  });

  return (
    <Card className="p-4">
      <div className="mb-4 flex items-center justify-between px-2">
        <div>
          <p className="font-mono text-xs uppercase tracking-[0.24em] text-[var(--muted)]">Watchlist</p>
          <h2 className="mt-2 text-xl font-medium text-white">Core market board</h2>
        </div>
      </div>

      <div className="space-y-2">
        {rows.map((item) => {
          const meta = ASSET_META[item.symbol as keyof typeof ASSET_META];
          const active = item.symbol === selectedSymbol;
          return (
            <div
              key={item.symbol}
              role="button"
              tabIndex={0}
              onClick={() => setSelectedSymbol(item.symbol)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setSelectedSymbol(item.symbol);
                }
              }}
              className={`w-full rounded-2xl border px-3 py-3 text-left transition ${
                active ? "border-[rgba(30,216,193,0.4)] bg-[rgba(30,216,193,0.08)]" : "border-white/6 bg-white/3 hover:border-white/14"
              }`}
            >
              <div className="mb-3 flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`text-sm font-semibold ${meta?.accent ?? "text-white"}`}>{meta?.base ?? item.symbol}</span>
                    <span className="text-xs text-[var(--muted)]">{meta?.name ?? item.symbol}</span>
                  </div>
                  <p className="mt-1 font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">{item.symbol}</p>
                </div>
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    toggleFavorite(item.symbol);
                  }}
                  className="rounded-full p-1 text-[var(--muted)] hover:text-amber-300"
                >
                  <Star className={`size-4 ${favorites.includes(item.symbol) ? "fill-amber-300 text-amber-300" : ""}`} />
                </button>
              </div>

              <div className="flex items-end justify-between">
                <div>
                  <p className="text-lg font-semibold text-white">{formatPrice(item.last_price)}</p>
                  <p className="mt-1 text-xs text-[var(--muted)]">Vol {formatCompact(item.quote_volume)}</p>
                </div>
                <div className={`text-right text-sm font-medium ${item.price_change_pct >= 0 ? "text-green-300" : "text-red-300"}`}>
                  <p>{formatPercent(item.price_change_pct)}</p>
                  <p className="mt-1 text-[11px] text-[var(--muted)]">24h move</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
