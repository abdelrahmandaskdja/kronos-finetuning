"use client";

import {
  Activity,
  Camera,
  Crosshair,
  LineChart,
  PencilLine,
  Ruler,
  Search,
  SlidersHorizontal,
  Smile,
  Type,
  ZoomIn,
} from "lucide-react";

import { TradingViewChart } from "@/components/terminal/tradingview-chart";
import { LiveStatusBadge } from "@/components/terminal/live-status-badge";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ASSET_META, CHART_INTERVALS } from "@/lib/constants";
import { formatCompact, formatPercent, formatPrice, signalTone } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

const MARKET_TABS = ["Chart", "Order Flow", "Trading Data", "Trading Analysis", "Kronos AI"] as const;
const TRADE_MODES = ["Spot", "Cross", "Isolated", "Grid"] as const;
const TOOLBAR_ICONS = [Crosshair, PencilLine, SlidersHorizontal, LineChart, Type, Smile, Ruler, ZoomIn] as const;
const ACTION_ICONS = [Activity, Search, Camera] as const;

function symbolPair(symbol: string) {
  if (symbol.endsWith("USDT")) {
    return `${symbol.slice(0, -4)}/USDT`;
  }
  return symbol;
}

function ForecastMiniChart() {
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);

  if (!latestPrediction?.forecast_path.length) {
    return (
      <div className="rounded-2xl border border-white/8 bg-[#101010] px-4 py-3 text-sm text-[#8d8d8d]">
        Run Kronos to paint a short forecast strip below the main chart.
      </div>
    );
  }

  const width = 260;
  const height = 72;
  const values = latestPrediction.forecast_path.map((point) => point.price);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const points = latestPrediction.forecast_path
    .map((point, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((point.price - min) / Math.max(max - min || 1, 1)) * (height - 16) - 8;
      return `${x},${y}`;
    })
    .join(" ");
  const stroke = latestPrediction.signal === "bearish" ? "#f6465d" : latestPrediction.signal === "bullish" ? "#0ecb81" : "#f0b90b";

  return (
    <div className="rounded-2xl border border-white/8 bg-[#101010] px-4 py-3">
      <div className="mb-2 flex items-center justify-between">
        <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#767676]">Kronos path</p>
        <p className="text-sm font-medium text-white">{formatPercent(latestPrediction.predicted_move_pct)}</p>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="h-[72px] w-full overflow-visible">
        <polyline
          fill="none"
          stroke={stroke}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points}
        />
      </svg>
    </div>
  );
}

export function ChartPanel() {
  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const chartInterval = useTerminalStore((state) => state.chartInterval);
  const setChartInterval = useTerminalStore((state) => state.setChartInterval);
  const liveStatus = useTerminalStore((state) => state.liveStatus);
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);
  const watchlist = useTerminalStore((state) => state.watchlist);
  const kronosPredLen = useTerminalStore((state) => state.kronosPredLen);
  const selectedSnapshot = watchlist.find((item) => item.symbol === selectedSymbol);
  const meta = ASSET_META[selectedSymbol as keyof typeof ASSET_META];
  const priceTone = (selectedSnapshot?.price_change_pct ?? 0) >= 0 ? "text-[#0ecb81]" : "text-[#f6465d]";

  return (
    <Card className="overflow-hidden rounded-[2rem] border border-[#1d1d1d] bg-[#050505] p-0 shadow-[0_32px_80px_rgba(0,0,0,0.55)]">
      <div className="border-b border-[#1b1b1b] bg-[#0a0a0a] px-5 py-4">
        <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex items-center gap-4">
            <div className="flex size-12 items-center justify-center rounded-full bg-[#f0b90b] text-lg font-bold text-black">
              {meta?.base?.slice(0, 1) ?? selectedSymbol.slice(0, 1)}
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h2 className="text-[1.9rem] font-semibold tracking-[-0.05em] text-white">{symbolPair(selectedSymbol)}</h2>
                <Badge tone={latestPrediction ? signalTone(latestPrediction.signal) : "warning"}>
                  {latestPrediction?.signal ?? "Live market"}
                </Badge>
              </div>
              <p className="mt-1 text-sm text-[#7b7b7b]">{meta?.name ?? selectedSymbol} perpetual-style market board</p>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-[#7b7b7b]">Last Price</p>
              <p className={`mt-1 text-[2rem] font-semibold tracking-[-0.05em] ${priceTone}`}>
                {selectedSnapshot ? formatPrice(selectedSnapshot.last_price) : "--"}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-[#7b7b7b]">24h Chg</p>
              <p className={`mt-2 text-lg font-medium ${priceTone}`}>
                {selectedSnapshot ? formatPercent(selectedSnapshot.price_change_pct) : "--"}
              </p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-[#7b7b7b]">24h High</p>
              <p className="mt-2 text-lg font-medium text-white">{selectedSnapshot ? formatPrice(selectedSnapshot.high_24h) : "--"}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-[#7b7b7b]">24h Low</p>
              <p className="mt-2 text-lg font-medium text-white">{selectedSnapshot ? formatPrice(selectedSnapshot.low_24h) : "--"}</p>
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-[#7b7b7b]">24h Vol</p>
              <p className="mt-2 text-lg font-medium text-white">
                {selectedSnapshot ? formatCompact(selectedSnapshot.quote_volume) : "--"}
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="border-b border-[#1b1b1b] bg-[#090909] px-5">
        <div className="flex flex-wrap items-center gap-6">
          {MARKET_TABS.map((tab) => (
            <button
              key={tab}
              type="button"
              className={`border-b-2 py-4 text-sm font-medium transition ${
                tab === "Chart" || tab === "Kronos AI"
                  ? "border-[#f0b90b] text-white"
                  : "border-transparent text-[#848484] hover:text-white"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="border-b border-[#1b1b1b] bg-[#080808] px-5 py-3">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex flex-wrap items-center gap-2">
            <span className="mr-3 text-sm text-[#787878]">Time</span>
            {CHART_INTERVALS.map((interval) => (
              <button
                key={interval}
                type="button"
                onClick={() => setChartInterval(interval)}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  chartInterval === interval ? "bg-[#1d1d1d] text-[#f0b90b]" : "text-[#9a9a9a] hover:bg-[#131313] hover:text-white"
                }`}
              >
                {interval}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2">
            {ACTION_ICONS.map((Icon, index) => (
              <button
                key={index}
                type="button"
                className="inline-flex size-9 items-center justify-center rounded-lg border border-[#232323] bg-[#0e0e0e] text-[#b1b1b1] transition hover:border-[#393939] hover:text-white"
              >
                <Icon className="size-4" />
              </button>
            ))}
            <LiveStatusBadge status={liveStatus} />
          </div>
        </div>
      </div>

      <div className="grid min-h-[48rem] grid-cols-[56px_minmax(0,1fr)] bg-[#050505]">
        <div className="border-r border-[#1b1b1b] bg-[#070707] py-3">
          <div className="flex flex-col items-center gap-2">
            {TOOLBAR_ICONS.map((Icon, index) => (
              <button
                key={index}
                type="button"
                className="inline-flex size-10 items-center justify-center rounded-xl text-[#a7a7a7] transition hover:bg-[#111111] hover:text-white"
              >
                <Icon className="size-4" />
              </button>
            ))}
          </div>
        </div>

        <div className="flex min-h-[48rem] flex-col">
          <div className="flex-1 bg-black">
            <TradingViewChart symbol={selectedSymbol} interval={chartInterval === "1d" ? "1D" : chartInterval} />
          </div>

          <div className="border-t border-[#1b1b1b] bg-[#080808] px-5 py-4">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
              <div className="flex flex-wrap items-center gap-2">
                {TRADE_MODES.map((mode) => (
                  <Button
                    key={mode}
                    size="sm"
                    variant={mode === "Spot" ? "secondary" : "ghost"}
                    className={mode === "Spot" ? "border-[#f0b90b]/30 bg-[#1a1507] text-[#f0b90b]" : "text-[#8c8c8c]"}
                  >
                    {mode}
                  </Button>
                ))}
              </div>

              <div className="grid gap-3 xl:grid-cols-[280px_1fr] xl:items-center">
                <ForecastMiniChart />
                <div className="rounded-2xl border border-[#1c1c1c] bg-[#101010] px-4 py-3">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#777]">Kronos runtime</p>
                      <p className="mt-1 text-sm text-white">
                        {latestPrediction ? latestPrediction.model_name : "Awaiting live prediction"}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#777]">Pred bars</p>
                      <p className="mt-1 text-sm text-white">{kronosPredLen}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#777]">Forecast</p>
                      <p className={`mt-1 text-sm font-medium ${latestPrediction?.signal === "bearish" ? "text-[#f6465d]" : "text-[#0ecb81]"}`}>
                        {latestPrediction ? formatPercent(latestPrediction.predicted_move_pct) : "--"}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[#777]">Confidence</p>
                      <p className="mt-1 text-sm text-white">
                        {latestPrediction ? `${Math.round(latestPrediction.confidence * 100)}%` : "--"}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}
