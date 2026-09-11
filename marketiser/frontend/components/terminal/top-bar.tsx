"use client";

import { Settings2 } from "lucide-react";

import { MarketiserLogo } from "@/components/brand/marketiser-logo";
import { LiveStatusBadge } from "@/components/terminal/live-status-badge";
import { Button } from "@/components/ui/button";
import { CHART_INTERVALS, FORECAST_HORIZONS } from "@/lib/constants";
import { useTerminalStore } from "@/store/terminal-store";

export function TopBar() {
  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const chartInterval = useTerminalStore((state) => state.chartInterval);
  const forecastHorizon = useTerminalStore((state) => state.forecastHorizon);
  const selectedModelId = useTerminalStore((state) => state.selectedModelId);
  const models = useTerminalStore((state) => state.models);
  const watchlist = useTerminalStore((state) => state.watchlist);
  const systemSummary = useTerminalStore((state) => state.systemSummary);
  const liveStatus = useTerminalStore((state) => state.liveStatus);
  const density = useTerminalStore((state) => state.density);
  const setSelectedSymbol = useTerminalStore((state) => state.setSelectedSymbol);
  const setChartInterval = useTerminalStore((state) => state.setChartInterval);
  const setForecastHorizon = useTerminalStore((state) => state.setForecastHorizon);
  const setSelectedModelId = useTerminalStore((state) => state.setSelectedModelId);
  const toggleDensity = useTerminalStore((state) => state.toggleDensity);

  const foundationModels = models.filter((model) => model.scope === "foundation");
  const finetunedModels = models.filter((model) => model.scope === "finetuned");
  const symbols = Array.from(
    new Set([
      ...watchlist.map((item) => item.symbol),
      ...(systemSummary?.symbols ?? []),
      selectedSymbol,
    ]),
  ).filter(Boolean);

  return (
    <header className="glass-panel rounded-[2rem] px-5 py-4">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <MarketiserLogo size="terminal" />

        <div className="flex flex-1 flex-wrap items-center gap-3 xl:justify-end">
          {symbols.length > 1 ? (
            <select
              value={selectedSymbol}
              onChange={(event) => setSelectedSymbol(event.target.value)}
              className="h-11 rounded-2xl border border-[var(--panel-border)] bg-white/6 px-4 text-sm text-white outline-none"
            >
              {symbols.map((symbol) => (
                <option key={symbol} value={symbol} className="bg-slate-900">
                  {symbol}
                </option>
              ))}
            </select>
          ) : (
            <div className="inline-flex h-11 items-center rounded-2xl border border-[var(--panel-border)] bg-white/6 px-4 font-mono text-sm uppercase tracking-[0.18em] text-white">
              {selectedSymbol}
            </div>
          )}

          <select
            value={chartInterval}
            onChange={(event) => setChartInterval(event.target.value as (typeof CHART_INTERVALS)[number])}
            className="h-11 rounded-2xl border border-[var(--panel-border)] bg-white/6 px-4 text-sm text-white outline-none"
          >
            {CHART_INTERVALS.map((interval) => (
              <option key={interval} value={interval} className="bg-slate-900">
                {interval}
              </option>
            ))}
          </select>

          <div className="flex items-center gap-1 rounded-2xl border border-[var(--panel-border)] bg-white/6 p-1">
            {FORECAST_HORIZONS.map((horizon) => (
              <button
                key={horizon}
                type="button"
                onClick={() => setForecastHorizon(horizon)}
                className={`rounded-xl px-3 py-2 text-xs font-mono uppercase tracking-[0.22em] transition ${
                  forecastHorizon === horizon ? "bg-white text-slate-950" : "text-[var(--muted)] hover:text-white"
                }`}
              >
                {horizon}
              </button>
            ))}
          </div>

          <select
            value={selectedModelId}
            onChange={(event) => setSelectedModelId(event.target.value)}
            className="h-11 min-w-72 rounded-2xl border border-[var(--panel-border)] bg-white/6 px-4 text-sm text-white outline-none"
          >
            {foundationModels.length ? (
              <optgroup label="Kronos foundation">
                {foundationModels.map((model) => (
                  <option key={model.id} value={model.id} className="bg-slate-900">
                    {model.display_name} · {model.supported_horizons.join(" / ")}
                  </option>
                ))}
              </optgroup>
            ) : null}
            {finetunedModels.length ? (
              <optgroup label="Finetuned Kronos">
                {finetunedModels.map((model) => (
                  <option key={model.id} value={model.id} className="bg-slate-900">
                    {model.supported_horizons.join(" / ")} · {model.display_name}
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>

          <LiveStatusBadge status={liveStatus} />

          <Button variant="secondary" onClick={toggleDensity}>
            <Settings2 className="size-4" />
            {density === "comfortable" ? "Dense view" : "Comfortable"}
          </Button>
        </div>
      </div>
    </header>
  );
}
