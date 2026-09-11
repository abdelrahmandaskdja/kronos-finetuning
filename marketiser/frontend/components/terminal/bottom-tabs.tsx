"use client";

import { BrainCircuit, History, LineChart, Newspaper, ShieldCheck } from "lucide-react";

import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ForecastOhlcvTable } from "@/components/terminal/forecast-ohlcv-table";
import { familyLabel, formatPercent, formatTime, signalTone } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

const TABS = [
  { key: "predictions", label: "Predictions", icon: History },
  { key: "performance", label: "Performance", icon: LineChart },
  { key: "backtest", label: "Backtest", icon: ShieldCheck },
  { key: "explanation", label: "Explanation", icon: BrainCircuit },
  { key: "news", label: "News + Sentiment", icon: Newspaper },
] as const;

export function BottomTabs() {
  const activeTab = useTerminalStore((state) => state.activeTab);
  const setActiveTab = useTerminalStore((state) => state.setActiveTab);
  const logs = useTerminalStore((state) => state.logs);
  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const models = useTerminalStore((state) => state.models);
  const selectedModelId = useTerminalStore((state) => state.selectedModelId);
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);
  const selectedModel = models.find((item) => item.id === selectedModelId);
  const visibleLogs = logs.filter((item) => item.symbol === selectedSymbol);

  return (
    <Card className="p-5">
      <div className="mb-5 flex flex-wrap gap-2">
        {TABS.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`inline-flex items-center gap-2 rounded-2xl border px-4 py-2.5 text-sm transition ${
                activeTab === tab.key
                  ? "border-[rgba(30,216,193,0.26)] bg-[rgba(30,216,193,0.12)] text-white"
                  : "border-white/8 bg-white/4 text-[var(--muted)] hover:text-white"
              }`}
            >
              <Icon className="size-4" />
              {tab.label}
            </button>
          );
        })}
      </div>

      {activeTab === "predictions" ? (
        <div className="space-y-3">
          {visibleLogs.length ? (
            <>
              {visibleLogs.map((item) => (
                <div key={item.id} className="grid gap-4 rounded-2xl border border-white/8 bg-black/16 p-4 lg:grid-cols-[1.2fr_0.8fr_0.7fr_0.8fr_1fr]">
                  <div>
                    <p className="text-sm font-medium text-white">{item.model_name}</p>
                    <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">
                      {item.symbol} · {item.horizon}
                    </p>
                  </div>
                  <div>
                    <Badge tone={signalTone(item.signal)}>{item.signal}</Badge>
                  </div>
                  <div className="text-sm text-white">{Math.round(item.confidence * 100)}%</div>
                  <div className={`text-sm ${item.predicted_move_pct >= 0 ? "text-green-300" : "text-red-300"}`}>
                    {formatPercent(item.predicted_move_pct)}
                  </div>
                  <div className="text-sm text-[var(--muted)]">{formatTime(item.generated_at)}</div>
                </div>
              ))}

              <ForecastOhlcvTable
                points={latestPrediction?.symbol === selectedSymbol ? latestPrediction.forecast_path : []}
                title={latestPrediction ? `${latestPrediction.model_name} active forecast` : "Active forecast OHLCV"}
                emptyLabel={`No active forecast OHLCV is available yet for ${selectedSymbol}.`}
              />
            </>
          ) : (
            <div className="rounded-2xl border border-dashed border-white/10 bg-black/14 p-6 text-sm text-[var(--muted)]">
              No predictions logged yet for {selectedSymbol}. Use the right sidebar to create the first live Marketiser signal.
            </div>
          )}
        </div>
      ) : null}

      {activeTab === "performance" ? (
        <div className="grid gap-4 xl:grid-cols-3">
          {models
            .filter((model) => model.benchmark)
            .map((model) => (
              <div key={model.id} className="rounded-2xl border border-white/8 bg-black/16 p-5">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-white">{model.display_name}</p>
                    <p className="mt-1 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">{model.branch}</p>
                  </div>
                  <Badge tone="live">{model.status}</Badge>
                </div>
                <div className="mt-4 grid grid-cols-3 gap-3">
                  <div>
                    <p className="text-xs text-[var(--muted)]">Winner metric</p>
                    <p className="mt-1 text-lg font-semibold text-white">
                      {model.benchmark?.winner_metric_value != null ? `${(model.benchmark.winner_metric_value * 100).toFixed(2)}%` : "--"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-[var(--muted)]">Close accuracy</p>
                    <p className="mt-1 text-lg font-semibold text-white">
                      {model.benchmark?.close_direction_accuracy != null
                        ? `${(model.benchmark.close_direction_accuracy * 100).toFixed(2)}%`
                        : model.benchmark?.accuracy != null
                          ? `${(model.benchmark.accuracy * 100).toFixed(2)}%`
                          : "--"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-[var(--muted)]">Path accuracy</p>
                    <p className="mt-1 text-lg font-semibold text-white">
                      {model.benchmark?.path_direction_accuracy != null
                        ? `${(model.benchmark.path_direction_accuracy * 100).toFixed(2)}%`
                        : "--"}
                    </p>
                  </div>
                </div>
                {model.benchmark?.winner_metric_name ? (
                  <p className="mt-4 text-sm text-white">
                    {model.benchmark.winner_metric_name}
                    {model.benchmark.selection_summary ? ` · ${model.benchmark.selection_summary}` : ""}
                  </p>
                ) : null}
                <p className="mt-4 text-sm leading-6 text-[var(--muted)]">{model.description}</p>
              </div>
            ))}
        </div>
      ) : null}

      {activeTab === "backtest" ? (
        <div className="rounded-2xl border border-white/8 bg-black/16 p-5">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Backtest snapshot</p>
          <h3 className="mt-3 text-xl font-medium text-white">{selectedModel?.display_name ?? "No model selected"}</h3>
          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="text-sm text-[var(--muted)]">Winner metric</p>
              <p className="mt-2 text-2xl font-semibold text-white">
                {selectedModel?.benchmark?.winner_metric_value != null
                  ? `${(selectedModel.benchmark.winner_metric_value * 100).toFixed(2)}%`
                  : "--"}
              </p>
              <p className="mt-1 text-xs text-[var(--muted)]">{selectedModel?.benchmark?.winner_metric_name ?? "Selection basis"}</p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="text-sm text-[var(--muted)]">Close accuracy</p>
              <p className="mt-2 text-2xl font-semibold text-white">
                {selectedModel?.benchmark?.close_direction_accuracy != null
                  ? `${(selectedModel.benchmark.close_direction_accuracy * 100).toFixed(2)}%`
                  : "--"}
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="text-sm text-[var(--muted)]">Window</p>
              <p className="mt-2 text-sm leading-6 text-white">{selectedModel?.benchmark?.evaluation_window ?? "Research window not supplied."}</p>
            </div>
          </div>
          {selectedModel?.benchmark?.selection_summary ? (
            <p className="mt-4 text-sm leading-7 text-white">{selectedModel.benchmark.selection_summary}</p>
          ) : null}
          <p className="mt-5 text-sm leading-7 text-[var(--muted)]">
            This MVP surfaces the repo-backed benchmark metadata inside the terminal. The next iteration can expose saved strategy curves,
            interval-specific confusion matrices, and model leaderboard comparisons directly in this tab.
          </p>
        </div>
      ) : null}

      {activeTab === "explanation" ? (
        <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-2xl border border-white/8 bg-black/16 p-5">
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Current signal explanation</p>
            <h3 className="mt-3 text-xl font-medium text-white">
              {latestPrediction ? `${latestPrediction.symbol} · ${latestPrediction.model_name}` : "Awaiting active forecast"}
            </h3>
            <p className="mt-4 text-sm leading-7 text-[var(--muted)]">
              {latestPrediction
                ? latestPrediction.reasoning
                : "A generated explanation will appear here once the active Marketiser forecast has been run."}
            </p>
          </div>
          <div className="rounded-2xl border border-white/8 bg-black/16 p-5">
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Family context</p>
            <h3 className="mt-3 text-xl font-medium text-white">{selectedModel ? familyLabel(selectedModel.family) : "No family selected"}</h3>
            <p className="mt-4 text-sm leading-7 text-[var(--muted)]">{selectedModel?.description ?? "Choose a model to see its architecture context."}</p>
          </div>
        </div>
      ) : null}

      {activeTab === "news" ? (
        <div className="rounded-2xl border border-dashed border-white/10 bg-black/14 p-6">
          <p className="font-mono text-xs uppercase tracking-[0.18em] text-[var(--muted)]">Future module</p>
          <h3 className="mt-3 text-xl font-medium text-white">News + sentiment integration</h3>
          <p className="mt-4 text-sm leading-7 text-[var(--muted)]">
            The backend and UI are reserved for a future extension that combines price-history models with news and sentiment pipelines,
            adds event markers to the terminal, and compares price-only vs fused models on the same dashboard.
          </p>
        </div>
      ) : null}
    </Card>
  );
}
