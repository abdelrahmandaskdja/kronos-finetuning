"use client";

import { Clock3, Cpu, Gauge, History, Newspaper, Play, ShieldAlert, Sparkles } from "lucide-react";

import { ForecastOhlcvTable } from "@/components/terminal/forecast-ohlcv-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { familyLabel, formatPercent, formatPrice, formatTime, signalTone } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

export function PredictionSidebar({
  onRunPrediction,
  onToggleAutoRun,
  autoRunEnabled,
  isRunning,
  errorMessage,
}: {
  onRunPrediction: (autoRun?: boolean) => Promise<void>;
  onToggleAutoRun: () => void;
  autoRunEnabled: boolean;
  isRunning: boolean;
  errorMessage: string | null;
}) {
  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const selectedModelId = useTerminalStore((state) => state.selectedModelId);
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);
  const models = useTerminalStore((state) => state.models);
  const modelFamilies = useTerminalStore((state) => state.modelFamilies);
  const kronosLookback = useTerminalStore((state) => state.kronosLookback);
  const kronosPredLen = useTerminalStore((state) => state.kronosPredLen);
  const setSelectedModelId = useTerminalStore((state) => state.setSelectedModelId);
  const setKronosLookback = useTerminalStore((state) => state.setKronosLookback);
  const setKronosPredLen = useTerminalStore((state) => state.setKronosPredLen);

  const model = models.find((item) => item.id === selectedModelId);
  const family = modelFamilies.find((item) => item.key === model?.family);
  const foundationModels = models.filter((item) => item.scope === "foundation");
  const finetunedModels = models.filter((item) => item.scope === "finetuned");
  const activePrediction =
    latestPrediction && latestPrediction.model_id === selectedModelId && latestPrediction.symbol === selectedSymbol
      ? latestPrediction
      : null;

  const isFoundationModel = model?.scope === "foundation";
  const isRunnableModel = Boolean(model?.supports_live_inference);
  const contextCap = model?.context_length ?? 512;

  return (
    <div className="space-y-4">
      <Card className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--muted)]">Signal card</p>
            <h2 className="mt-2 text-2xl font-medium text-white">{selectedSymbol}</h2>
          </div>
          <Badge tone={activePrediction ? signalTone(activePrediction.signal) : "neutral"}>
            {activePrediction?.signal ?? (isRunnableModel ? "Standby" : "Metrics only")}
          </Badge>
        </div>

        <div className="mt-5 grid gap-3">
          <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
            <div className="mb-3 flex items-center gap-2 text-[var(--muted)]">
              <Gauge className="size-4" />
              <span className="text-sm">Forecast confidence</span>
            </div>
            <p className="text-3xl font-semibold tracking-[-0.05em] text-white">
              {activePrediction ? `${Math.round(activePrediction.confidence * 100)}%` : "--"}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="mb-2 text-sm text-[var(--muted)]">Predicted move</p>
              <p className="text-lg font-semibold text-white">
                {activePrediction ? formatPercent(activePrediction.predicted_move_pct) : "--"}
              </p>
            </div>
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="mb-2 text-sm text-[var(--muted)]">Last reference</p>
              <p className="text-lg font-semibold text-white">
                {activePrediction ? formatPrice(activePrediction.reference_price) : "--"}
              </p>
            </div>
          </div>

          <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-sm text-[var(--muted)]">Latest inference</p>
                <p className="mt-1 text-sm leading-6 text-white">
                  {activePrediction
                    ? activePrediction.reasoning
                    : isFoundationModel
                      ? "Select a Kronos foundation model and run a live prediction. The forecast will use the current market candles with your chosen lookback and pred_len."
                      : isRunnableModel
                        ? `This finetuned checkpoint can run live inference with its fixed training defaults. Current runtime uses lookback ${model?.default_lookback ?? "--"} and pred_len ${model?.default_pred_len ?? "--"}.`
                        : "This finetuned entry is shown as benchmark-only metadata."}
                </p>
              </div>
              <Clock3 className="mt-1 size-4 text-[var(--muted)]" />
            </div>
            {activePrediction ? (
              <p className="mt-3 font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">
                {formatTime(activePrediction.generated_at)}
              </p>
            ) : null}
          </div>
        </div>
      </Card>

      <Card className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--muted)]">Model output</p>
            <h3 className="mt-2 text-xl font-medium text-white">Predicted OHLCV</h3>
          </div>
          <Badge tone={activePrediction ? "live" : "neutral"}>{activePrediction ? "live output" : "idle"}</Badge>
        </div>

        <div className="mt-5">
          <ForecastOhlcvTable
            points={activePrediction?.forecast_path ?? []}
            title={activePrediction ? `${activePrediction.model_name} forecast` : "Forecast OHLCV"}
            emptyLabel="Run the selected model to inspect its predicted open, high, low, close, and volume bars."
          />
        </div>
      </Card>

      <Card className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--muted)]">Prediction controls</p>
            <h3 className="mt-2 text-xl font-medium text-white">Kronos runtime</h3>
          </div>
          <Sparkles className="size-5 text-[var(--accent)]" />
        </div>

        <div className="mt-5 space-y-4">
          <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
            <p className="text-sm text-[var(--muted)]">Current model</p>
            <div className="mt-2 flex items-center gap-2">
              <p className="text-base font-medium text-white">{model?.display_name ?? "Loading..."}</p>
              {model ? <Badge tone={isRunnableModel ? "live" : "warning"}>{model.scope}</Badge> : null}
            </div>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{model?.description}</p>
            {model?.params ? <p className="mt-2 text-xs uppercase tracking-[0.18em] text-[var(--muted)]">{model.params}</p> : null}
          </div>

          <div>
            <p className="mb-3 text-sm text-[var(--muted)]">Foundation variants</p>
            <div className="grid grid-cols-3 gap-2">
              {foundationModels.map((item) => (
                <Button
                  key={item.id}
                  variant={item.id === selectedModelId ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedModelId(item.id)}
                  className="w-full"
                >
                  {item.display_name.replace("Kronos-", "")}
                </Button>
              ))}
            </div>
          </div>

          <div>
            <p className="mb-3 text-sm text-[var(--muted)]">Finetuned checkpoints</p>
            <div className="grid grid-cols-3 gap-2">
              {finetunedModels.map((item) => (
                <Button
                  key={item.id}
                  variant={item.id === selectedModelId ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => setSelectedModelId(item.id)}
                  className="w-full"
                >
                  {item.supported_horizons[0] ?? item.display_name}
                </Button>
              ))}
            </div>
          </div>

          {isRunnableModel ? (
            <>
              {isFoundationModel ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <label className="rounded-2xl border border-white/8 bg-black/18 p-4">
                      <p className="text-sm text-[var(--muted)]">Lookback</p>
                      <input
                        type="number"
                        min={16}
                        max={contextCap}
                        step={1}
                        value={kronosLookback}
                        onChange={(event) => {
                          const nextValue = Number(event.target.value);
                          if (Number.isFinite(nextValue)) {
                            setKronosLookback(nextValue);
                          }
                        }}
                        className="mt-3 h-11 w-full rounded-2xl border border-white/10 bg-black/30 px-4 text-sm text-white outline-none"
                      />
                      <p className="mt-2 text-xs text-[var(--muted)]">Max context {contextCap}</p>
                    </label>

                    <label className="rounded-2xl border border-white/8 bg-black/18 p-4">
                      <p className="text-sm text-[var(--muted)]">Pred length</p>
                      <input
                        type="number"
                        min={1}
                        max={64}
                        step={1}
                        value={kronosPredLen}
                        onChange={(event) => {
                          const nextValue = Number(event.target.value);
                          if (Number.isFinite(nextValue)) {
                            setKronosPredLen(nextValue);
                          }
                        }}
                        className="mt-3 h-11 w-full rounded-2xl border border-white/10 bg-black/30 px-4 text-sm text-white outline-none"
                      />
                      <p className="mt-2 text-xs text-[var(--muted)]">Future bars to generate</p>
                    </label>
                  </div>

                  <div className="rounded-2xl border border-white/8 bg-black/18 p-4 text-sm leading-6 text-[var(--muted)]">
                    Live foundation runs use the repo’s real Kronos predictor on CPU with the latest cached market candles for the selected horizon.
                  </div>
                </>
              ) : (
                <div className="rounded-2xl border border-emerald-500/16 bg-emerald-500/8 p-4 text-sm leading-6 text-emerald-100">
                  This finetuned checkpoint runs live with its training-time defaults: lookback {model?.default_lookback ?? "--"} and
                  pred_len {model?.default_pred_len ?? "--"}. Benchmark cards below still show the saved evaluation metrics for this
                  checkpoint.
                </div>
              )}

              <div className="grid gap-3">
                <Button size="lg" onClick={() => onRunPrediction(false)} disabled={isRunning || !selectedModelId}>
                  <Play className="size-4" />
                  {isRunning ? "Running prediction" : isFoundationModel ? "Run Kronos prediction" : "Run finetuned prediction"}
                </Button>
                <Button size="md" variant={autoRunEnabled ? "secondary" : "ghost"} onClick={onToggleAutoRun}>
                  <History className="size-4" />
                  {autoRunEnabled ? "Disable auto-run" : "Enable auto-run"}
                </Button>
              </div>
            </>
          ) : (
            <div className="rounded-2xl border border-amber-500/16 bg-amber-500/8 p-4 text-sm leading-6 text-amber-100">
              This model is benchmark-only in Marketiser. Switch to `Kronos-mini`, `Kronos-small`, or `Kronos-base`
              above to run live inference with adjustable lookback and pred_len.
            </div>
          )}

          {errorMessage ? (
            <div className="rounded-2xl border border-red-500/16 bg-red-500/10 p-4 text-sm text-red-100">
              <div className="mb-2 flex items-center gap-2">
                <ShieldAlert className="size-4" />
                Prediction service notice
              </div>
              <p>{errorMessage}</p>
            </div>
          ) : null}
        </div>
      </Card>

      <Card className="p-5">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-[0.22em] text-[var(--muted)]">Model info</p>
            <h3 className="mt-2 text-xl font-medium text-white">
              {model?.scope === "finetuned" ? "Finetuned metrics" : "Kronos model info"}
            </h3>
          </div>
          <Cpu className="size-5 text-[var(--accent-2)]" />
        </div>

        <div className="mt-5 space-y-4">
          <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
            <p className="text-sm text-[var(--muted)]">Family</p>
            <p className="mt-2 text-base font-medium text-white">{model ? familyLabel(model.family) : "Loading..."}</p>
            <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{family?.summary ?? "Awaiting model family metadata."}</p>
          </div>

          {model?.scope === "foundation" ? (
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                <p className="text-sm text-[var(--muted)]">Context length</p>
                <p className="mt-2 text-lg font-semibold text-white">{model.context_length ?? "--"}</p>
              </div>
              <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                <p className="text-sm text-[var(--muted)]">Runtime defaults</p>
                <p className="mt-2 text-lg font-semibold text-white">
                  {model.default_lookback ?? "--"} / {model.default_pred_len ?? "--"}
                </p>
                <p className="mt-1 text-xs text-[var(--muted)]">lookback / pred_len</p>
              </div>
            </div>
          ) : model?.benchmark ? (
            <>
              {model.benchmark.winner_metric_value != null ? (
                <div className="rounded-2xl border border-emerald-500/16 bg-emerald-500/8 p-4">
                  <p className="text-sm text-emerald-100/80">Winner metric</p>
                  <p className="mt-2 text-lg font-semibold text-white">
                    {model.benchmark.winner_metric_name ?? "Selection metric"}: {(model.benchmark.winner_metric_value * 100).toFixed(2)}%
                  </p>
                  {model.benchmark.selection_summary ? (
                    <p className="mt-2 text-sm leading-6 text-emerald-50/85">{model.benchmark.selection_summary}</p>
                  ) : null}
                </div>
              ) : null}

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                  <p className="text-sm text-[var(--muted)]">Close accuracy</p>
                  <p className="mt-2 text-lg font-semibold text-white">
                    {model.benchmark.close_direction_accuracy != null
                      ? `${(model.benchmark.close_direction_accuracy * 100).toFixed(2)}%`
                      : model.benchmark.accuracy != null
                        ? `${(model.benchmark.accuracy * 100).toFixed(2)}%`
                        : "--"}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                  <p className="text-sm text-[var(--muted)]">Majority baseline</p>
                  <p className="mt-2 text-lg font-semibold text-white">
                    {model.benchmark.majority_baseline_accuracy != null
                      ? `${(model.benchmark.majority_baseline_accuracy * 100).toFixed(2)}%`
                      : "--"}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                  <p className="text-sm text-[var(--muted)]">Lookback / pred_len</p>
                  <p className="mt-2 text-lg font-semibold text-white">
                    {model.benchmark.lookback ?? "--"} / {model.benchmark.pred_len ?? "--"}
                  </p>
                </div>
                <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
                  <p className="text-sm text-[var(--muted)]">Close MAE</p>
                  <p className="mt-2 text-lg font-semibold text-white">
                    {model.benchmark.close_mae != null ? formatPrice(model.benchmark.close_mae) : "--"}
                  </p>
                </div>
              </div>
            </>
          ) : null}

          {model?.benchmark?.path_direction_accuracy != null ? (
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="text-sm text-[var(--muted)]">Path direction accuracy</p>
              <p className="mt-2 text-lg font-semibold text-white">{(model.benchmark.path_direction_accuracy * 100).toFixed(2)}%</p>
            </div>
          ) : null}

          {model?.benchmark?.evaluation_window ? (
            <div className="rounded-2xl border border-white/8 bg-black/18 p-4">
              <p className="text-sm text-[var(--muted)]">Evaluation window</p>
              <p className="mt-2 text-sm leading-6 text-white">{model.benchmark.evaluation_window}</p>
            </div>
          ) : null}

          <div className="rounded-2xl border border-dashed border-white/10 bg-black/14 p-4 text-sm leading-6 text-[var(--muted)]">
            <div className="mb-2 flex items-center gap-2 text-white">
              <Newspaper className="size-4 text-[var(--accent)]" />
              Product note
            </div>
            Marketiser now renders the primary market chart locally from the backend candle feed. The right rail keeps live Kronos runtime controls and finetuned benchmark context together.
          </div>
        </div>
      </Card>
    </div>
  );
}
