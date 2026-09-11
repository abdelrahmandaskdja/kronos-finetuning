"use client";

import { startTransition, useCallback, useEffect, useMemo, useState } from "react";

import { BottomTabs } from "@/components/terminal/bottom-tabs";
import { ChartPanel } from "@/components/terminal/chart-panel";
import { PredictionSidebar } from "@/components/terminal/prediction-sidebar";
import { TopBar } from "@/components/terminal/top-bar";
import { WatchlistPanel } from "@/components/terminal/watchlist-panel";
import { marketiserApi, resolveWsUrl } from "@/lib/api";
import type { PredictionResult } from "@/lib/types";
import { useMarketiserStream } from "@/hooks/use-marketiser-stream";
import { useTerminalStore } from "@/store/terminal-store";

export function TerminalShell() {
  useMarketiserStream();
  const wsUrl = resolveWsUrl();

  const selectedSymbol = useTerminalStore((state) => state.selectedSymbol);
  const forecastHorizon = useTerminalStore((state) => state.forecastHorizon);
  const selectedModelId = useTerminalStore((state) => state.selectedModelId);
  const kronosLookback = useTerminalStore((state) => state.kronosLookback);
  const kronosPredLen = useTerminalStore((state) => state.kronosPredLen);
  const models = useTerminalStore((state) => state.models);
  const density = useTerminalStore((state) => state.density);
  const applyBootstrap = useTerminalStore((state) => state.applyBootstrap);
  const setModels = useTerminalStore((state) => state.setModels);
  const setModelFamilies = useTerminalStore((state) => state.setModelFamilies);
  const setWatchlist = useTerminalStore((state) => state.setWatchlist);
  const setLogs = useTerminalStore((state) => state.setLogs);
  const setLiveStatus = useTerminalStore((state) => state.setLiveStatus);
  const setSystemSummary = useTerminalStore((state) => state.setSystemSummary);
  const setLatestPrediction = useTerminalStore((state) => state.setLatestPrediction);

  const [bootError, setBootError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [isRunningPrediction, setIsRunningPrediction] = useState(false);
  const [autoRunEnabled, setAutoRunEnabled] = useState(false);
  const selectedModel = useMemo(() => models.find((item) => item.id === selectedModelId), [models, selectedModelId]);

  useEffect(() => {
    let mounted = true;

    const bootstrap = async () => {
      try {
        const [summary, watchlist, models, families, logs] = await Promise.all([
          marketiserApi.getSystemSummary(),
          marketiserApi.getWatchlist(),
          marketiserApi.getModels(),
          marketiserApi.getModelFamilies(),
          marketiserApi.getPredictionLogs(20),
        ]);
        if (!mounted) return;
        setBootError(null);
        startTransition(() => {
          setSystemSummary(summary);
          setWatchlist(watchlist.items);
          setModels(models.items);
          setModelFamilies(families.items);
          setLogs(logs.items);
          setLiveStatus(summary.live_status);
          applyBootstrap({
            status: summary.live_status,
            watchlist: watchlist.items,
            models: models.items,
            families: families.items,
            logs: logs.items,
          });
        });
      } catch (error) {
        if (!mounted) return;
        setBootError(error instanceof Error ? error.message : "Failed to bootstrap Marketiser.");
        setLiveStatus({
          state: "offline",
          detail: "Backend API unreachable during bootstrap",
          updated_at: new Date().toISOString(),
        });
      }
    };

    void bootstrap();

    return () => {
      mounted = false;
    };
  }, [applyBootstrap, setLiveStatus, setLogs, setModelFamilies, setModels, setSystemSummary, setWatchlist]);

  useEffect(() => {
    if (!selectedModelId) return;
    let mounted = true;
    void marketiserApi
      .getLatestPrediction(selectedSymbol, forecastHorizon, selectedModelId)
      .then((response) => {
        if (!mounted) return;
        setLatestPrediction(response.item);
      })
      .catch(() => {
        if (!mounted) return;
        setLatestPrediction(null);
      });
    return () => {
      mounted = false;
    };
  }, [forecastHorizon, selectedModelId, selectedSymbol, setLatestPrediction]);

  useEffect(() => {
    let disposed = false;
    if (wsUrl) {
      return () => {
        disposed = true;
      };
    }

    const refresh = async () => {
      try {
        const [summary, watchlist, logs, latest] = await Promise.all([
          marketiserApi.getSystemSummary(),
          marketiserApi.getWatchlist(),
          marketiserApi.getPredictionLogs(20),
          selectedModelId
            ? marketiserApi.getLatestPrediction(selectedSymbol, forecastHorizon, selectedModelId)
            : Promise.resolve({ item: null }),
        ]);
        if (disposed) return;
        setBootError(null);
        startTransition(() => {
          setSystemSummary(summary);
          setWatchlist(watchlist.items);
          setLogs(logs.items);
          setLiveStatus(summary.live_status);
          setLatestPrediction(latest.item);
        });
      } catch (error) {
        if (disposed) return;
        setBootError(error instanceof Error ? error.message : "Polling Marketiser failed.");
        setLiveStatus({
          state: "offline",
          detail: "Backend polling lost",
          updated_at: new Date().toISOString(),
        });
      }
    };

    const timer = window.setInterval(() => {
      void refresh();
    }, 15_000);

    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [
    forecastHorizon,
    selectedModelId,
    selectedSymbol,
    setLatestPrediction,
    setLiveStatus,
    setLogs,
    setSystemSummary,
    setWatchlist,
    wsUrl,
  ]);

  const runPrediction = useCallback(
    async (autoRun = false) => {
      if (!selectedModelId) return;
      setPredictionError(null);
      setIsRunningPrediction(true);
      try {
        const result = await marketiserApi.runPrediction({
          symbol: selectedSymbol,
          horizon: forecastHorizon,
          model_id: selectedModelId,
          auto_run: autoRun,
          lookback: kronosLookback,
          pred_len: kronosPredLen,
        });
        startTransition(() => {
          setLatestPrediction(result as PredictionResult);
        });
      } catch (error) {
        setPredictionError(error instanceof Error ? error.message : "Prediction run failed.");
      } finally {
        setIsRunningPrediction(false);
      }
    },
    [forecastHorizon, kronosLookback, kronosPredLen, selectedModelId, selectedSymbol, setLatestPrediction],
  );

  useEffect(() => {
    if (!autoRunEnabled || !selectedModelId || !selectedModel?.supports_live_inference) return;
    const intervalMs = forecastHorizon === "5m" ? 20_000 : forecastHorizon === "1h" ? 40_000 : 60_000;
    const initialTimer = window.setTimeout(() => {
      void runPrediction(true);
    }, 0);
    const timer = window.setInterval(() => {
      void runPrediction(true);
    }, intervalMs);
    return () => {
      window.clearTimeout(initialTimer);
      window.clearInterval(timer);
    };
  }, [autoRunEnabled, forecastHorizon, runPrediction, selectedModel, selectedModelId]);

  const sidebarError = useMemo(() => predictionError ?? bootError, [bootError, predictionError]);

  return (
    <main className="terminal-surface min-h-screen px-4 py-4 text-[var(--foreground)] md:px-5 xl:px-6">
      <div className={`mx-auto flex max-w-[1800px] flex-col gap-4 ${density === "dense" ? "text-[15px]" : "text-base"}`}>
        <TopBar />

        <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)_360px]">
          <WatchlistPanel />

          <div className="space-y-4">
            <ChartPanel />
            <BottomTabs />
          </div>

          <PredictionSidebar
            onRunPrediction={runPrediction}
            onToggleAutoRun={() => setAutoRunEnabled((current) => !current)}
            autoRunEnabled={autoRunEnabled}
            isRunning={isRunningPrediction}
            errorMessage={sidebarError}
          />
        </div>
      </div>
    </main>
  );
}
