"use client";

import { create } from "zustand";

import { CHART_INTERVALS, FORECAST_HORIZONS, HORIZON_MODEL_DEFAULTS } from "@/lib/constants";
import type { CandlePoint, LiveStatus, MarketSnapshot, ModelFamilyRecord, ModelRecord, PredictionResult, SystemSummary } from "@/lib/types";

type ActiveTab = "predictions" | "performance" | "backtest" | "explanation" | "news";
type Density = "comfortable" | "dense";

const DEFAULT_STATUS: LiveStatus = {
  state: "offline",
  detail: "Waiting for backend",
  updated_at: new Date(0).toISOString(),
};

function dedupeLogs(items: PredictionResult[]) {
  const seen = new Set<string>();
  return items.filter((item) => {
    if (seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
}

function candleKey(symbol: string, interval: string) {
  return `${symbol}:${interval}`;
}

function sortCandles(candles: CandlePoint[]) {
  const deduped = new Map<number, CandlePoint>();
  for (const candle of candles) {
    const time = new Date(candle.open_time).getTime();
    if (!Number.isFinite(time)) continue;
    deduped.set(time, candle);
  }
  return [...deduped.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([, candle]) => candle);
}

function foundationRuntimePatch(
  model: ModelRecord | undefined,
  currentLookback: number,
  currentPredLen: number,
) {
  if (!model || model.scope !== "foundation") {
    return {
      kronosLookback: currentLookback,
      kronosPredLen: currentPredLen,
    };
  }

  const contextLength = model.context_length ?? currentLookback;
  return {
    kronosLookback: Math.max(16, Math.min(model.default_lookback ?? currentLookback, contextLength)),
    kronosPredLen: Math.max(1, model.default_pred_len ?? currentPredLen),
  };
}

interface TerminalState {
  selectedSymbol: string;
  chartInterval: (typeof CHART_INTERVALS)[number];
  forecastHorizon: (typeof FORECAST_HORIZONS)[number];
  selectedModelId: string;
  kronosLookback: number;
  kronosPredLen: number;
  liveStatus: LiveStatus;
  watchlist: MarketSnapshot[];
  candles: Record<string, CandlePoint[]>;
  models: ModelRecord[];
  modelFamilies: ModelFamilyRecord[];
  logs: PredictionResult[];
  latestPrediction: PredictionResult | null;
  systemSummary: SystemSummary | null;
  favorites: string[];
  activeTab: ActiveTab;
  density: Density;
  applyBootstrap: (payload: {
    status?: LiveStatus;
    watchlist?: MarketSnapshot[];
    models?: ModelRecord[];
    families?: ModelFamilyRecord[];
    logs?: PredictionResult[];
  }) => void;
  setSelectedSymbol: (symbol: string) => void;
  setChartInterval: (interval: (typeof CHART_INTERVALS)[number]) => void;
  setForecastHorizon: (horizon: (typeof FORECAST_HORIZONS)[number]) => void;
  setSelectedModelId: (modelId: string) => void;
  setKronosLookback: (lookback: number) => void;
  setKronosPredLen: (predLen: number) => void;
  setLiveStatus: (status: LiveStatus) => void;
  setModels: (models: ModelRecord[]) => void;
  setModelFamilies: (families: ModelFamilyRecord[]) => void;
  setWatchlist: (watchlist: MarketSnapshot[]) => void;
  upsertSnapshot: (snapshot: MarketSnapshot) => void;
  setCandles: (symbol: string, interval: string, candles: CandlePoint[]) => void;
  upsertCandle: (candle: CandlePoint) => void;
  setSystemSummary: (summary: SystemSummary) => void;
  setLatestPrediction: (prediction: PredictionResult | null) => void;
  pushPredictionLog: (prediction: PredictionResult) => void;
  setLogs: (logs: PredictionResult[]) => void;
  toggleFavorite: (symbol: string) => void;
  setActiveTab: (tab: ActiveTab) => void;
  toggleDensity: () => void;
}

export const useTerminalStore = create<TerminalState>((set) => ({
  selectedSymbol: "BTCUSDT",
  chartInterval: "1h",
  forecastHorizon: "1h",
  selectedModelId: HORIZON_MODEL_DEFAULTS["1h"],
  kronosLookback: 256,
  kronosPredLen: 16,
  liveStatus: DEFAULT_STATUS,
  watchlist: [],
  candles: {},
  models: [],
  modelFamilies: [],
  logs: [],
  latestPrediction: null,
  systemSummary: null,
  favorites: ["BTCUSDT"],
  activeTab: "predictions",
  density: "comfortable",
  applyBootstrap: (payload) =>
    set((state) => {
      const nextModels = payload.models ?? state.models;
      const nextWatchlist = payload.watchlist ?? state.watchlist;
      const preferredModel =
        nextModels.find((item) => item.id === HORIZON_MODEL_DEFAULTS[state.forecastHorizon]) ??
        nextModels.find((item) => item.supported_horizons.includes(state.forecastHorizon)) ??
        nextModels[0];
      const selectedSymbol = nextWatchlist.some((item) => item.symbol === state.selectedSymbol)
        ? state.selectedSymbol
        : nextWatchlist[0]?.symbol ?? state.selectedSymbol;
      const runtimePatch = foundationRuntimePatch(preferredModel, state.kronosLookback, state.kronosPredLen);
      return {
        liveStatus: payload.status ?? state.liveStatus,
        watchlist: nextWatchlist,
        models: nextModels,
        modelFamilies: payload.families ?? state.modelFamilies,
        logs: payload.logs ? dedupeLogs(payload.logs) : state.logs,
        selectedSymbol,
        selectedModelId: preferredModel?.id ?? state.selectedModelId,
        ...runtimePatch,
      };
    }),
  setSelectedSymbol: (selectedSymbol) => set({ selectedSymbol }),
  setChartInterval: (chartInterval) => set({ chartInterval }),
  setForecastHorizon: (forecastHorizon) =>
    set((state) => {
      const match =
        state.models.find((item) => item.id === HORIZON_MODEL_DEFAULTS[forecastHorizon]) ??
        state.models.find((item) => item.supported_horizons.includes(forecastHorizon)) ??
        state.models[0];
      const runtimePatch = foundationRuntimePatch(match, state.kronosLookback, state.kronosPredLen);
      return {
        forecastHorizon,
        selectedModelId: match?.id ?? state.selectedModelId,
        ...runtimePatch,
      };
    }),
  setSelectedModelId: (selectedModelId) =>
    set((state) => {
      const model = state.models.find((item) => item.id === selectedModelId);
      const nextForecastHorizon =
        model?.supported_horizons.includes(state.forecastHorizon)
          ? state.forecastHorizon
          : ((model?.supported_horizons[0] as (typeof FORECAST_HORIZONS)[number] | undefined) ?? state.forecastHorizon);
      return {
        selectedModelId,
        forecastHorizon: nextForecastHorizon,
        ...foundationRuntimePatch(model, state.kronosLookback, state.kronosPredLen),
      };
    }),
  setKronosLookback: (kronosLookback) => set({ kronosLookback }),
  setKronosPredLen: (kronosPredLen) => set({ kronosPredLen }),
  setLiveStatus: (liveStatus) => set({ liveStatus }),
  setModels: (models) =>
    set((state) => ({
      ...(() => {
        const nextSelectedModelId =
          models.find((item) => item.id === state.selectedModelId)?.id ??
          models.find((item) => item.id === HORIZON_MODEL_DEFAULTS[state.forecastHorizon])?.id ??
          models[0]?.id ??
          state.selectedModelId;
        const nextModel = models.find((item) => item.id === nextSelectedModelId);
        return {
          models,
          selectedModelId: nextSelectedModelId,
          ...foundationRuntimePatch(nextModel, state.kronosLookback, state.kronosPredLen),
        };
      })(),
    })),
  setModelFamilies: (modelFamilies) => set({ modelFamilies }),
  setWatchlist: (watchlist) =>
    set((state) => ({
      watchlist,
      selectedSymbol: watchlist.some((item) => item.symbol === state.selectedSymbol)
        ? state.selectedSymbol
        : watchlist[0]?.symbol ?? state.selectedSymbol,
      favorites: state.favorites.filter((symbol) => watchlist.some((item) => item.symbol === symbol)),
    })),
  upsertSnapshot: (snapshot) =>
    set((state) => {
      const exists = state.watchlist.some((item) => item.symbol === snapshot.symbol);
      return {
        watchlist: exists
          ? state.watchlist.map((item) => (item.symbol === snapshot.symbol ? snapshot : item))
          : [...state.watchlist, snapshot],
      };
    }),
  setCandles: (symbol, interval, candles) =>
    set((state) => ({
      candles: {
        ...state.candles,
        [candleKey(symbol, interval)]: sortCandles(candles).slice(-240),
      },
    })),
  upsertCandle: (candle) =>
    set((state) => {
      const key = candleKey(candle.symbol, candle.interval);
      const current = state.candles[key] ?? [];
      const next = [...current];
      const existingIndex = next.findIndex((item) => item.open_time === candle.open_time);
      if (existingIndex >= 0) {
        next[existingIndex] = candle;
      } else {
        next.push(candle);
      }
      const sorted = sortCandles(next);
      return {
        candles: {
          ...state.candles,
          [key]: sorted.slice(-240),
        },
      };
    }),
  setSystemSummary: (systemSummary) => set({ systemSummary }),
  setLatestPrediction: (latestPrediction) => set({ latestPrediction }),
  pushPredictionLog: (prediction) =>
    set((state) => ({
      latestPrediction:
        prediction.symbol === state.selectedSymbol &&
        prediction.horizon === state.forecastHorizon &&
        prediction.model_id === state.selectedModelId
          ? prediction
          : state.latestPrediction,
      logs: dedupeLogs([prediction, ...state.logs]).slice(0, 30),
    })),
  setLogs: (logs) => set({ logs: dedupeLogs(logs).slice(0, 30) }),
  toggleFavorite: (symbol) =>
    set((state) => ({
      favorites: state.favorites.includes(symbol)
        ? state.favorites.filter((item) => item !== symbol)
        : [...state.favorites, symbol],
    })),
  setActiveTab: (activeTab) => set({ activeTab }),
  toggleDensity: () =>
    set((state) => ({
      density: state.density === "comfortable" ? "dense" : "comfortable",
    })),
}));
