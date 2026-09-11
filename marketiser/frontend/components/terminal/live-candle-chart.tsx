"use client";

import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type {
  CandlestickData,
  HistogramData,
  IChartApi,
  ISeriesApi,
  LineData,
  MouseEventParams,
  Time,
  UTCTimestamp,
} from "lightweight-charts";

import { marketiserApi } from "@/lib/api";
import type { CandlePoint, ForecastPoint, PredictionResult } from "@/lib/types";
import { formatCompact, formatPercent, formatPrice, formatTime } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

const EMPTY_CANDLES: CandlePoint[] = [];
const INITIAL_CANDLE_LIMIT = 160;
const LIGHTWEIGHT_CHARTS_MIN_WIDTH = 960;
const LIGHTWEIGHT_CHARTS_MIN_HEIGHT = 560;
let lightweightChartsPromise: Promise<typeof import("lightweight-charts")> | null = null;

const POLL_MS_BY_INTERVAL: Record<string, number> = {
  "1m": 5_000,
  "5m": 8_000,
  "15m": 12_000,
  "1h": 18_000,
  "4h": 24_000,
  "1d": 30_000,
};

type HoverBar = {
  kind: "market" | "forecast";
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type ForecastBar = {
  time: UTCTimestamp;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  color: string;
};

function chartKey(symbol: string, interval: string) {
  return `${symbol}:${interval}`;
}

function normalizeInterval(interval: string) {
  return interval === "1D" ? "1d" : interval.toLowerCase();
}

function toUtcTimestamp(value: string) {
  return Math.floor(new Date(value).getTime() / 1000) as UTCTimestamp;
}

function numericTime(time: Time) {
  if (typeof time === "number") return time;
  if (typeof time === "string") return Math.floor(new Date(time).getTime() / 1000);
  return Date.UTC(time.year, time.month - 1, time.day) / 1000;
}

function loadLightweightCharts() {
  if (!lightweightChartsPromise) {
    lightweightChartsPromise = import("lightweight-charts");
  }
  return lightweightChartsPromise;
}

function timeKey(time: Time | undefined) {
  if (time == null) return null;
  if (typeof time === "number") return `utc:${time}`;
  if (typeof time === "string") return `date:${time}`;
  return `day:${time.year}-${time.month}-${time.day}`;
}

function priceFor(point: ForecastPoint, field: "open" | "high" | "low" | "close", fallback: number) {
  if (field === "open") return point.open ?? fallback;
  if (field === "high") return point.high ?? Math.max(point.open ?? fallback, point.close ?? point.price, point.price);
  if (field === "low") return point.low ?? Math.min(point.open ?? fallback, point.close ?? point.price, point.price);
  return point.close ?? point.price;
}

function buildForecastBars(prediction: PredictionResult | null) {
  if (!prediction?.forecast_path.length) {
    return [] as ForecastBar[];
  }

  let priorClose = prediction.reference_price || prediction.last_price;

  return prediction.forecast_path.map((point) => {
    const open = priceFor(point, "open", priorClose);
    const close = priceFor(point, "close", priorClose);
    const high = Math.max(priceFor(point, "high", priorClose), open, close);
    const low = Math.min(priceFor(point, "low", priorClose), open, close);
    const rising = close >= open;
    const bar: ForecastBar = {
      time: toUtcTimestamp(point.timestamp),
      timestamp: point.timestamp,
      open,
      high,
      low,
      close,
      volume: point.volume ?? 0,
      color: rising ? "#1ed8c1" : "#f97316",
    };
    priorClose = close;
    return bar;
  });
}

function isFiniteBar(value: { open: number; high: number; low: number; close: number; volume?: number }) {
  return (
    Number.isFinite(value.open) &&
    Number.isFinite(value.high) &&
    Number.isFinite(value.low) &&
    Number.isFinite(value.close) &&
    (value.volume == null || Number.isFinite(value.volume))
  );
}

function sanitizeCandles(candles: CandlePoint[]) {
  const deduped = new Map<number, CandlePoint>();
  for (const candle of candles) {
    if (!isFiniteBar(candle)) continue;
    const time = toUtcTimestamp(candle.open_time);
    if (!Number.isFinite(time)) continue;
    deduped.set(time, candle);
  }
  return [...deduped.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([, candle]) => candle);
}

function sanitizeForecastBars(bars: ForecastBar[]) {
  const deduped = new Map<number, ForecastBar>();
  for (const bar of bars) {
    if (!isFiniteBar(bar)) continue;
    if (!Number.isFinite(bar.time)) continue;
    deduped.set(bar.time, bar);
  }
  return [...deduped.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([, bar]) => bar);
}

function sortedUniqueHistogramData(items: HistogramData<Time>[]) {
  const deduped = new Map<number, HistogramData<Time>>();
  for (const item of items) {
    const sortTime = numericTime(item.time);
    if (!Number.isFinite(sortTime) || !Number.isFinite(item.value)) continue;
    deduped.set(sortTime, item);
  }
  return [...deduped.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([, item]) => item);
}

function projectionNotice(prediction: PredictionResult | null, interval: string) {
  if (!prediction) return null;
  const predictionInterval = normalizeInterval(prediction.forecast_horizon);
  const chartInterval = normalizeInterval(interval);
  if (predictionInterval === chartInterval) return null;
  return `Latest model output is ${prediction.forecast_horizon}. Switch the chart interval to view its forecast bars.`;
}

export function LiveCandleChart({
  symbol,
  interval,
}: {
  symbol: string;
  interval: string;
}) {
  const normalizedInterval = normalizeInterval(interval);
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);
  const selectedModelId = useTerminalStore((state) => state.selectedModelId);
  const candles = useTerminalStore((state) => state.candles[chartKey(symbol, normalizedInterval)] ?? EMPTY_CANDLES);
  const setCandles = useTerminalStore((state) => state.setCandles);

  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const marketSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const forecastSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const forecastLineSeriesRef = useRef<ISeriesApi<"Line"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const actualLookupRef = useRef<Map<string, HoverBar>>(new Map());
  const forecastLookupRef = useRef<Map<string, HoverBar>>(new Map());
  const fittedViewKeyRef = useRef<string | null>(null);
  const candlesRef = useRef<CandlePoint[]>(candles);
  const activePredictionRef = useRef<PredictionResult | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [chartInitError, setChartInitError] = useState<string | null>(null);
  const [hoveredBar, setHoveredBar] = useState<HoverBar | null>(null);

  const activePrediction = useMemo(() => {
    if (
      !latestPrediction ||
      latestPrediction.symbol !== symbol ||
      latestPrediction.model_id !== selectedModelId ||
      normalizeInterval(latestPrediction.forecast_horizon) !== normalizedInterval
    ) {
      return null;
    }

    return latestPrediction;
  }, [latestPrediction, normalizedInterval, selectedModelId, symbol]);

  const hiddenPredictionNotice = useMemo(() => {
    if (!latestPrediction || latestPrediction.symbol !== symbol || latestPrediction.model_id !== selectedModelId) {
      return null;
    }
    return projectionNotice(latestPrediction, normalizedInterval);
  }, [latestPrediction, normalizedInterval, selectedModelId, symbol]);

  const syncChartData = useCallback(
    (nextCandles: CandlePoint[], nextPrediction: PredictionResult | null, nextSymbol: string, nextInterval: string) => {
      const chart = chartRef.current;
      const marketSeries = marketSeriesRef.current;
      const forecastSeries = forecastSeriesRef.current;
      const forecastLineSeries = forecastLineSeriesRef.current;
      const volumeSeries = volumeSeriesRef.current;

      if (!chart || !marketSeries || !forecastSeries || !forecastLineSeries || !volumeSeries) {
        return;
      }

      try {
        const safeCandles = sanitizeCandles(nextCandles).slice(-INITIAL_CANDLE_LIMIT);
        const actualBars: CandlestickData<Time>[] = safeCandles.map((candle) => ({
          time: toUtcTimestamp(candle.open_time),
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
        }));

        const actualLookup = new Map<string, HoverBar>();
        const actualVolumes: HistogramData<Time>[] = safeCandles.map((candle) => {
          const rising = candle.close >= candle.open;
          const time = toUtcTimestamp(candle.open_time);
          actualLookup.set(timeKey(time) ?? candle.open_time, {
            kind: "market",
            timestamp: candle.open_time,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
          });

          return {
            time,
            value: candle.volume,
            color: rising ? "rgba(14,203,129,0.32)" : "rgba(246,70,93,0.32)",
          };
        });

        const latestActualTime =
          actualBars.length && typeof actualBars[actualBars.length - 1].time === "number"
            ? actualBars[actualBars.length - 1].time
            : null;
        const forecastBars = sanitizeForecastBars(buildForecastBars(nextPrediction)).filter(
          (bar) => latestActualTime == null || bar.time > latestActualTime,
        );
        const forecastLookup = new Map<string, HoverBar>();
        const forecastVolumes: HistogramData<Time>[] = forecastBars.map((bar) => {
          forecastLookup.set(timeKey(bar.time) ?? bar.timestamp, {
            kind: "forecast",
            timestamp: bar.timestamp,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
          });

          return {
            time: bar.time,
            value: bar.volume,
            color: bar.close >= bar.open ? "rgba(30,216,193,0.6)" : "rgba(249,115,22,0.6)",
          };
        });

        actualLookupRef.current = actualLookup;
        forecastLookupRef.current = forecastLookup;

        marketSeries.setData(actualBars);
        forecastSeries.setData(
          forecastBars.map((bar) => ({
            time: bar.time,
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
          })),
        );

        const forecastLineData: LineData<Time>[] =
          actualBars.length && forecastBars.length
            ? [
                {
                  time: actualBars[actualBars.length - 1].time,
                  value: actualBars[actualBars.length - 1].close,
                },
                ...forecastBars.map((bar) => ({
                  time: bar.time,
                  value: bar.close,
                })),
              ]
            : [];
        forecastLineSeries.setData(forecastLineData);
        volumeSeries.setData(sortedUniqueHistogramData([...actualVolumes, ...forecastVolumes]));

        chart.timeScale().applyOptions({
          rightOffset: Math.max(forecastBars.length + 3, 8),
        });

        const nextViewKey = `${nextSymbol}:${nextInterval}`;
        if (actualBars.length && fittedViewKeyRef.current !== nextViewKey) {
          chart.timeScale().fitContent();
          fittedViewKeyRef.current = nextViewKey;
        }
        setChartInitError(null);
      } catch (syncError) {
        setChartInitError(syncError instanceof Error ? syncError.message : "Failed to render chart data.");
      }
    },
    [],
  );

  useEffect(() => {
    candlesRef.current = candles;
    activePredictionRef.current = activePrediction;
  }, [activePrediction, candles]);

  useEffect(() => {
    let disposed = false;

    const loadCandles = async (showLoader: boolean) => {
      if (showLoader) setLoading(true);
      try {
        const response = await marketiserApi.getCandles(symbol, normalizedInterval, INITIAL_CANDLE_LIMIT);
        if (disposed) return;
        setCandles(symbol, normalizedInterval, response.items);
        setError(null);
      } catch (fetchError) {
        if (disposed) return;
        setError(fetchError instanceof Error ? fetchError.message : "Failed to load market candles.");
      } finally {
        if (!disposed) setLoading(false);
      }
    };

    void loadCandles(true);
    const timer = window.setInterval(() => {
      void loadCandles(false);
    }, POLL_MS_BY_INTERVAL[normalizedInterval] ?? 10_000);

    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }, [normalizedInterval, setCandles, symbol]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    let cleanup: (() => void) | null = null;

    void loadLightweightCharts()
      .then((lwc) => {
        if (disposed) return;

        const initialWidth = Math.max(container.clientWidth, LIGHTWEIGHT_CHARTS_MIN_WIDTH);
        const initialHeight = Math.max(container.clientHeight, LIGHTWEIGHT_CHARTS_MIN_HEIGHT);

        const chart = lwc.createChart(container, {
          width: initialWidth,
          height: initialHeight,
          layout: {
            background: { type: lwc.ColorType.Solid, color: "#000000" },
            textColor: "#8a94a7",
          },
          grid: {
            vertLines: { color: "rgba(255,255,255,0.06)" },
            horzLines: { color: "rgba(255,255,255,0.06)" },
          },
          crosshair: {
            mode: lwc.CrosshairMode.MagnetOHLC,
            vertLine: {
              color: "rgba(255,255,255,0.22)",
              style: lwc.LineStyle.Dashed,
              labelBackgroundColor: "#111827",
            },
            horzLine: {
              color: "rgba(255,255,255,0.18)",
              style: lwc.LineStyle.Dashed,
              labelBackgroundColor: "#111827",
            },
          },
          rightPriceScale: {
            borderVisible: false,
            scaleMargins: { top: 0.08, bottom: 0.22 },
          },
          timeScale: {
            borderVisible: false,
            rightOffset: 10,
            barSpacing: 12,
            minBarSpacing: 6,
            timeVisible: true,
            secondsVisible: false,
          },
          handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true,
          },
          handleScale: {
            axisPressedMouseMove: true,
            mouseWheel: true,
            pinch: true,
          },
        });

        const marketSeries = chart.addSeries(lwc.CandlestickSeries, {
          upColor: "#0ecb81",
          downColor: "#f6465d",
          borderUpColor: "#0ecb81",
          borderDownColor: "#f6465d",
          wickUpColor: "#0ecb81",
          wickDownColor: "#f6465d",
          priceLineVisible: true,
          lastValueVisible: true,
        });

        const forecastSeries = chart.addSeries(lwc.CandlestickSeries, {
          upColor: "rgba(30,216,193,0.92)",
          downColor: "rgba(249,115,22,0.92)",
          borderUpColor: "#1ed8c1",
          borderDownColor: "#f97316",
          wickUpColor: "#5eead4",
          wickDownColor: "#fdba74",
          priceLineVisible: false,
          lastValueVisible: false,
        });

        const forecastLineSeries = chart.addSeries(lwc.LineSeries, {
          color: "#67e8f9",
          lineWidth: 2,
          lineStyle: lwc.LineStyle.LargeDashed,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        const volumeSeries = chart.addSeries(lwc.HistogramSeries, {
          priceScaleId: "",
          priceFormat: { type: "volume" },
          base: 0,
          lastValueVisible: false,
          priceLineVisible: false,
        });

        marketSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.08, bottom: 0.24 },
        });

        volumeSeries.priceScale().applyOptions({
          scaleMargins: { top: 0.82, bottom: 0 },
        });

        chart.subscribeCrosshairMove((param: MouseEventParams<Time>) => {
          const key = timeKey(param.time);
          if (!key) {
            startTransition(() => setHoveredBar(null));
            return;
          }

          const marketHovered = actualLookupRef.current.get(key);
          const forecastHovered = forecastLookupRef.current.get(key);

          startTransition(() => setHoveredBar(marketHovered ?? forecastHovered ?? null));
        });

        const resizeObserver = new ResizeObserver((entries) => {
          const entry = entries[0];
          if (!entry) return;
          const { width, height } = entry.contentRect;
          chart.applyOptions({ width, height });
        });
        resizeObserver.observe(container);

        chartRef.current = chart;
        marketSeriesRef.current = marketSeries;
        forecastSeriesRef.current = forecastSeries;
        forecastLineSeriesRef.current = forecastLineSeries;
        volumeSeriesRef.current = volumeSeries;
        setChartInitError(null);
        syncChartData(candlesRef.current, activePredictionRef.current, symbol, normalizedInterval);

        cleanup = () => {
          resizeObserver.disconnect();
          chart.remove();
          chartRef.current = null;
          marketSeriesRef.current = null;
          forecastSeriesRef.current = null;
          forecastLineSeriesRef.current = null;
          volumeSeriesRef.current = null;
          actualLookupRef.current = new Map();
          forecastLookupRef.current = new Map();
          fittedViewKeyRef.current = null;
        };
      })
      .catch((initError) => {
        if (disposed) return;
        setChartInitError(initError instanceof Error ? initError.message : "Failed to initialize chart engine.");
      });

    return () => {
      disposed = true;
      cleanup?.();
    };
  }, [normalizedInterval, symbol, syncChartData]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      syncChartData(candles, activePrediction, symbol, normalizedInterval);
    });
    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [activePrediction, candles, normalizedInterval, symbol, syncChartData]);

  const displayBar = useMemo(() => {
    if (hoveredBar) return hoveredBar;
    if (activePrediction?.forecast_path.length) {
      const lastPoint = activePrediction.forecast_path[activePrediction.forecast_path.length - 1];
      const open = priceFor(lastPoint, "open", activePrediction.reference_price || activePrediction.last_price);
      const close = priceFor(lastPoint, "close", open);
      return {
        kind: "forecast" as const,
        timestamp: lastPoint.timestamp,
        open,
        high: priceFor(lastPoint, "high", open),
        low: priceFor(lastPoint, "low", open),
        close,
        volume: lastPoint.volume ?? 0,
      };
    }

    const lastCandle = candles[candles.length - 1];
    if (!lastCandle) return null;

    return {
      kind: "market" as const,
      timestamp: lastCandle.open_time,
      open: lastCandle.open,
      high: lastCandle.high,
      low: lastCandle.low,
      close: lastCandle.close,
      volume: lastCandle.volume,
    };
  }, [activePrediction, candles, hoveredBar]);

  const latestForecastMove =
    activePrediction?.forecast_path.length && activePrediction.forecast_path[activePrediction.forecast_path.length - 1]
      ? activePrediction.forecast_path[activePrediction.forecast_path.length - 1].pct_from_last
      : null;

  if (loading && !candles.length) {
    return (
      <div className="flex h-full min-h-[34rem] items-center justify-center bg-[radial-gradient(circle_at_top,rgba(30,216,193,0.1),transparent_45%)]">
        <div className="rounded-3xl border border-white/10 bg-black/30 px-6 py-5 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[var(--muted)]">Chart bootstrap</p>
          <p className="mt-3 text-lg font-medium text-white">Loading TradingView chart surface</p>
        </div>
      </div>
    );
  }

  if (error && !candles.length) {
    return (
      <div className="flex h-full min-h-[34rem] items-center justify-center bg-[linear-gradient(180deg,rgba(246,70,93,0.06),transparent)]">
        <div className="max-w-md rounded-3xl border border-red-500/20 bg-black/35 px-6 py-5 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-red-200/80">Feed issue</p>
          <p className="mt-3 text-lg font-medium text-white">The chart could not load candle data.</p>
          <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{error}</p>
        </div>
      </div>
    );
  }

  if (chartInitError) {
    return (
      <div className="flex h-full min-h-[34rem] items-center justify-center bg-[linear-gradient(180deg,rgba(246,70,93,0.06),transparent)]">
        <div className="max-w-md rounded-3xl border border-red-500/20 bg-black/35 px-6 py-5 text-center">
          <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-red-200/80">Chart init error</p>
          <p className="mt-3 text-lg font-medium text-white">The TradingView chart engine failed to initialize.</p>
          <p className="mt-2 text-sm leading-6 text-[var(--muted)]">{chartInitError}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full min-h-[34rem] bg-black">
      <div className="pointer-events-none absolute left-4 top-4 z-20 rounded-2xl border border-white/10 bg-black/70 px-4 py-3 backdrop-blur">
        <div className="flex items-center gap-2">
          <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-[var(--muted)]">{symbol}</p>
          <span className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
            {normalizedInterval}
          </span>
          {displayBar ? (
            <span
              className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.16em] ${
                displayBar.kind === "forecast"
                  ? "border-amber-400/20 bg-amber-400/10 text-amber-300"
                  : "border-emerald-400/20 bg-emerald-400/10 text-emerald-300"
              }`}
            >
              {displayBar.kind}
            </span>
          ) : null}
        </div>
        {displayBar ? (
          <>
            <p className="mt-2 text-sm text-white">{formatTime(displayBar.timestamp)}</p>
            <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span className="text-white/70">O {formatPrice(displayBar.open)}</span>
              <span className="text-white/70">H {formatPrice(displayBar.high)}</span>
              <span className="text-white/70">L {formatPrice(displayBar.low)}</span>
              <span className="font-medium text-white">C {formatPrice(displayBar.close)}</span>
              <span className="text-white/70">Vol {formatCompact(displayBar.volume)}</span>
            </div>
          </>
        ) : null}
      </div>

      <div className="pointer-events-none absolute right-4 top-4 z-20 max-w-[22rem] space-y-2">
        {activePrediction ? (
          <div className="rounded-2xl border border-cyan-400/18 bg-black/72 px-4 py-3 backdrop-blur">
            <p className="font-mono text-[11px] uppercase tracking-[0.18em] text-cyan-300">Kronos forecast</p>
            <p className="mt-1 text-sm font-medium text-white">{activePrediction.model_name}</p>
            <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
              <span className="text-white/75">{activePrediction.forecast_path.length} bars</span>
              <span className="text-white/75">{Math.round(activePrediction.confidence * 100)}% confidence</span>
              {latestForecastMove != null ? (
                <span className={latestForecastMove >= 0 ? "text-emerald-300" : "text-orange-300"}>
                  {formatPercent(latestForecastMove)}
                </span>
              ) : null}
            </div>
          </div>
        ) : null}

        {hiddenPredictionNotice ? (
          <div className="rounded-2xl border border-amber-400/18 bg-black/72 px-4 py-3 text-sm leading-6 text-amber-100 backdrop-blur">
            {hiddenPredictionNotice}
          </div>
        ) : null}

        {error && candles.length ? (
          <div className="rounded-2xl border border-red-500/18 bg-black/72 px-4 py-3 text-sm text-red-100 backdrop-blur">
            {error}
          </div>
        ) : null}
      </div>

      <div ref={containerRef} className="h-full min-h-[34rem] w-full" />

      <div className="pointer-events-none absolute bottom-3 left-4 z-20 rounded-full border border-white/8 bg-black/70 px-3 py-1.5 text-[11px] text-[var(--muted)] backdrop-blur">
        Charting with TradingView Lightweight Charts
      </div>
    </div>
  );
}
