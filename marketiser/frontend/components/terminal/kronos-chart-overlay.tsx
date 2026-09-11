"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import { formatCompact, formatPercent, formatPrice } from "@/lib/utils";
import { useTerminalStore } from "@/store/terminal-store";

const EMPTY_CANDLES: {
  open_time: string;
  high: number;
  low: number;
  close: number;
  volume?: number;
}[] = [];

function candleKey(symbol: string, interval: string) {
  return `${symbol}:${interval}`;
}

function normalizeInterval(interval: string) {
  return interval === "1D" ? "1d" : interval;
}

function formatShortTimestamp(value: string) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

type ForecastBar = {
  key: string;
  timestamp: string;
  x: number;
  open: number;
  close: number;
  high: number;
  low: number;
  volume: number;
  openY: number;
  closeY: number;
  highY: number;
  lowY: number;
  bodyTop: number;
  bodyHeight: number;
  volumeTop: number;
  volumeHeight: number;
  fill: string;
  stroke: string;
};

export function KronosChartOverlay({
  symbol,
  interval,
}: {
  symbol: string;
  interval: string;
}) {
  const normalizedInterval = normalizeInterval(interval);
  const latestPrediction = useTerminalStore((state) => state.latestPrediction);
  const candles = useTerminalStore((state) => state.candles[candleKey(symbol, normalizedInterval)] ?? EMPTY_CANDLES);

  const rootRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = rootRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      setSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    });

    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const overlay = useMemo(() => {
    if (
      !latestPrediction ||
      latestPrediction.symbol !== symbol ||
      latestPrediction.forecast_horizon !== normalizedInterval ||
      !latestPrediction.forecast_path.length ||
      size.width < 260 ||
      size.height < 220
    ) {
      return null;
    }

    const visibleCandles = candles.slice(-80);
    const lastClose = visibleCandles.at(-1)?.close ?? latestPrediction.last_price ?? latestPrediction.reference_price;
    const forecastHighs = latestPrediction.forecast_path.map((point) => point.high ?? point.close ?? point.price);
    const forecastLows = latestPrediction.forecast_path.map((point) => point.low ?? point.close ?? point.price);
    const highs = [...visibleCandles.map((candle) => candle.high), ...forecastHighs, lastClose];
    const lows = [...visibleCandles.map((candle) => candle.low), ...forecastLows, lastClose];

    const rawMin = Math.min(...lows);
    const rawMax = Math.max(...highs);
    const padding = Math.max((rawMax - rawMin) * 0.12, rawMax * 0.0024, 1e-6);
    const priceMin = rawMin - padding;
    const priceMax = rawMax + padding;

    const plotTop = 18;
    const axisBottom = size.height - 34;
    const plotLeft = 18;
    const plotRight = size.width - 76;
    const volumeGap = 10;
    const volumeHeight = Math.min(Math.max(size.height * 0.16, 46), 86);
    const volumeTop = axisBottom - volumeHeight;
    const priceBottom = volumeTop - volumeGap;
    const priceHeight = Math.max(priceBottom - plotTop, 1);

    const priceToY = (price: number) => {
      const ratio = (price - priceMin) / Math.max(priceMax - priceMin, 1e-9);
      return plotTop + (1 - ratio) * priceHeight;
    };

    const projectionWidth = Math.min(Math.max(size.width * 0.24, 190), 300);
    const projectionLeft = Math.max(plotLeft + 32, plotRight - projectionWidth);
    const step = projectionWidth / Math.max(latestPrediction.forecast_path.length, 1);
    const candleWidth = Math.max(Math.min(step * 0.54, 20), 8);
    const baselineX = projectionLeft;
    const signalColor =
      latestPrediction.signal === "bearish" ? "#f6465d" : latestPrediction.signal === "bullish" ? "#0ecb81" : "#f0b90b";

    const recentVolumes = visibleCandles.slice(-36).map((candle) => candle.volume ?? 0);
    const forecastVolumes = latestPrediction.forecast_path.map((point) => point.volume ?? 0);
    const volumeMax = Math.max(...recentVolumes, ...forecastVolumes, 1);
    const volumeToY = (value: number) => {
      const ratio = Math.max(value, 0) / volumeMax;
      return axisBottom - ratio * volumeHeight;
    };

    const forecastBars = latestPrediction.forecast_path.reduce<ForecastBar[]>((bars, point, index) => {
      const x = projectionLeft + step * (index + 0.55);
      const priorClose = bars.at(-1)?.close ?? lastClose;
      const open = point.open ?? priorClose;
      const close = point.close ?? point.price;
      const high = point.high ?? Math.max(open, close);
      const low = point.low ?? Math.min(open, close);
      const volume = Math.max(point.volume ?? 0, 0);
      const openY = priceToY(open);
      const closeY = priceToY(close);
      const highY = priceToY(high);
      const lowY = priceToY(low);
      const rising = close >= open;
      const volumeTopY = volumeToY(volume);

      bars.push({
        key: `${point.timestamp}-${index}`,
        timestamp: point.timestamp,
        x,
        open,
        close,
        high,
        low,
        volume,
        openY,
        closeY,
        highY,
        lowY,
        bodyTop: Math.min(openY, closeY),
        bodyHeight: Math.max(Math.abs(closeY - openY), 2.2),
        volumeTop: volumeTopY,
        volumeHeight: Math.max(axisBottom - volumeTopY, 1.5),
        fill: rising ? "#0ecb81" : "#f6465d",
        stroke: rising ? "#49dca0" : "#ff7d91",
      });
      return bars;
    }, []);

    const finalBar = forecastBars.at(-1);
    const finalPrice = finalBar?.close ?? lastClose;
    const finalY = finalBar?.closeY ?? priceToY(finalPrice);
    const finalX = finalBar?.x ?? baselineX;
    const priceTagWidth = 76;
    const priceTagX = size.width - priceTagWidth - 8;

    const labels = latestPrediction.forecast_path.flatMap((point, index, items) => {
      if (index !== 0 && index !== items.length - 1) {
        return [];
      }
      return [
        {
          key: `${point.timestamp}-${index}`,
          text: formatShortTimestamp(point.timestamp),
          x: projectionLeft + step * (index + 0.55),
        },
      ];
    });

    return {
      axisBottom,
      baselineX,
      candleWidth,
      finalBar,
      finalPrice,
      finalX,
      finalY,
      forecastBars,
      labels,
      predictedMovePct: latestPrediction.predicted_move_pct,
      priceBottom,
      priceTagX,
      priceTagWidth,
      priceToY,
      projectionLeft,
      projectionWidth,
      signalColor,
      lastClose,
      plotTop,
      volumeTop,
      volumeMax,
    };
  }, [candles, latestPrediction, normalizedInterval, size.height, size.width, symbol]);

  return (
    <div ref={rootRef} className="pointer-events-none absolute inset-0 z-10">
      {overlay ? (
        <svg viewBox={`0 0 ${size.width} ${size.height}`} className="h-full w-full">
          <defs>
            <linearGradient id="kronos-price-band" x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(20, 27, 38, 0.76)" />
              <stop offset="100%" stopColor="rgba(9, 13, 19, 0.18)" />
            </linearGradient>
            <linearGradient id="kronos-volume-band" x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(15, 19, 27, 0.86)" />
              <stop offset="100%" stopColor="rgba(11, 15, 23, 0.54)" />
            </linearGradient>
            <filter id="kronos-candle-glow" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur stdDeviation="2" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          <rect
            x={overlay.projectionLeft - 14}
            y={overlay.plotTop}
            width={overlay.projectionWidth + 14}
            height={overlay.priceBottom - overlay.plotTop}
            rx="12"
            fill="url(#kronos-price-band)"
            stroke="rgba(255,255,255,0.06)"
          />
          <rect
            x={overlay.projectionLeft - 14}
            y={overlay.volumeTop - 4}
            width={overlay.projectionWidth + 14}
            height={overlay.axisBottom - overlay.volumeTop + 4}
            rx="12"
            fill="url(#kronos-volume-band)"
            stroke="rgba(255,255,255,0.04)"
          />

          <line
            x1={overlay.projectionLeft - 14}
            x2={overlay.projectionLeft + overlay.projectionWidth}
            y1={overlay.volumeTop - 4}
            y2={overlay.volumeTop - 4}
            stroke="rgba(255,255,255,0.08)"
          />
          <line
            x1={overlay.baselineX}
            x2={overlay.baselineX}
            y1={overlay.plotTop}
            y2={overlay.axisBottom}
            stroke="rgba(240,185,11,0.48)"
            strokeDasharray="5 5"
          />
          <line
            x1={overlay.projectionLeft - 14}
            x2={overlay.priceTagX}
            y1={overlay.finalY}
            y2={overlay.finalY}
            stroke={overlay.signalColor}
            strokeDasharray="4 4"
            strokeOpacity="0.62"
          />
          <circle cx={overlay.baselineX} cy={overlay.priceToY(overlay.lastClose)} r="4" fill="#f5f5f5" opacity="0.95" />

          {overlay.forecastBars.map((bar, index) => (
            <g key={bar.key}>
              <line
                x1={bar.x + overlay.candleWidth / 2 + 4}
                x2={bar.x + overlay.candleWidth / 2 + 4}
                y1={overlay.plotTop}
                y2={overlay.axisBottom}
                stroke="rgba(255,255,255,0.045)"
                opacity={index === overlay.forecastBars.length - 1 ? 0 : 1}
              />
              <rect
                x={bar.x - overlay.candleWidth / 2}
                y={bar.volumeTop}
                width={overlay.candleWidth}
                height={bar.volumeHeight}
                rx="2"
                fill={bar.fill}
                opacity="0.28"
              />
              <g filter="url(#kronos-candle-glow)">
                <line x1={bar.x} x2={bar.x} y1={bar.highY} y2={bar.lowY} stroke={bar.stroke} strokeWidth="1.6" />
                <rect
                  x={bar.x - overlay.candleWidth / 2}
                  y={bar.bodyTop}
                  width={overlay.candleWidth}
                  height={bar.bodyHeight}
                  rx="2"
                  fill={bar.fill}
                  stroke={bar.stroke}
                  strokeWidth="1"
                />
              </g>
            </g>
          ))}

          <rect x={overlay.projectionLeft - 6} y={16} width={138} height={22} rx="11" fill="rgba(4,8,12,0.86)" />
          <text x={overlay.projectionLeft + 8} y={31} fill="#dcefff" fontSize="10" letterSpacing="1.5">
            KRONOS FORECAST
          </text>

          <rect x={overlay.projectionLeft - 6} y={44} width={126} height={22} rx="11" fill="rgba(4,8,12,0.78)" />
          <text x={overlay.projectionLeft + 8} y={59} fill={overlay.signalColor} fontSize="10.5">
            {formatPercent(overlay.predictedMovePct)}
          </text>

          <rect
            x={overlay.priceTagX}
            y={overlay.finalY - 11}
            width={overlay.priceTagWidth}
            height={22}
            rx="5"
            fill={overlay.signalColor}
          />
          <text x={overlay.priceTagX + 10} y={overlay.finalY + 4} fill="#ffffff" fontSize="10.5" fontWeight="600">
            {formatPrice(overlay.finalPrice)}
          </text>

          <text x={overlay.projectionLeft - 2} y={overlay.volumeTop + 12} fill="rgba(255,255,255,0.5)" fontSize="9">
            VOL {formatCompact(overlay.volumeMax)}
          </text>

          {overlay.labels.map((label) => (
            <text
              key={label.key}
              x={label.x}
              y={size.height - 12}
              fill="rgba(255,255,255,0.68)"
              fontSize="9.5"
              textAnchor="middle"
            >
              {label.text}
            </text>
          ))}
        </svg>
      ) : null}
    </div>
  );
}
