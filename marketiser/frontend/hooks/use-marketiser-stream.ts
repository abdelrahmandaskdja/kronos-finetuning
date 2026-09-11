"use client";

import { useEffect } from "react";

import { resolveWsUrl } from "@/lib/api";
import type { CandlePoint, LiveStatus, MarketSnapshot, ModelFamilyRecord, ModelRecord, PredictionResult } from "@/lib/types";
import { useTerminalStore } from "@/store/terminal-store";

interface BootstrapPayload {
  status?: LiveStatus;
  watchlist?: MarketSnapshot[];
  models?: ModelRecord[];
  families?: ModelFamilyRecord[];
  logs?: PredictionResult[];
}

export function useMarketiserStream() {
  const applyBootstrap = useTerminalStore((state) => state.applyBootstrap);
  const setLiveStatus = useTerminalStore((state) => state.setLiveStatus);
  const upsertSnapshot = useTerminalStore((state) => state.upsertSnapshot);
  const upsertCandle = useTerminalStore((state) => state.upsertCandle);
  const pushPredictionLog = useTerminalStore((state) => state.pushPredictionLog);

  useEffect(() => {
    const wsUrl = resolveWsUrl();
    if (!wsUrl) return;

    let socket: WebSocket | null = null;
    let retryTimer: number | null = null;
    let disposed = false;

    const connect = () => {
      socket = new WebSocket(wsUrl);

      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as {
          type: string;
          payload: BootstrapPayload | LiveStatus | MarketSnapshot | CandlePoint | PredictionResult;
        };
        switch (message.type) {
          case "bootstrap":
            applyBootstrap(message.payload as BootstrapPayload);
            break;
          case "system.status":
            setLiveStatus(message.payload as LiveStatus);
            break;
          case "market.snapshot":
            upsertSnapshot(message.payload as MarketSnapshot);
            break;
          case "market.kline":
            upsertCandle(message.payload as CandlePoint);
            break;
          case "prediction.created":
            pushPredictionLog(message.payload as PredictionResult);
            break;
          default:
            break;
        }
      };

      socket.onclose = () => {
        if (disposed) return;
        retryTimer = window.setTimeout(connect, 4500);
      };

      socket.onerror = () => {};
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer) window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, [applyBootstrap, pushPredictionLog, setLiveStatus, upsertCandle, upsertSnapshot]);
}
