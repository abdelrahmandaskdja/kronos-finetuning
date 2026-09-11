"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";

declare global {
  interface Window {
    TradingView?: {
      onready?: (callback: () => void) => void;
      widget: new (config: Record<string, unknown>) => TradingViewAdvancedWidget;
    };
    Datafeeds?: {
      UDFCompatibleDatafeed: new (datafeedURL: string, updateFrequency?: number) => unknown;
    };
  }
}

interface TradingViewChartApi {
  createStudy?: (name: string, forceOverlay?: boolean, lock?: boolean) => void;
}

interface TradingViewAdvancedWidget {
  remove?: () => void;
  onChartReady?: (callback: () => void) => void;
  chart?: () => TradingViewChartApi;
}

export interface TradingViewDebugEntry {
  at: string;
  stage: string;
  detail?: string;
}

const TRADINGVIEW_HOSTED_VERSION = "31.1.0";
const ADVANCED_LIBRARY_SRC = `https://charting-library.tradingview-widget.com/versions/${TRADINGVIEW_HOSTED_VERSION}/charting_library/charting_library.standalone.js`;
const ADVANCED_LIBRARY_PATH = `https://charting-library.tradingview-widget.com/versions/${TRADINGVIEW_HOSTED_VERSION}/charting_library/`;
const UDF_DATAFEED_SRC = "https://charting-library.tradingview-widget.com/datafeeds/udf/dist/bundle.js";
const ADVANCED_FALLBACK_TIMEOUT_MS = 20_000;
const ADVANCED_INIT_DELAY_MS = 180;

let advancedScriptPromise: Promise<void> | null = null;

function resolveUdfDatafeedUrl() {
  if (typeof window === "undefined") {
    return "http://localhost:3000/api/marketiser/tv";
  }
  const origin = new URL(window.location.origin);
  if (origin.hostname === "127.0.0.1" || origin.hostname === "::1") {
    origin.hostname = "localhost";
  }
  origin.pathname = "/api/marketiser/tv";
  origin.search = "";
  origin.hash = "";
  return origin.toString();
}

function loadAdvancedChartingLibrary(report?: (stage: string, detail?: string) => void) {
  if (typeof window === "undefined") {
    return Promise.resolve();
  }
  if (window.TradingView?.widget && window.TradingView?.onready) {
    report?.("script:advanced:reuse-runtime");
    return Promise.resolve();
  }
  if (!advancedScriptPromise) {
    advancedScriptPromise = new Promise<void>((resolve, reject) => {
      const loadScript = (
        selector: string,
        src: string,
        errorMessage: string,
        stageBase: string,
      ) =>
        new Promise<void>((resolveScript, rejectScript) => {
          const existing = document.querySelector<HTMLScriptElement>(selector);
          if (existing) {
            if (existing.dataset.loaded === "true") {
              report?.(`${stageBase}:already-loaded`);
              resolveScript();
              return;
            }
            report?.(`${stageBase}:await-existing`, src);
            existing.addEventListener("load", () => resolveScript(), { once: true });
            existing.addEventListener("error", () => rejectScript(new Error(errorMessage)), { once: true });
            return;
          }

          report?.(`${stageBase}:append`, src);
          const script = document.createElement("script");
          script.src = src;
          script.async = true;
          if (selector.includes('data-tradingview-advanced')) {
            script.dataset.tradingviewAdvanced = "true";
          }
          if (selector.includes('data-tradingview-udf')) {
            script.dataset.tradingviewUdf = "true";
          }
          script.onload = () => {
            script.dataset.loaded = "true";
            report?.(`${stageBase}:loaded`, src);
            resolveScript();
          };
          script.onerror = () => {
            report?.(`${stageBase}:error`, src);
            rejectScript(new Error(errorMessage));
          };
          document.head.appendChild(script);
        });

      loadScript(
        'script[data-tradingview-advanced="true"]',
        ADVANCED_LIBRARY_SRC,
        "TradingView Advanced Charts failed to load.",
        "script:advanced",
      )
        .then(() =>
          loadScript(
            'script[data-tradingview-udf="true"]',
            UDF_DATAFEED_SRC,
            "TradingView UDF bundle failed to load.",
            "script:udf",
          ),
        )
        .then(() => resolve())
        .catch((error) => reject(error));
    });
  }
  return advancedScriptPromise;
}

async function waitForTradingViewReady(report?: (stage: string, detail?: string) => void) {
  if (typeof window === "undefined") {
    return;
  }
  if (!window.TradingView?.onready) {
    report?.("tradingview:onready:missing");
    return;
  }
  report?.("tradingview:onready:wait");
  await new Promise<void>((resolve) => {
    window.TradingView?.onready?.(() => {
      report?.("tradingview:onready:ready");
      resolve();
    });
  });
}

export function AdvancedTradingViewChart({
  symbol,
  interval,
  onFallbackRequested,
}: {
  symbol: string;
  interval: string;
  onFallbackRequested: (message: string, trace?: TradingViewDebugEntry[]) => void;
}) {
  const rawId = useId();
  const containerId = useMemo(() => `advanced-tradingview-${rawId.replace(/:/g, "")}`, [rawId]);
  const sessionId = useMemo(() => `tv-${rawId.replace(/:/g, "")}-${symbol}-${interval}`, [interval, rawId, symbol]);
  const traceRef = useRef<TradingViewDebugEntry[]>([]);
  const [debugTrace, setDebugTrace] = useState<TradingViewDebugEntry[]>([]);

  useEffect(() => {
    let cancelled = false;
    let widget: TradingViewAdvancedWidget | null = null;
    let ready = false;
    let bootstrapStarted = false;
    const udfUrl = resolveUdfDatafeedUrl();
    const cleanupCallbacks: Array<() => void> = [];
    const observedIframes = new WeakSet<HTMLIFrameElement>();

    const report = (stage: string, detail?: string) => {
      const entry: TradingViewDebugEntry = {
        at: new Date().toISOString(),
        stage,
        detail,
      };
      traceRef.current = [...traceRef.current.slice(-19), entry];
      if (!cancelled) {
        setDebugTrace(traceRef.current);
      }
      const suffix = detail ? ` :: ${detail}` : "";
      console.info(`[tv-debug][${sessionId}] ${stage}${suffix}`);
      void fetch("/api/client-tv-debug", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          symbol,
          interval,
          href: typeof window === "undefined" ? null : window.location.href,
          stage,
          detail,
          at: entry.at,
        }),
        keepalive: true,
      }).catch(() => undefined);
    };

    report("mount", `origin=${window.location.origin}`);
    report("udf:url", udfUrl);
    report(
      "runtime:flags",
      `widget=${typeof window.TradingView?.widget} onready=${typeof window.TradingView?.onready} udf=${typeof window.Datafeeds?.UDFCompatibleDatafeed}`,
    );

    const handleWindowError = (event: ErrorEvent) => {
      report("window:error", event.message || "unknown");
    };
    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      const reason = event.reason instanceof Error ? event.reason.message : String(event.reason ?? "unknown");
      report("window:unhandledrejection", reason);
    };
    window.addEventListener("error", handleWindowError);
    window.addEventListener("unhandledrejection", handleUnhandledRejection);
    cleanupCallbacks.push(() => {
      window.removeEventListener("error", handleWindowError);
      window.removeEventListener("unhandledrejection", handleUnhandledRejection);
    });

    const fallbackTimer = window.setTimeout(() => {
      if (!cancelled && !ready) {
        report("fallback:timeout", `waited=${ADVANCED_FALLBACK_TIMEOUT_MS}ms`);
        onFallbackRequested(
          `TradingView Advanced Charts hosted runtime v${TRADINGVIEW_HOSTED_VERSION} did not finish loading. The chart is falling back to the public widget.`,
          traceRef.current,
        );
      }
    }, ADVANCED_FALLBACK_TIMEOUT_MS);

    const bootstrapTimer = window.setTimeout(() => {
      if (cancelled) {
        return;
      }
      bootstrapStarted = true;
      report("bootstrap:start", `symbol=BINANCE:${symbol} interval=${interval}`);

      void loadAdvancedChartingLibrary(report)
        .then(async () => {
          await waitForTradingViewReady(report);
          if (cancelled) return;
          if (!window.TradingView?.widget || !window.Datafeeds?.UDFCompatibleDatafeed) {
            report("runtime:missing-constructors");
            throw new Error("TradingView Advanced Charts runtime is unavailable.");
          }

          const container = document.getElementById(containerId);
          if (!container) {
            report("container:missing", containerId);
            return;
          }
          report("container:ready", containerId);
          container.innerHTML = "";

          const observer = new MutationObserver(() => {
            const iframe = container.querySelector("iframe");
            report(
              "container:mutation",
              `children=${container.childElementCount} iframe=${iframe ? "yes" : "no"} src=${iframe?.getAttribute("src") ?? "-"}`,
            );
            if (iframe && !observedIframes.has(iframe)) {
              observedIframes.add(iframe);
              const inspectIframe = (stage: string) => {
                try {
                  const doc = iframe.contentDocument;
                  const bodyText = doc?.body?.innerText?.replace(/\s+/g, " ").slice(0, 180) ?? "";
                  const iframeWindow = iframe.contentWindow as (Window & {
                    doWhenApiIsReady?: unknown;
                    initializationFinished?: unknown;
                    tradingViewApi?: unknown;
                    urlParams?: unknown;
                    widgetReady?: unknown;
                  }) | null;
                  report(
                    stage,
                    [
                      `readyState=${doc?.readyState ?? "-"}`,
                      `title=${doc?.title ?? "-"}`,
                      `href=${iframe.contentWindow?.location?.href ?? "-"}`,
                      `body=${bodyText || "-"}`,
                      `urlParams=${typeof iframeWindow?.urlParams}`,
                      `widgetReady=${typeof iframeWindow?.widgetReady}`,
                      `doWhenApiIsReady=${typeof iframeWindow?.doWhenApiIsReady}`,
                      `tradingViewApi=${typeof iframeWindow?.tradingViewApi}`,
                      `initializationFinished=${typeof iframeWindow?.initializationFinished}`,
                    ].join(" "),
                  );
                } catch (error) {
                  report(stage, error instanceof Error ? error.message : "iframe_inspect_failed");
                }
              };
              const handleIframeLoad = () => inspectIframe("iframe:load");
              iframe.addEventListener("load", handleIframeLoad);
              cleanupCallbacks.push(() => iframe.removeEventListener("load", handleIframeLoad));
              const inspectOne = window.setTimeout(() => inspectIframe("iframe:inspect:1000ms"), 1_000);
              const inspectFive = window.setTimeout(() => inspectIframe("iframe:inspect:5000ms"), 5_000);
              cleanupCallbacks.push(() => window.clearTimeout(inspectOne));
              cleanupCallbacks.push(() => window.clearTimeout(inspectFive));
            }
          });
          observer.observe(container, { childList: true, subtree: true });
          cleanupCallbacks.push(() => observer.disconnect());

          report("widget:create");
          widget = new window.TradingView.widget({
            container: containerId,
            library_path: ADVANCED_LIBRARY_PATH,
            datafeed: new window.Datafeeds.UDFCompatibleDatafeed(udfUrl, 10_000),
            symbol: `BINANCE:${symbol}`,
            interval,
            locale: "en",
            timezone: "Etc/UTC",
            theme: "dark",
            autosize: true,
            fullscreen: false,
            debug: false,
            load_last_chart: false,
            disabled_features: ["use_localstorage_for_settings"],
            enabled_features: ["hide_left_toolbar_by_default", "iframe_loading_compatibility_mode"],
            favorites: {
              intervals: ["1", "5", "15", "60", "240", "1D"],
            },
            loading_screen: {
              backgroundColor: "#000000",
              foregroundColor: "#f0b90b",
            },
            overrides: {
              "paneProperties.background": "#000000",
              "paneProperties.vertGridProperties.color": "rgba(255,255,255,0.08)",
              "paneProperties.horzGridProperties.color": "rgba(255,255,255,0.08)",
              "paneProperties.crossHairProperties.color": "#6b6b6b",
              "scalesProperties.textColor": "#9f9f9f",
              "mainSeriesProperties.candleStyle.upColor": "#0ecb81",
              "mainSeriesProperties.candleStyle.downColor": "#f6465d",
              "mainSeriesProperties.candleStyle.borderUpColor": "#0ecb81",
              "mainSeriesProperties.candleStyle.borderDownColor": "#f6465d",
              "mainSeriesProperties.candleStyle.wickUpColor": "#0ecb81",
              "mainSeriesProperties.candleStyle.wickDownColor": "#f6465d",
              "mainSeriesProperties.showPriceLine": true,
              "symbolWatermarkProperties.transparency": 92,
            },
          });
          report("widget:created");
          [5_000, 10_000, 15_000].forEach((delay) => {
            const timer = window.setTimeout(() => {
              if (cancelled || ready) return;
              const iframe = container.querySelector("iframe");
              report(
                "widget:pending",
                `after=${delay}ms children=${container.childElementCount} iframe=${iframe ? "yes" : "no"} src=${iframe?.getAttribute("src") ?? "-"}`,
              );
            }, delay);
            cleanupCallbacks.push(() => window.clearTimeout(timer));
          });

          widget.onChartReady?.(() => {
            if (cancelled) return;
            ready = true;
            window.clearTimeout(fallbackTimer);
            report("widget:onChartReady");
            try {
              widget?.chart?.()?.createStudy?.("Volume", false, false);
              report("widget:study:volume:ok");
            } catch {
              report("widget:study:volume:error");
              return;
            }
          });
        })
        .catch((error) => {
          if (cancelled) return;
          window.clearTimeout(fallbackTimer);
          const message = error instanceof Error ? error.message : "TradingView Advanced Charts failed to initialize.";
          report("fallback:error", message);
          onFallbackRequested(message, traceRef.current);
        });
    }, ADVANCED_INIT_DELAY_MS);
    cleanupCallbacks.push(() => window.clearTimeout(bootstrapTimer));

    return () => {
      cancelled = true;
      if (!bootstrapStarted) {
        report("cleanup:before-bootstrap");
      }
      report("cleanup:start");
      window.clearTimeout(fallbackTimer);
      cleanupCallbacks.forEach((callback) => callback());
      widget?.remove?.();
      const container = document.getElementById(containerId);
      if (container) {
        container.innerHTML = "";
      }
    };
  }, [containerId, interval, onFallbackRequested, sessionId, symbol]);

  return (
    <div className="relative h-full w-full bg-black">
      <div id={containerId} className="h-full w-full bg-black" />
      {!debugTrace.some((entry) => entry.stage === "widget:onChartReady") ? (
        <div className="pointer-events-none absolute bottom-3 left-3 max-w-[32rem] rounded-xl border border-white/10 bg-black/70 px-3 py-2 font-mono text-[10px] uppercase tracking-[0.14em] text-white/70 backdrop-blur">
          {debugTrace.slice(-6).map((entry) => (
            <div key={`${entry.at}-${entry.stage}`} className="truncate">
              {entry.stage}
              {entry.detail ? ` :: ${entry.detail}` : ""}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
