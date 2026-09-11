import Link from "next/link";

import { MarketiserLogo } from "@/components/brand/marketiser-logo";
import { FeatureGrid } from "@/components/home/feature-grid";
import { ModelFamiliesSection } from "@/components/home/model-families-section";
import { Button } from "@/components/ui/button";

export default function Home() {
  return (
    <main className="relative isolate min-h-screen overflow-hidden bg-[var(--background)] text-[var(--foreground)]">
      <div className="marketiser-grid pointer-events-none absolute inset-0 opacity-60" />
      <div className="marketiser-orb absolute left-1/2 top-24 h-[26rem] w-[26rem] -translate-x-1/2 rounded-full bg-[radial-gradient(circle,rgba(30,216,193,0.18),rgba(7,16,24,0))]" />
      <div className="marketiser-orb absolute right-0 top-0 h-[32rem] w-[32rem] rounded-full bg-[radial-gradient(circle,rgba(245,158,11,0.12),rgba(7,16,24,0))]" />

      <section className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col px-6 py-8 lg:px-10">
        <header className="flex items-center justify-between border-b border-white/8 pb-5">
          <Link href="/" className="inline-flex">
            <MarketiserLogo size="terminal" />
          </Link>
          <Link href="/terminal">
            <Button variant="ghost">Open Trading Terminal</Button>
          </Link>
        </header>

        <div className="flex flex-1 flex-col items-center justify-center gap-12 py-16 text-center">
          <Link href="/terminal" className="group inline-flex flex-col items-center gap-8">
            <MarketiserLogo size="hero" />
            <div className="space-y-5">
              <p className="inline-flex rounded-full border border-[var(--panel-border)] bg-white/5 px-4 py-2 font-mono text-xs uppercase tracking-[0.28em] text-[var(--muted)]">
                BTCUSDT forecasting powered by Kronos models
              </p>
              <h1 className="max-w-5xl text-5xl font-semibold tracking-[-0.06em] text-white md:text-7xl">
                Real-time BTCUSDT market intelligence.
              </h1>
              <p className="mx-auto max-w-3xl text-balance text-lg leading-8 text-[var(--muted)] md:text-xl">
                Marketiser combines exchange-style terminal ergonomics with interval-aware Kronos forecasting,
                live BTCUSDT market data, and research-grade model controls across 5-minute, 1-hour, and 1-day horizons.
              </p>
            </div>
          </Link>

          <div className="flex flex-col gap-4 sm:flex-row">
            <Link href="/terminal">
              <Button size="lg" className="min-w-56">
                Enter Marketiser
              </Button>
            </Link>
            <Link href="/terminal">
              <Button size="lg" variant="secondary" className="min-w-56">
                Open Trading Terminal
              </Button>
            </Link>
          </div>

          <div className="grid w-full max-w-5xl grid-cols-2 gap-4 text-left md:grid-cols-4">
            {[
              ["Live BTCUSDT data", "Continuous Binance-fed terminal state with a single-symbol focus"],
              ["Forecasting engine", "Run Kronos-aligned signals instantly with model switching"],
              ["Multi-horizon view", "5-minute, 1-hour, and 1-day research modes"],
              ["Future-ready stack", "Prepared for news, sentiment, alerts, and model comparison"],
            ].map(([title, body]) => (
              <div key={title} className="rounded-3xl border border-[var(--panel-border)] bg-[var(--panel)]/86 p-5 backdrop-blur">
                <p className="mb-2 text-sm font-medium text-white">{title}</p>
                <p className="text-sm leading-6 text-[var(--muted)]">{body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="relative mx-auto flex w-full max-w-7xl flex-col gap-10 px-6 pb-16 lg:px-10">
        <FeatureGrid />
        <ModelFamiliesSection />
      </section>
    </main>
  );
}
