import clsx from "clsx";

type LogoSize = "hero" | "terminal";

export function MarketiserLogo({ size = "terminal" }: { size?: LogoSize }) {
  const hero = size === "hero";
  const bars = hero ? [72, 112, 64, 148, 94] : [20, 30, 16, 38, 24];

  return (
    <div className={clsx("inline-flex items-center gap-4", hero && "flex-col gap-8")}>
      <div className="relative">
        <div className={clsx("absolute inset-0 rounded-full blur-3xl", hero ? "marketiser-pulse bg-[rgba(30,216,193,0.16)]" : "bg-[rgba(30,216,193,0.12)]")} />
        <div className="relative flex items-end gap-2 rounded-[2rem] border border-[var(--panel-border)] bg-[var(--panel)] px-4 py-4 backdrop-blur">
          {bars.map((height, index) => (
            <span
              key={height}
              className="marketiser-bar rounded-full bg-[linear-gradient(180deg,rgba(30,216,193,1),rgba(245,158,11,0.72))]"
              style={{
                height,
                width: hero ? 14 : 7,
                animationDelay: `${index * 0.15}s`,
              }}
            />
          ))}
        </div>
      </div>
      <div className={clsx("space-y-2", hero && "space-y-3 text-center")}>
        <p
          className={clsx(
            "font-mono uppercase tracking-[0.34em] text-[var(--muted)]",
            hero ? "text-xs" : "text-[10px]",
          )}
        >
          Research terminal
        </p>
        <h1
          className={clsx(
            "font-semibold tracking-[-0.08em] text-white",
            hero ? "text-7xl md:text-[8rem]" : "text-3xl",
          )}
        >
          Marketiser
        </h1>
      </div>
    </div>
  );
}

