import { MODEL_FAMILY_COPY } from "@/lib/constants";
import { Card } from "@/components/ui/card";

export function ModelFamiliesSection() {
  return (
    <section className="space-y-5">
      <div className="space-y-3">
        <p className="font-mono text-xs uppercase tracking-[0.34em] text-[var(--muted)]">Model families</p>
        <h2 className="text-3xl font-semibold tracking-[-0.05em] text-white md:text-4xl">
          Kronos models are first-class citizens in the product.
        </h2>
        <p className="max-w-3xl text-base leading-7 text-[var(--muted)]">
          Marketiser is built around interval-aware model selection. Different branches dominate different horizons, and
          the interface is designed to surface that explicitly instead of hiding it behind a single generic forecast.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {MODEL_FAMILY_COPY.map((family) => (
          <Card key={family.key} className="p-6">
            <p className="mb-3 font-mono text-xs uppercase tracking-[0.28em] text-[var(--accent)]">{family.key}</p>
            <h3 className="mb-3 text-lg font-medium text-white">{family.title}</h3>
            <p className="text-sm leading-7 text-[var(--muted)]">{family.body}</p>
          </Card>
        ))}
      </div>
    </section>
  );
}

