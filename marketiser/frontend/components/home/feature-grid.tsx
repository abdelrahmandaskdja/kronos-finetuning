import { Activity, BrainCircuit, CandlestickChart, RadioTower } from "lucide-react";

import { HOME_FEATURES } from "@/lib/constants";
import { Card } from "@/components/ui/card";

const icons = [RadioTower, BrainCircuit, CandlestickChart, Activity];

export function FeatureGrid() {
  return (
    <section className="grid gap-4 lg:grid-cols-4">
      {HOME_FEATURES.map((feature, index) => {
        const Icon = icons[index];
        return (
          <Card key={feature.title} className="p-6">
            <div className="mb-5 inline-flex rounded-2xl border border-white/8 bg-white/5 p-3 text-[var(--accent)]">
              <Icon className="size-5" />
            </div>
            <h2 className="mb-3 text-xl font-medium text-white">{feature.title}</h2>
            <p className="text-sm leading-7 text-[var(--muted)]">{feature.body}</p>
          </Card>
        );
      })}
    </section>
  );
}

