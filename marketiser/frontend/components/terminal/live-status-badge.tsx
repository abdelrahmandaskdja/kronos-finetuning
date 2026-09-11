"use client";

import { Activity, PlugZap, Radio, WifiOff } from "lucide-react";

import type { LiveStatus } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

export function LiveStatusBadge({ status }: { status: LiveStatus }) {
  if (status.state === "live") {
    return (
      <Badge tone="live">
        <Radio className="size-3.5" />
        Live
      </Badge>
    );
  }
  if (status.state === "mock") {
    return (
      <Badge tone="warning">
        <Activity className="size-3.5" />
        Live Sim
      </Badge>
    );
  }
  if (status.state === "reconnecting") {
    return (
      <Badge tone="warning">
        <PlugZap className="size-3.5" />
        Reconnecting
      </Badge>
    );
  }
  return (
    <Badge tone="bearish">
      <WifiOff className="size-3.5" />
      Offline
    </Badge>
  );
}

