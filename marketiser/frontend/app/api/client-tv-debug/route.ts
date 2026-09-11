import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

type ClientTvDebugEvent = {
  at: string;
  detail?: string;
  href?: string | null;
  interval?: string;
  sessionId?: string;
  stage?: string;
  symbol?: string;
};

const globalState = globalThis as typeof globalThis & {
  __marketiserTvDebugEvents?: ClientTvDebugEvent[];
};

function getEventStore() {
  if (!globalState.__marketiserTvDebugEvents) {
    globalState.__marketiserTvDebugEvents = [];
  }
  return globalState.__marketiserTvDebugEvents;
}

export async function GET() {
  const items = getEventStore().slice(-100);
  return NextResponse.json({ items });
}

export async function POST(request: NextRequest) {
  const payload = (await request.json().catch(() => null)) as ClientTvDebugEvent | null;
  if (!payload) {
    return NextResponse.json({ ok: false, error: "invalid_payload" }, { status: 400 });
  }

  const event: ClientTvDebugEvent = {
    ...payload,
    at: payload.at ?? new Date().toISOString(),
  };
  const store = getEventStore();
  store.push(event);
  if (store.length > 200) {
    store.splice(0, store.length - 200);
  }

  console.log("[tv-debug]", JSON.stringify(event));
  return NextResponse.json({ ok: true });
}
