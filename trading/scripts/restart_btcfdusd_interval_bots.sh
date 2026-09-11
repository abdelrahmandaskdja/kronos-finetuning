#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  intervals=(15m 1m 1h 4h 1d)
else
  intervals=("$@")
fi

validate_interval() {
  case "$1" in
    1m|15m|1h|4h|1d) ;;
    *)
      echo "Unsupported interval: $1" >&2
      exit 1
      ;;
  esac
}

kill_interval_sessions() {
  local interval="$1"
  local prefix="kronos_${interval}_btcfdusd_live"
  while IFS= read -r session_name; do
    [[ "$session_name" == "$prefix"* ]] || continue
    tmux kill-session -t "$session_name"
  done < <(tmux list-sessions -F '#S' 2>/dev/null || true)
}

session_command() {
  local interval="$1"
  local env_file="$ROOT_DIR/trading/secrets/${interval}.env"
  if [[ ! -f "$env_file" ]]; then
    echo "Missing env file: $env_file" >&2
    exit 1
  fi
  case "$interval" in
    1m)
      printf "cd '%s' && exec python3 trading/binance_spot_multi_interval_bot.py --mode live --send-orders --loop --config trading/configs/strategy_1m_only_btcfdusd.json --env-file '%s'" "$ROOT_DIR" "$env_file"
      ;;
    15m)
      printf "cd '%s' && exec python3 trading/binance_spot_multi_interval_bot.py --mode live --send-orders --loop --poll-seconds 5 --config trading/configs/strategy_15m_only_btcfdusd.json --env-file '%s'" "$ROOT_DIR" "$env_file"
      ;;
    1h)
      printf "cd '%s' && exec python3 trading/binance_spot_multi_interval_bot.py --mode live --send-orders --loop --poll-seconds 5 --config trading/configs/strategy_1h_only_btcfdusd.json --env-file '%s'" "$ROOT_DIR" "$env_file"
      ;;
    4h)
      printf "cd '%s' && exec python3 trading/binance_spot_multi_interval_bot.py --mode live --send-orders --loop --poll-seconds 5 --config trading/configs/strategy_4h_only_btcfdusd.json --env-file '%s'" "$ROOT_DIR" "$env_file"
      ;;
    1d)
      printf "cd '%s' && exec python3 trading/independent_spot_bots/run_1d_bot.py --env-file '%s'" "$ROOT_DIR" "$env_file"
      ;;
  esac
}

for interval in "${intervals[@]}"; do
  validate_interval "$interval"
  kill_interval_sessions "$interval"
done

for interval in "${intervals[@]}"; do
  session_name="kronos_${interval}_btcfdusd_live_$(date -u +%Y%m%d_%H%M%S)"
  tmux new-session -d -s "$session_name" "$(session_command "$interval")"
  echo "started $session_name"
  sleep 1
done
