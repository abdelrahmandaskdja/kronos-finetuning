# Security policy

## Credentials

Never commit Binance API keys, API secrets, private keys, access tokens, or a
populated `.env` file. Use the checked-in `.env.example` files as templates and
keep real values only in local environment files.

If a credential is ever committed, remove it from history and rotate it at the
provider immediately. Deleting it only from the latest commit is not enough.

## Live trading

Live order placement is disabled by default. Keep `LIVE_TRADING=false` while
testing. Use least-privilege exchange keys, disable withdrawals, restrict keys
to known IPs where supported, and validate strategies in a sandbox or with
paper trading before enabling execution.

## Public artifacts

This public source tree excludes model weights, caches, runtime state, order and
signal logs, and real environment files. Large trained artifacts should be
published separately with an explicit model card and license review.
