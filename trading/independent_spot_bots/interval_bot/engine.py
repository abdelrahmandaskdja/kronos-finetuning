from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from decimal import Decimal

from .config import BotConfig
from .exchange import BinanceAPIError, BinanceSpotGateway
from .models import build_model_adapter
from .state import BotState, StateStore
from .types import BalanceSnapshot, ManagedOrder, OrderSide, PredictionStep, Regime, SignalDirection, SymbolFilters


@dataclass(slots=True)
class OrderPlan:
    side: OrderSide
    price: Decimal
    quantity: Decimal
    client_order_id: str


class SpotIntervalBot:
    def __init__(self, config: BotConfig, logger) -> None:
        self.config = config
        self.logger = logger
        self.gateway = BinanceSpotGateway(config, logger)
        self.state_store = StateStore(config.state_file_path)
        self.state: BotState = self.state_store.load()
        self.model = build_model_adapter(
            model_type=config.model_type,
            model_path=config.model_path,
            predlen=config.predlen,
            lookback=config.lookback,
            logger=logger.getChild("model"),
            python_callable=config.python_callable,
            tokenizer_path=config.tokenizer_path,
            device=config.model_device,
            clip=config.model_clip,
            direction_mode=config.model_direction_mode,
            stabilize_output=config.model_stabilize_output,
            stability_context_path=config.model_stability_context_path,
        )
        self.symbol_filters: SymbolFilters | None = None
        self.stop_event = asyncio.Event()
        self._client_order_counter = 0

    def request_stop(self) -> None:
        self.logger.info("Stop requested.")
        self.stop_event.set()

    async def run_forever(self, process_once: bool = False) -> None:
        try:
            self._initialize()
            await self._prepare_runtime()
            await self._catch_up_to_latest_closed_candle()

            if process_once:
                return

            while not self.stop_event.is_set():
                after_close_time = self._processing_cutoff()
                close_time = await self.gateway.wait_for_next_closed_candle(
                    symbol=self.config.symbol,
                    interval=self.config.timeframe,
                    after_close_time=after_close_time,
                    polling_fallback_enabled=self.config.polling_fallback_enabled,
                    poll_interval_seconds=self.config.polling_interval_seconds,
                    stop_event=self.stop_event,
                )
                if close_time is None:
                    continue
                await self._process_pending_candles_up_to(close_time)
        finally:
            await self.gateway.close()

    def _initialize(self) -> None:
        self.logger.info(
            "Starting bot=%s symbol=%s timeframe=%s predlen=%s dry_run=%s environment=%s",
            self.config.bot_name,
            self.config.symbol,
            self.config.timeframe,
            self.config.predlen,
            self.config.dry_run,
            self.config.environment.value,
        )
        self.gateway.sync_time_offset()
        self.symbol_filters = self.gateway.get_symbol_filters(self.config.symbol)

    async def _prepare_runtime(self) -> None:
        await self._refresh_balances_and_regime(reason="startup")
        await self._cancel_all_open_orders(reason="startup")

        if self.config.predlen == 16 and not self.state.has_active_forecast():
            latest_closed = self.gateway.get_latest_closed_candle(self.config.symbol, self.config.timeframe)
            prediction = self._run_model_for_window(end_close_time=latest_closed.close_time)
            self.state.replace_forecast(prediction, cycle_started_at=latest_closed.close_time)
            self.state_store.save(self.state)
            self.logger.info(
                "Initialized new 16-step forecast cycle at close_time=%s",
                latest_closed.close_time,
            )

    async def _catch_up_to_latest_closed_candle(self) -> None:
        latest_closed = self.gateway.get_latest_closed_candle(self.config.symbol, self.config.timeframe)
        await self._process_pending_candles_up_to(latest_closed.close_time)

    async def _process_pending_candles_up_to(self, target_close_time: int) -> None:
        candles = self.gateway.get_closed_candles(
            self.config.symbol,
            self.config.timeframe,
            limit=max(self.config.lookback + 64, 300),
            end_close_time=target_close_time,
        )
        cutoff = self._processing_cutoff()
        pending = [candle.close_time for candle in candles if cutoff < candle.close_time <= target_close_time]
        for close_time in pending:
            await self._process_single_candle(close_time)

    async def _process_single_candle(self, close_time: int) -> None:
        self.logger.info("Processing closed candle close_time=%s", close_time)
        if self.config.predlen == 1:
            prediction = self._run_model_for_window(end_close_time=close_time)
            step = prediction.steps[0]
            await self._apply_signal(step, close_time)
            self.state.last_processed_candle_close_time = close_time
            self.state_store.save(self.state)
            return

        if not self.state.has_active_forecast():
            prediction = self._run_model_for_window(
                end_close_time=self.state.last_processed_candle_close_time
                or self.state.forecast_cycle_started_at
                or close_time
            )
            self.state.replace_forecast(prediction, cycle_started_at=close_time)
            self.state_store.save(self.state)
            self.logger.warning(
                "Forecast buffer was empty at close_time=%s. A new cycle was created and trading resumes on the next candle.",
                close_time,
            )
            return

        step = self.state.consume_next_forecast_step()
        await self._apply_signal(step, close_time)
        self.state.last_processed_candle_close_time = close_time
        self.state_store.save(self.state)

        if not self.state.has_active_forecast():
            prediction = self._run_model_for_window(end_close_time=close_time)
            self.state.replace_forecast(prediction, cycle_started_at=close_time)
            self.state_store.save(self.state)
            self.logger.info("Refreshed 16-step forecast cycle at close_time=%s", close_time)

    def _run_model_for_window(self, end_close_time: int) -> "ModelPrediction":
        window = self.gateway.get_closed_window(
            self.config.symbol,
            self.config.timeframe,
            self.config.lookback,
            end_close_time=end_close_time,
        )
        prediction = self.model.predict(window)
        return prediction.validate(self.config.predlen)

    def _processing_cutoff(self) -> int:
        values = [self.state.last_processed_candle_close_time or 0]
        if self.config.predlen == 16:
            values.append(self.state.forecast_cycle_started_at or 0)
        return max(values)

    async def _apply_signal(self, step: PredictionStep, close_time: int) -> None:
        desired_regime = Regime.LONG if step.direction == SignalDirection.UP else Regime.CASH
        self.logger.info(
            "Signal direction=%s score=%.6f desired_regime=%s close_time=%s",
            step.direction.value,
            step.score,
            desired_regime.value,
            close_time,
        )

        await self._cancel_all_open_orders(reason=f"pre_signal_{close_time}")
        balances = await self._refresh_balances_and_regime(reason=f"pre_trade_{close_time}")
        current_regime = self._infer_regime(balances, close_time)
        self.state.regime = current_regime
        self.state_store.save(self.state)

        if current_regime == desired_regime:
            self.logger.info("Already in desired regime=%s. No action taken.", desired_regime.value)
            return

        plan = self._build_order_plan(desired_regime, close_time, balances)
        if plan is None:
            self.logger.warning("No valid order plan could be built for desired regime=%s", desired_regime.value)
            return

        await self._execute_order_plan(plan, desired_regime, close_time)

    async def _execute_order_plan(self, plan: OrderPlan, desired_regime: Regime, close_time: int) -> None:
        if self.config.dry_run:
            self._simulate_fill(plan, close_time)
            self.state.regime = desired_regime
            self.state_store.save(self.state)
            self.logger.info(
                "Dry-run simulated fill side=%s qty=%s price=%s regime=%s",
                plan.side.value,
                self.gateway.decimal_to_str(plan.quantity),
                self.gateway.decimal_to_str(plan.price),
                desired_regime.value,
            )
            return

        attempts = 1 + (
            self.config.max_cancel_replace_attempts if self.config.cancel_replace_enabled else 0
        )
        current_plan = plan
        overall_deadline = asyncio.get_running_loop().time() + self.config.order_timeout_seconds

        for attempt in range(1, attempts + 1):
            order = self.gateway.place_limit_order(
                symbol=self.config.symbol,
                side=current_plan.side,
                price=current_plan.price,
                quantity=current_plan.quantity,
                client_order_id=current_plan.client_order_id,
            )
            self.state.open_order_ids = [order.order_id]
            self.state_store.save(self.state)
            self.logger.info(
                "Placed limit order attempt=%s/%s order_id=%s side=%s qty=%s price=%s",
                attempt,
                attempts,
                order.order_id,
                order.side.value,
                self.gateway.decimal_to_str(order.orig_qty),
                self.gateway.decimal_to_str(order.price),
            )

            outcome = await self._wait_for_fill(
                order=order,
                desired_regime=desired_regime,
                close_time=close_time,
                overall_deadline=overall_deadline,
                replace_allowed=attempt < attempts,
            )
            if outcome == "filled":
                return
            if outcome != "replace":
                return

            if attempt >= attempts:
                break

            balances = await self._refresh_balances_and_regime(reason=f"replace_attempt_{attempt}_{close_time}")
            current_plan = self._build_order_plan(desired_regime, close_time, balances)
            if current_plan is None:
                self.logger.warning("Cancel/replace enabled but no valid replacement order could be built.")
                return

    async def _wait_for_fill(
        self,
        order: ManagedOrder,
        desired_regime: Regime,
        close_time: int,
        overall_deadline: float,
        replace_allowed: bool,
    ) -> str:
        loop = asyncio.get_running_loop()
        replace_deadline = min(
            overall_deadline,
            loop.time() + self.config.cancel_replace_interval_seconds,
        ) if replace_allowed else overall_deadline

        while loop.time() < overall_deadline and not self.stop_event.is_set():
            latest = self.gateway.get_order(self.config.symbol, order.order_id)
            if latest.status == "FILLED":
                self.state.clear_open_orders()
                balances = await self._refresh_balances_and_regime(reason=f"fill_{order.order_id}")
                self.state.regime = self._infer_regime(balances, close_time)
                self.state_store.save(self.state)
                self.logger.info("Order filled order_id=%s regime=%s", order.order_id, self.state.regime.value)
                return "filled"
            if latest.status in {"CANCELED", "EXPIRED", "REJECTED"}:
                self.state.clear_open_orders()
                balances = await self._refresh_balances_and_regime(reason=f"terminal_{latest.status.lower()}_{order.order_id}")
                self.state.regime = self._infer_regime(balances, close_time)
                self.state_store.save(self.state)
                self.logger.warning("Order ended without fill order_id=%s status=%s", order.order_id, latest.status)
                return "done"
            if replace_allowed and loop.time() >= replace_deadline:
                self.logger.info("Cancel/replace interval reached. Canceling order_id=%s", order.order_id)
                with contextlib.suppress(BinanceAPIError):
                    self.gateway.cancel_order(self.config.symbol, order.order_id)
                self.state.clear_open_orders()
                self.state_store.save(self.state)
                return "replace"
            await asyncio.sleep(self.config.order_status_poll_seconds)

        self.logger.warning("Order timeout reached. Canceling order_id=%s", order.order_id)
        with contextlib.suppress(BinanceAPIError):
            self.gateway.cancel_order(self.config.symbol, order.order_id)
        self.state.clear_open_orders()
        balances = await self._refresh_balances_and_regime(reason=f"timeout_cancel_{order.order_id}")
        self.state.regime = self._infer_regime(balances, close_time)
        self.state_store.save(self.state)
        if self.state.regime == desired_regime:
            self.logger.info("Timeout reached but balances already reflect desired regime=%s", desired_regime.value)
        return "done"

    def _build_order_plan(
        self,
        desired_regime: Regime,
        close_time: int,
        balances: BalanceSnapshot,
    ) -> OrderPlan | None:
        assert self.symbol_filters is not None

        reference_close = self.gateway.get_closed_window(
            self.config.symbol,
            self.config.timeframe,
            1,
            end_close_time=close_time,
        )[-1].close
        bid_price, ask_price = self.gateway.get_book_ticker(self.config.symbol)

        side = OrderSide.BUY if desired_regime == Regime.LONG else OrderSide.SELL
        if side == OrderSide.BUY:
            reference_price = bid_price if self.config.price_reference.value == "book" else reference_close
            raw_price = reference_price * (
                Decimal("1") - (self.config.limit_price_offset_bps / Decimal("10000"))
            )
            if self.config.prefer_maker_pricing and ask_price > 0 and raw_price >= ask_price:
                raw_price = max(bid_price, ask_price - self.symbol_filters.tick_size)
            price = self.gateway.quantize_price(raw_price, self.symbol_filters.tick_size, side)
            spendable_quote = balances.quote_free * self.config.quote_allocation_pct
            quantity = self.gateway.quantize_qty(
                spendable_quote / price if price > 0 else Decimal("0"),
                self.symbol_filters.step_size,
            )
        else:
            reference_price = ask_price if self.config.price_reference.value == "book" else reference_close
            raw_price = reference_price * (
                Decimal("1") + (self.config.limit_price_offset_bps / Decimal("10000"))
            )
            if self.config.prefer_maker_pricing and bid_price > 0 and raw_price <= bid_price:
                raw_price = max(raw_price, bid_price + self.symbol_filters.tick_size)
            price = self.gateway.quantize_price(raw_price, self.symbol_filters.tick_size, side)
            quantity = self.gateway.quantize_qty(
                balances.base_free * self.config.base_allocation_pct,
                self.symbol_filters.step_size,
            )

        if quantity < self.symbol_filters.min_qty:
            self.logger.warning("Quantity %s below minQty %s", quantity, self.symbol_filters.min_qty)
            return None
        if quantity > self.symbol_filters.max_qty:
            quantity = self.symbol_filters.max_qty

        notional = quantity * price
        if notional < self.symbol_filters.min_notional:
            self.logger.warning("Notional %s below minNotional %s", notional, self.symbol_filters.min_notional)
            return None
        if self.symbol_filters.max_notional and notional > self.symbol_filters.max_notional:
            self.logger.warning("Notional %s above maxNotional %s", notional, self.symbol_filters.max_notional)
            return None

        return OrderPlan(
            side=side,
            price=price,
            quantity=quantity,
            client_order_id=self._next_client_order_id(side),
        )

    def _next_client_order_id(self, side: OrderSide) -> str:
        self._client_order_counter += 1
        return f"{self.config.client_order_id_prefix}-{side.value.lower()}-{self._client_order_counter}"

    async def _refresh_balances_and_regime(self, reason: str) -> BalanceSnapshot:
        assert self.symbol_filters is not None

        if self.config.dry_run and not (self.gateway.api_key and self.gateway.api_secret):
            balances = self.state.last_balances
            if balances.captured_at_ms == 0:
                balances = BalanceSnapshot(
                    base_free=self.config.simulated_base_balance,
                    base_locked=Decimal("0"),
                    quote_free=self.config.simulated_quote_balance,
                    quote_locked=Decimal("0"),
                    captured_at_ms=self.gateway.server_time_ms(),
                )
        else:
            balances = self.gateway.get_account_balances(
                self.symbol_filters.base_asset,
                self.symbol_filters.quote_asset,
            )

        self.state.update_balances(balances)
        self.state.regime = self._infer_regime(balances, self._processing_cutoff() or self.gateway.server_time_ms())
        self.state_store.save(self.state)
        self.logger.info(
            "Refreshed balances reason=%s base_free=%s quote_free=%s regime=%s",
            reason,
            self.gateway.decimal_to_str(balances.base_free),
            self.gateway.decimal_to_str(balances.quote_free),
            self.state.regime.value,
        )
        return balances

    async def _cancel_all_open_orders(self, reason: str) -> None:
        if self.config.dry_run:
            self.state.clear_open_orders()
            self.state_store.save(self.state)
            return

        try:
            open_orders = self.gateway.list_open_orders(self.config.symbol)
        except BinanceAPIError as exc:
            self.logger.warning("Failed to list open orders during %s: %s", reason, exc)
            return

        if not open_orders:
            self.state.clear_open_orders()
            self.state_store.save(self.state)
            return

        self.logger.info("Canceling %s open orders reason=%s", len(open_orders), reason)
        for order in open_orders:
            with contextlib.suppress(BinanceAPIError):
                self.gateway.cancel_order(self.config.symbol, order.order_id)
        self.state.clear_open_orders()
        self.state_store.save(self.state)

    def _infer_regime(self, balances: BalanceSnapshot, close_time: int) -> Regime:
        assert self.symbol_filters is not None

        try:
            reference_close = self.gateway.get_closed_window(
                self.config.symbol,
                self.config.timeframe,
                1,
                end_close_time=close_time if close_time > 0 else None,
            )[-1].close
        except Exception:
            reference_close = Decimal("0")

        if balances.total_base < self.symbol_filters.min_qty:
            return Regime.CASH
        if reference_close <= 0:
            return Regime.LONG if balances.total_base > 0 else Regime.CASH
        notional = balances.total_base * reference_close
        return Regime.LONG if notional >= self.symbol_filters.min_notional else Regime.CASH

    def _simulate_fill(self, plan: OrderPlan, close_time: int) -> None:
        balances = self.state.last_balances
        if plan.side == OrderSide.BUY:
            quote_spent = plan.quantity * plan.price
            balances.quote_free = max(Decimal("0"), balances.quote_free - quote_spent)
            balances.base_free += plan.quantity
        else:
            base_sold = min(plan.quantity, balances.base_free)
            balances.base_free = max(Decimal("0"), balances.base_free - base_sold)
            balances.quote_free += base_sold * plan.price
        balances.captured_at_ms = self.gateway.server_time_ms()
        self.state.update_balances(balances)
        self.state.regime = self._infer_regime(balances, close_time)
