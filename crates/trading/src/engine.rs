//! Trading engine — wires all components and manages lifecycle.
//!
//! [`TradingEngine`] owns the shared state (circuit breaker, position tracker,
//! order manager, risk pipeline) and spawns the event loop, HTTP server, and
//! market data feed tasks.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Result};
use dashmap::DashMap;
use tokio::signal;
use tokio_util::sync::CancellationToken;

use cm_core::config::{AppConfig, TradingMode};
use cm_core::types::{Exchange, OrderId, Price, Symbol, Timestamp};
use cm_execution::gateway::ExchangeGateway;
use cm_market_data::ws::ReconnectConfig as WsReconnectConfig;
use cm_oms::{FillDeduplicator, OrderManager, PositionTracker};
use cm_risk::{
    CircuitBreaker, DrawdownCheck, FatFingerCheck, MaxOrderSizeCheck, MaxPositionCheck,
    OrderRateLimitCheck, RiskPipeline,
};
use cm_strategy::traits::Fill;
use cm_strategy::{default_registry, StrategyParams};

use crate::event_loop;
use crate::paper_executor::{PaperExecutor, RawFill};
use crate::server;
use crate::ws_feeds;

/// Shared state accessible by all engine components.
pub struct SharedState {
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub position_tracker: Arc<PositionTracker>,
    pub order_manager: Arc<OrderManager>,
    pub fill_dedup: Arc<FillDeduplicator>,
    pub risk_pipeline: Arc<RiskPipeline>,
    pub executor: Arc<dyn ExchangeGateway>,
    pub config: AppConfig,
    /// Maps `client_order_id` → internal `OrderId` for fill resolution.
    pub order_id_map: DashMap<String, OrderId>,
    /// Monotonic counter for generating internal `OrderId`s.
    pub next_order_id: AtomicU64,
    /// Latest mid price per (exchange, symbol), updated by WS feed tasks.
    pub mid_prices: DashMap<(Exchange, Symbol), Price>,
}

impl SharedState {
    /// Allocate the next internal OrderId.
    pub fn next_oid(&self) -> OrderId {
        OrderId(self.next_order_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Total daily PnL (realized + unrealized) across all positions.
    pub fn daily_pnl(&self) -> f64 {
        let positions = self.position_tracker.all_positions();
        let mut total = 0.0;
        for pos in &positions {
            total += pos.realized_pnl.to_f64();
            if let Some(mid_entry) = self.mid_prices.get(&(pos.exchange, pos.symbol.clone())) {
                let mid = *mid_entry.value();
                let unrealized =
                    self.position_tracker
                        .unrealized_pnl(&pos.exchange, &pos.symbol, mid);
                total += unrealized.to_f64();
            }
        }
        total
    }
}

/// The main trading engine.
pub struct TradingEngine {
    state: Arc<SharedState>,
    cancel: CancellationToken,
}

impl TradingEngine {
    /// Build a new engine from configuration.
    pub async fn new(config: AppConfig) -> Result<Self> {
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let position_tracker = Arc::new(PositionTracker::new());
        let order_manager = Arc::new(OrderManager::new());
        let fill_dedup = Arc::new(FillDeduplicator::new(format!("{}", std::process::id())));

        // Build risk pipeline from config
        let mut pipeline = RiskPipeline::new();
        pipeline.add_check(MaxPositionCheck {
            max_position_size: config.risk.max_position_size,
        });
        pipeline.add_check(MaxOrderSizeCheck {
            max_order_size: config.risk.max_order_size,
            max_market_order_size: config.risk.max_order_size * 0.5,
        });
        pipeline.add_check(OrderRateLimitCheck::new(
            config.risk.max_orders_per_second,
            config.risk.max_orders_per_second * 30,
        ));
        pipeline.add_check(DrawdownCheck {
            max_hourly_drawdown: config.risk.daily_loss_limit_usd * 0.5,
            max_daily_drawdown: config.risk.daily_loss_limit_usd,
        });
        pipeline.add_check(FatFingerCheck {
            max_deviation_bps: config.risk.fat_finger_bps as f64,
            max_post_only_deviation_bps: config.risk.fat_finger_bps as f64 * 2.0,
        });
        let risk_pipeline = Arc::new(pipeline);

        // Placeholder executor — real one is built in run() with the fill channel
        let executor: Arc<dyn ExchangeGateway> = match config.trading.mode {
            TradingMode::Paper => {
                let (fill_tx, _) = crossbeam::channel::unbounded::<RawFill>();
                Arc::new(PaperExecutor::new(config.paper.clone(), fill_tx))
            }
            TradingMode::Live => {
                bail!("Live trading mode not yet implemented — use paper mode");
            }
        };

        let state = Arc::new(SharedState {
            circuit_breaker,
            position_tracker,
            order_manager,
            fill_dedup,
            risk_pipeline,
            executor,
            config,
            order_id_map: DashMap::new(),
            next_order_id: AtomicU64::new(1),
            mid_prices: DashMap::new(),
        });

        Ok(Self {
            state,
            cancel: CancellationToken::new(),
        })
    }

    /// Run the trading engine.
    ///
    /// Spawns all tasks and blocks until SIGTERM / SIGINT / circuit breaker
    /// triggers shutdown.
    pub async fn run(self) -> Result<()> {
        let config = &self.state.config;

        // ── Channels ─────────────────────────────────────────────
        let (md_tx, md_rx) = crossbeam::channel::bounded::<event_loop::MarketDataEvent>(4096);
        let (action_tx, action_rx) =
            crossbeam::channel::bounded::<Vec<cm_strategy::OrderAction>>(1024);
        // Raw fills from executor → fill processor
        let (raw_fill_tx, raw_fill_rx) = crossbeam::channel::unbounded::<RawFill>();
        // Resolved fills from fill processor → strategy thread
        let (fill_tx, fill_rx) = crossbeam::channel::unbounded::<Fill>();

        // ── Build executor with the real fill channel ────────────
        let (executor, paper_executor): (Arc<dyn ExchangeGateway>, Option<Arc<PaperExecutor>>) =
            match config.trading.mode {
                TradingMode::Paper => {
                    let pe = Arc::new(PaperExecutor::new(
                        config.paper.clone(),
                        raw_fill_tx.clone(),
                    ));
                    (pe.clone() as Arc<dyn ExchangeGateway>, Some(pe))
                }
                TradingMode::Live => {
                    bail!("Live trading mode not yet implemented");
                }
            };

        // Rebuild shared state with real executor
        let state = Arc::new(SharedState {
            circuit_breaker: self.state.circuit_breaker.clone(),
            position_tracker: self.state.position_tracker.clone(),
            order_manager: self.state.order_manager.clone(),
            fill_dedup: self.state.fill_dedup.clone(),
            risk_pipeline: self.state.risk_pipeline.clone(),
            executor: executor.clone(),
            config: config.clone(),
            order_id_map: DashMap::new(),
            next_order_id: AtomicU64::new(1),
            mid_prices: DashMap::new(),
        });

        // ── Strategy instantiation ───────────────────────────────
        let registry = default_registry();
        let strategy_name = &config.trading.strategy;
        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        let strategy = registry.create(strategy_name, &params).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown strategy '{}'; available: {:?}",
                strategy_name,
                registry.available_strategies()
            )
        })?;

        tracing::info!(
            mode = ?config.trading.mode,
            strategy = %strategy_name,
            symbols = ?config.market_data.symbols,
            "engine starting"
        );

        let cancel = self.cancel.clone();

        // ── 1. HTTP server ───────────────────────────────────────
        let server_state = state.clone();
        let server_cancel = cancel.clone();
        tokio::spawn(async move {
            if let Err(e) = server::run_server(server_state, 8080, server_cancel).await {
                tracing::error!(error = %e, "HTTP server failed");
            }
        });

        // ── 2. Market data feeds ─────────────────────────────────
        let ws_reconnect = WsReconnectConfig {
            initial_backoff: Duration::from_millis(config.market_data.reconnect.initial_backoff_ms),
            max_backoff: Duration::from_millis(config.market_data.reconnect.max_backoff_ms),
            max_retries: config.market_data.reconnect.max_retries,
            alert_after: 5,
        };

        // 2a. Binance feed
        {
            let binance_md_tx = md_tx.clone();
            let binance_state = state.clone();
            let binance_cancel = cancel.clone();
            let binance_config = config.binance.clone();
            let binance_symbols = config.market_data.symbols.clone();
            let binance_pe = paper_executor.clone();
            tokio::spawn(async move {
                ws_feeds::run_binance_feed(
                    binance_config,
                    binance_symbols,
                    binance_md_tx,
                    binance_state,
                    binance_pe,
                    binance_cancel,
                )
                .await;
            });
        }

        // 2b. Bybit feed
        {
            let bybit_md_tx = md_tx.clone();
            let bybit_state = state.clone();
            let bybit_cancel = cancel.clone();
            let bybit_config = config.bybit.clone();
            let bybit_symbols = config.market_data.symbols.clone();
            let bybit_pe = paper_executor.clone();
            let bybit_reconnect = ws_reconnect;
            tokio::spawn(async move {
                ws_feeds::run_bybit_feed(
                    bybit_config,
                    bybit_reconnect,
                    bybit_symbols,
                    bybit_md_tx,
                    bybit_state,
                    bybit_pe,
                    bybit_cancel,
                )
                .await;
            });
        }

        // ── 3. Timer task (100ms periodic) ───────────────────────
        let timer_md_tx = md_tx.clone();
        let timer_cancel = cancel.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let _ = timer_md_tx.try_send(event_loop::MarketDataEvent::Timer(Timestamp::now()));
                    }
                    _ = timer_cancel.cancelled() => break,
                }
            }
        });

        // ── 4. Fill processor (tokio task) ───────────────────────
        // Reads RawFill from executor, resolves OrderId, updates OMS +
        // PositionTracker, forwards resolved Fill to strategy thread.
        let fill_state = state.clone();
        let fill_cancel = cancel.clone();
        tokio::spawn(async move {
            event_loop::fill_processor(fill_state, raw_fill_rx, fill_tx, fill_cancel).await;
        });

        // ── 5. Strategy thread (dedicated OS thread) ─────────────
        let strat_state = state.clone();
        let strat_cancel = cancel.clone();
        let strat_handle =
            std::thread::Builder::new()
                .name("strategy".into())
                .spawn(move || {
                    event_loop::strategy_loop(
                        strategy,
                        strat_state,
                        md_rx,
                        action_tx,
                        fill_rx,
                        strat_cancel,
                    );
                })?;

        // ── 6. Action processor (tokio task) ─────────────────────
        let proc_state = state.clone();
        let proc_executor = executor.clone();
        let proc_cancel = cancel.clone();
        tokio::spawn(async move {
            event_loop::action_processor(proc_state, proc_executor, action_rx, proc_cancel).await;
        });

        // ── Shutdown signal ──────────────────────────────────────
        tokio::select! {
            _ = signal::ctrl_c() => {
                tracing::info!("received SIGINT, shutting down");
            }
            _ = cancel.cancelled() => {
                tracing::info!("cancellation token triggered");
            }
        }

        cancel.cancel();

        let _ = strat_handle.join();

        tracing::info!("engine stopped");
        Ok(())
    }
}
