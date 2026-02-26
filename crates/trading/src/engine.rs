//! Trading engine — wires all components and manages lifecycle.
//!
//! [`TradingEngine`] owns the shared state (circuit breaker, position tracker,
//! order manager, risk pipeline) and spawns the event loop, HTTP server, and
//! market data feed tasks.

use std::sync::Arc;

use anyhow::{bail, Result};
use tokio::signal;
use tokio_util::sync::CancellationToken;

use cm_core::config::{AppConfig, TradingMode};
use cm_core::types::{Symbol, Timestamp};
use cm_execution::gateway::ExchangeGateway;
use cm_oms::{FillDeduplicator, OrderManager, PositionTracker};
use cm_risk::{
    CircuitBreaker, DrawdownCheck, FatFingerCheck, MaxOrderSizeCheck,
    MaxPositionCheck, OrderRateLimitCheck, RiskPipeline,
};
use cm_strategy::traits::Fill;
use cm_strategy::{default_registry, StrategyParams};

use crate::event_loop;
use crate::paper_executor::PaperExecutor;
use crate::server;

/// Shared state accessible by all engine components.
pub struct SharedState {
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub position_tracker: Arc<PositionTracker>,
    pub order_manager: Arc<OrderManager>,
    pub fill_dedup: Arc<FillDeduplicator>,
    pub risk_pipeline: Arc<RiskPipeline>,
    pub executor: Arc<dyn ExchangeGateway>,
    pub config: AppConfig,
}

/// The main trading engine.
pub struct TradingEngine {
    state: Arc<SharedState>,
    cancel: CancellationToken,
}

impl TradingEngine {
    /// Build a new engine from configuration.
    ///
    /// Resolves the executor (paper vs live), builds the risk pipeline, and
    /// instantiates all shared state.
    pub async fn new(config: AppConfig) -> Result<Self> {
        let circuit_breaker = Arc::new(CircuitBreaker::new());
        let position_tracker = Arc::new(PositionTracker::new());
        let order_manager = Arc::new(OrderManager::new());
        let fill_dedup = Arc::new(FillDeduplicator::new(
            format!("{}", std::process::id()),
        ));

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

        // Build executor based on trading mode
        // For now we create a dummy fill channel; the real one is set up in run()
        let executor: Arc<dyn ExchangeGateway> = match config.trading.mode {
            TradingMode::Paper => {
                // We'll create the real PaperExecutor in run() so it has the
                // correct fill channel. For now, store a placeholder config.
                // Actually, we need the fill_tx here. Let's create it now.
                let (fill_tx, _fill_rx) = crossbeam::channel::unbounded::<Fill>();
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
        let (action_tx, action_rx) = crossbeam::channel::bounded::<Vec<cm_strategy::OrderAction>>(1024);
        let (fill_tx, fill_rx) = crossbeam::channel::unbounded::<Fill>();

        // ── Rebuild executor with the real fill channel ──────────
        let executor: Arc<dyn ExchangeGateway> = match config.trading.mode {
            TradingMode::Paper => {
                Arc::new(PaperExecutor::new(config.paper.clone(), fill_tx.clone()))
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
        });

        // ── Strategy instantiation ───────────────────────────────
        let registry = default_registry();
        let strategy_name = &config.trading.strategy;
        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        let strategy = registry
            .create(strategy_name, &params)
            .ok_or_else(|| {
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
        // For each symbol, spawn WS feed tasks that send to md_tx.
        // In paper mode, also call PaperExecutor::update_market_data().
        for symbol in &config.market_data.symbols {
            let md_tx = md_tx.clone();
            let symbol = Symbol::new(symbol.clone());
            let cancel = cancel.clone();

            tracing::info!(%symbol, "subscribing to market data");

            // Spawn a simulated feed that sends timer events
            // (Real WS feeds would be spawned here; for now we just log)
            let sym = symbol.clone();
            tokio::spawn(async move {
                tracing::info!(%sym, "market data feed task started (connect WS here)");
                cancel.cancelled().await;
                tracing::info!(%sym, "market data feed task stopped");
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

        // ── 4. Strategy thread (dedicated OS thread) ─────────────
        let strat_state = state.clone();
        let strat_cancel = cancel.clone();
        let strat_handle = std::thread::Builder::new()
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

        // ── 5. Action processor (tokio task) ─────────────────────
        let proc_state = state.clone();
        let proc_executor = executor.clone();
        let proc_fill_tx = fill_tx;
        let proc_cancel = cancel.clone();
        tokio::spawn(async move {
            event_loop::action_processor(
                proc_state,
                proc_executor,
                action_rx,
                proc_fill_tx,
                proc_cancel,
            )
            .await;
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

        // Wait for strategy thread to finish
        let _ = strat_handle.join();

        tracing::info!("engine stopped");
        Ok(())
    }
}
