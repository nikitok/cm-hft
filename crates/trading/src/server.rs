//! HTTP server — merges kill switch routes with trading endpoints.
//!
//! Exposes `/positions`, `/orders`, and `/metrics` alongside the existing
//! kill switch routes (`/kill`, `/status`, `/reset`, `/health`).

use std::sync::Arc;

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

use cm_risk::{kill_switch_router, KillSwitchState};

use crate::engine::SharedState;

/// JSON-serializable position for the REST API.
#[derive(Debug, Serialize)]
struct PositionResponse {
    exchange: String,
    symbol: String,
    net_quantity: f64,
    avg_entry_price: f64,
    realized_pnl: f64,
    fill_count: u64,
}

/// JSON-serializable order for the REST API.
#[derive(Debug, Serialize)]
struct OrderResponse {
    id: u64,
    client_order_id: String,
    exchange_order_id: Option<String>,
    exchange: String,
    symbol: String,
    side: String,
    order_type: String,
    price: f64,
    quantity: f64,
    filled_quantity: f64,
    status: String,
}

/// `GET /positions` — return all current positions as JSON.
async fn positions_handler(State(state): State<Arc<SharedState>>) -> Json<Vec<PositionResponse>> {
    let positions = state.position_tracker.all_positions();
    let resp: Vec<PositionResponse> = positions
        .into_iter()
        .map(|p| PositionResponse {
            exchange: format!("{}", p.exchange),
            symbol: format!("{}", p.symbol),
            net_quantity: p.net_quantity.to_f64(),
            avg_entry_price: p.avg_entry_price.to_f64(),
            realized_pnl: p.realized_pnl.to_f64(),
            fill_count: p.fill_count,
        })
        .collect();
    Json(resp)
}

/// `GET /orders` — return all open (non-terminal) orders as JSON.
async fn orders_handler(State(state): State<Arc<SharedState>>) -> Json<Vec<OrderResponse>> {
    let orders = state.order_manager.get_open_orders();
    let resp: Vec<OrderResponse> = orders
        .into_iter()
        .map(|o| OrderResponse {
            id: o.id.0,
            client_order_id: o.client_order_id,
            exchange_order_id: o.exchange_order_id.map(|e| e.0),
            exchange: format!("{}", o.exchange),
            symbol: format!("{}", o.symbol),
            side: format!("{}", o.side),
            order_type: format!("{}", o.order_type),
            price: o.price.to_f64(),
            quantity: o.quantity.to_f64(),
            filled_quantity: o.filled_quantity.to_f64(),
            status: format!("{:?}", o.status),
        })
        .collect();
    Json(resp)
}

/// `GET /metrics` — simple Prometheus-style text metrics.
async fn metrics_handler(State(state): State<Arc<SharedState>>) -> String {
    let positions = state.position_tracker.all_positions();
    let open_orders = state.order_manager.get_open_orders();
    let total_orders = state.order_manager.order_count();
    let trading_enabled = state.circuit_breaker.is_trading_enabled();

    let mut out = String::new();
    out.push_str(&format!(
        "# HELP cm_trading_enabled Whether trading is enabled\n\
         # TYPE cm_trading_enabled gauge\n\
         cm_trading_enabled {}\n",
        if trading_enabled { 1 } else { 0 }
    ));
    out.push_str(&format!(
        "# HELP cm_open_orders Number of open orders\n\
         # TYPE cm_open_orders gauge\n\
         cm_open_orders {}\n",
        open_orders.len()
    ));
    out.push_str(&format!(
        "# HELP cm_total_orders Total orders created\n\
         # TYPE cm_total_orders counter\n\
         cm_total_orders {}\n",
        total_orders
    ));

    for pos in &positions {
        let labels = format!(
            "exchange=\"{}\",symbol=\"{}\"",
            pos.exchange, pos.symbol
        );
        out.push_str(&format!(
            "cm_position_net_qty{{{}}} {}\n",
            labels,
            pos.net_quantity.to_f64()
        ));
        out.push_str(&format!(
            "cm_position_realized_pnl{{{}}} {}\n",
            labels,
            pos.realized_pnl.to_f64()
        ));
    }

    out
}

/// Build and run the combined HTTP server.
pub async fn run_server(
    state: Arc<SharedState>,
    port: u16,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    // Kill switch routes need their own state type
    let ks_state = Arc::new(KillSwitchState {
        circuit_breaker: state.circuit_breaker.clone(),
        reset_token: std::env::var("CM_RESET_TOKEN").ok(),
    });

    let ks_router = kill_switch_router(ks_state);

    // Trading-specific routes
    let trading_router = Router::new()
        .route("/positions", get(positions_handler))
        .route("/orders", get(orders_handler))
        .route("/metrics", get(metrics_handler))
        .with_state(state);

    // Merge both routers
    let app = trading_router.merge(ks_router);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!(%addr, "HTTP server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            cancel.cancelled().await;
        })
        .await?;

    Ok(())
}
