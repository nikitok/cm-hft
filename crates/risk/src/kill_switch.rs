//! Kill switch HTTP server.
//!
//! Provides a minimal HTTP API (via axum) for emergency trading control.
//! The kill switch is designed to work even if the main trading engine
//! has crashed or is unresponsive.
//!
//! ## Endpoints
//!
//! - `POST /kill` — trigger the circuit breaker (no auth required; safety first)
//! - `GET /status` — return current trading status as JSON
//! - `POST /reset?token=<TOKEN>` — reset the circuit breaker (requires auth token)
//! - `GET /health` — simple health check

use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::circuit_breaker::CircuitBreaker;

/// Shared state for kill switch HTTP handlers.
pub struct KillSwitchState {
    /// The circuit breaker to control.
    pub circuit_breaker: Arc<CircuitBreaker>,
    /// Token required for the reset endpoint. If `None`, reset is disabled.
    pub reset_token: Option<String>,
}

/// JSON response for the `/status` endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResponse {
    /// Whether trading is currently enabled.
    pub trading_enabled: bool,
    /// Whether the circuit breaker has been triggered.
    pub circuit_breaker_triggered: bool,
    /// The reason for triggering, if any.
    pub trigger_reason: Option<String>,
}

/// JSON response for the `/kill` endpoint.
#[derive(Debug, Serialize)]
struct KillResponse {
    status: &'static str,
    message: &'static str,
}

/// JSON response for the `/reset` endpoint.
#[derive(Debug, Serialize)]
struct ResetResponse {
    status: &'static str,
    message: &'static str,
}

/// Query parameters for the `/reset` endpoint.
#[derive(Debug, Deserialize)]
struct ResetQuery {
    token: Option<String>,
}

/// JSON response for the `/health` endpoint.
#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

/// Build the kill switch axum router.
pub fn kill_switch_router(state: Arc<KillSwitchState>) -> Router {
    Router::new()
        .route("/kill", post(kill_handler))
        .route("/status", get(status_handler))
        .route("/reset", post(reset_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

/// `POST /kill` — trigger the circuit breaker immediately.
///
/// No authentication required. In an emergency, speed is more important
/// than access control.
async fn kill_handler(State(state): State<Arc<KillSwitchState>>) -> Json<KillResponse> {
    state
        .circuit_breaker
        .trigger("kill switch activated via HTTP".to_string());
    Json(KillResponse {
        status: "ok",
        message: "trading halted",
    })
}

/// `GET /status` — return current trading and circuit breaker status.
async fn status_handler(State(state): State<Arc<KillSwitchState>>) -> Json<StatusResponse> {
    let cb_status = state.circuit_breaker.status();
    Json(StatusResponse {
        trading_enabled: cb_status.trading_enabled,
        circuit_breaker_triggered: cb_status.triggered,
        trigger_reason: cb_status.trigger_reason,
    })
}

/// `POST /reset?token=<TOKEN>` — reset the circuit breaker.
///
/// Requires a valid token in the query string. If no token is configured
/// on the server, reset is disabled.
async fn reset_handler(
    State(state): State<Arc<KillSwitchState>>,
    Query(query): Query<ResetQuery>,
) -> Result<Json<ResetResponse>, StatusCode> {
    let expected = match &state.reset_token {
        Some(t) => t,
        None => return Err(StatusCode::FORBIDDEN),
    };

    match &query.token {
        Some(token) if token == expected => {
            state.circuit_breaker.reset();
            Ok(Json(ResetResponse {
                status: "ok",
                message: "circuit breaker reset",
            }))
        }
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

/// `GET /health` — simple liveness check.
async fn health_handler() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    fn make_state() -> Arc<KillSwitchState> {
        Arc::new(KillSwitchState {
            circuit_breaker: Arc::new(CircuitBreaker::new()),
            reset_token: Some("secret123".to_string()),
        })
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let state = make_state();
        let app = kill_switch_router(state);

        let req = Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_status_initially_enabled() {
        let state = make_state();
        let app = kill_switch_router(state);

        let req = Request::builder()
            .uri("/status")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: StatusResponse = serde_json::from_slice(&body).unwrap();
        assert!(json.trading_enabled);
        assert!(!json.circuit_breaker_triggered);
        assert!(json.trigger_reason.is_none());
    }

    #[tokio::test]
    async fn test_kill_triggers_circuit_breaker() {
        let state = make_state();
        let cb = state.circuit_breaker.clone();
        let app = kill_switch_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/kill")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Verify circuit breaker was triggered
        assert!(!cb.is_trading_enabled());
    }

    #[tokio::test]
    async fn test_reset_with_valid_token() {
        let state = make_state();
        let cb = state.circuit_breaker.clone();
        cb.trigger("test".to_string());

        let app = kill_switch_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/reset?token=secret123")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(cb.is_trading_enabled());
    }

    #[tokio::test]
    async fn test_reset_with_invalid_token() {
        let state = make_state();
        state.circuit_breaker.trigger("test".to_string());
        let app = kill_switch_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/reset?token=wrong")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_reset_without_token() {
        let state = make_state();
        state.circuit_breaker.trigger("test".to_string());
        let app = kill_switch_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/reset")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_reset_disabled_when_no_token_configured() {
        let state = Arc::new(KillSwitchState {
            circuit_breaker: Arc::new(CircuitBreaker::new()),
            reset_token: None,
        });
        state.circuit_breaker.trigger("test".to_string());
        let app = kill_switch_router(state);

        let req = Request::builder()
            .method("POST")
            .uri("/reset?token=anything")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn test_status_after_kill() {
        let state = make_state();
        state.circuit_breaker.trigger("manual kill".to_string());
        let app = kill_switch_router(state);

        let req = Request::builder()
            .uri("/status")
            .body(Body::empty())
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let body = resp.into_body().collect().await.unwrap().to_bytes();
        let json: StatusResponse = serde_json::from_slice(&body).unwrap();
        assert!(!json.trading_enabled);
        assert!(json.circuit_breaker_triggered);
        assert_eq!(json.trigger_reason, Some("manual kill".to_string()));
    }
}
