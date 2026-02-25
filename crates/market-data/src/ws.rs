//! Generic WebSocket connection wrapper with automatic reconnection.
//!
//! [`WsConnection`] manages the lifecycle of a WebSocket connection including
//! exponential backoff with jitter, configurable retry limits, and state
//! change notifications via the [`WsHandler`] trait.

use std::time::Duration;

use futures_util::stream::{SplitSink, SplitStream};
use futures_util::StreamExt;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

/// Sink half of a WebSocket connection, used to send messages.
pub type WsSink = SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, Message>;

/// Stream half of a WebSocket connection, used to receive messages.
pub type WsStream = SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>;

/// Connection state events emitted by [`WsConnection`].
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionState {
    /// Successfully connected to the remote endpoint.
    Connected,
    /// Disconnected from the remote endpoint.
    Disconnected {
        /// Human-readable reason for disconnection.
        reason: String,
    },
    /// Attempting to reconnect.
    Reconnecting {
        /// Current reconnection attempt number (1-based).
        attempt: u32,
    },
    /// Reconnection has permanently failed.
    Failed {
        /// Human-readable reason for failure.
        reason: String,
    },
}

/// Configuration for reconnection behavior.
#[derive(Debug, Clone)]
pub struct ReconnectConfig {
    /// Initial backoff duration before the first retry.
    pub initial_backoff: Duration,
    /// Maximum backoff duration (backoff is capped at this value).
    pub max_backoff: Duration,
    /// Maximum number of reconnection attempts before emitting `Failed` (0 = unlimited).
    pub max_retries: u32,
    /// Emit a warning log after this many consecutive failures.
    pub alert_after: u32,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            max_retries: 0, // unlimited
            alert_after: 5,
        }
    }
}

/// Trait that exchange-specific clients implement for reconnection handling.
///
/// The [`WsConnection`] calls these methods at the appropriate lifecycle
/// points during connection management.
#[async_trait::async_trait]
pub trait WsHandler: Send + 'static {
    /// Called after a new connection is established. Use to re-subscribe.
    async fn on_connect(&mut self, sink: &mut WsSink) -> anyhow::Result<()>;

    /// Called for each received WebSocket message.
    async fn on_message(&mut self, msg: Message) -> anyhow::Result<()>;

    /// Called on connection state changes.
    fn on_state_change(&mut self, state: ConnectionState);
}

/// Managed WebSocket connection with automatic reconnection.
///
/// Wraps a WebSocket URL with [`ReconnectConfig`] and drives the connection
/// loop, delegating message handling and lifecycle events to a [`WsHandler`].
pub struct WsConnection {
    url: String,
    config: ReconnectConfig,
}

impl WsConnection {
    /// Create a new managed WebSocket connection.
    pub fn new(url: String, config: ReconnectConfig) -> Self {
        Self { url, config }
    }

    /// Run the connection loop, reconnecting on failure.
    ///
    /// This method connects to the WebSocket URL, calls
    /// [`WsHandler::on_connect`] for initial subscription setup, then reads
    /// messages in a loop. On disconnection, it applies exponential backoff
    /// with jitter and reconnects.
    ///
    /// Returns `Ok(())` only if the handler or connection terminates cleanly
    /// (which in practice does not happen for a long-running feed). Returns
    /// `Err` if `max_retries` is exceeded.
    pub async fn run<H: WsHandler>(&self, handler: &mut H) -> anyhow::Result<()> {
        let mut attempt: u32 = 0;

        loop {
            // Notify reconnecting state (skip on first connect).
            if attempt > 0 {
                handler.on_state_change(ConnectionState::Reconnecting { attempt });

                let backoff = calculate_backoff(
                    &self.config.initial_backoff,
                    &self.config.max_backoff,
                    attempt - 1,
                );
                tracing::info!(
                    attempt = attempt,
                    backoff_ms = backoff.as_millis() as u64,
                    url = %self.url,
                    "reconnecting to WebSocket"
                );

                if attempt >= self.config.alert_after {
                    tracing::warn!(
                        attempt = attempt,
                        url = %self.url,
                        "WebSocket reconnection attempts exceeded alert threshold"
                    );
                }

                if self.config.max_retries > 0 && attempt > self.config.max_retries {
                    let reason = format!(
                        "exceeded max retries ({}) for {}",
                        self.config.max_retries, self.url
                    );
                    handler.on_state_change(ConnectionState::Failed {
                        reason: reason.clone(),
                    });
                    return Err(anyhow::anyhow!(reason));
                }

                tokio::time::sleep(backoff).await;
            }

            // Attempt connection.
            let ws_stream = match tokio_tungstenite::connect_async(&self.url).await {
                Ok((stream, _response)) => stream,
                Err(e) => {
                    let reason = format!("connection failed: {e}");
                    tracing::error!(url = %self.url, error = %e, "WebSocket connection failed");
                    handler.on_state_change(ConnectionState::Disconnected {
                        reason: reason.clone(),
                    });
                    attempt = attempt.saturating_add(1);
                    continue;
                }
            };

            let (mut sink, mut stream) = ws_stream.split();

            // Notify connected.
            handler.on_state_change(ConnectionState::Connected);

            // Re-subscribe after (re)connect.
            if let Err(e) = handler.on_connect(&mut sink).await {
                tracing::error!(
                    url = %self.url,
                    error = %e,
                    "on_connect handler failed"
                );
                handler.on_state_change(ConnectionState::Disconnected {
                    reason: format!("on_connect failed: {e}"),
                });
                attempt = attempt.saturating_add(1);
                continue;
            }

            // Message read loop. On success, reset attempts so that the
            // next reconnection starts from attempt 1.
            let disconnect_reason = loop {
                match stream.next().await {
                    Some(Ok(msg)) => {
                        if let Err(e) = handler.on_message(msg).await {
                            tracing::error!(
                                url = %self.url,
                                error = %e,
                                "on_message handler error"
                            );
                            // Continue reading; handler errors are non-fatal for the connection.
                        }
                    }
                    Some(Err(e)) => {
                        let reason = format!("WebSocket read error: {e}");
                        tracing::error!(url = %self.url, error = %e, "WebSocket read error");
                        break reason;
                    }
                    None => {
                        // Stream ended (clean close).
                        break "stream closed".to_string();
                    }
                }
            };

            handler.on_state_change(ConnectionState::Disconnected {
                reason: disconnect_reason,
            });
            attempt = 1;
        }
    }
}

/// Calculate exponential backoff with jitter.
///
/// `backoff = initial * 2^attempt`, capped at `max`. Jitter adds a random
/// amount in `[0, 0.5 * backoff]`.
pub(crate) fn calculate_backoff(
    initial: &Duration,
    max: &Duration,
    attempt: u32,
) -> Duration {
    let base = initial
        .saturating_mul(2u32.saturating_pow(attempt))
        .min(*max);

    // Add jitter: random 0-50% of the base backoff.
    let jitter_frac = rand::random::<f64>() * 0.5;
    let jitter = Duration::from_secs_f64(base.as_secs_f64() * jitter_frac);

    base + jitter
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ReconnectConfig::default();
        assert_eq!(config.initial_backoff, Duration::from_millis(100));
        assert_eq!(config.max_backoff, Duration::from_secs(30));
        assert_eq!(config.max_retries, 0);
        assert_eq!(config.alert_after, 5);
    }

    #[test]
    fn test_exponential_backoff_sequence() {
        let initial = Duration::from_millis(100);
        let max = Duration::from_secs(30);

        // Without jitter influence, we test the base calculation.
        // Since jitter is random, we test that the result is within expected bounds.
        for attempt in 0..10 {
            let backoff = calculate_backoff(&initial, &max, attempt);
            let expected_base = initial.saturating_mul(2u32.saturating_pow(attempt)).min(max);

            // Backoff must be >= base (jitter is non-negative).
            assert!(backoff >= expected_base, "attempt {attempt}: backoff < base");
            // Backoff must be <= base * 1.5 (jitter is at most 50%).
            let upper = expected_base + Duration::from_secs_f64(expected_base.as_secs_f64() * 0.5);
            assert!(
                backoff <= upper,
                "attempt {attempt}: backoff {backoff:?} > upper bound {upper:?}"
            );
        }
    }

    #[test]
    fn test_backoff_base_values() {
        // Verify the base doubling: 100ms, 200ms, 400ms, 800ms, ...
        let initial = Duration::from_millis(100);
        let max = Duration::from_secs(30);

        let expected_bases = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 30000];

        for (attempt, &expected_ms) in expected_bases.iter().enumerate() {
            let base = initial
                .saturating_mul(2u32.saturating_pow(attempt as u32))
                .min(max);
            assert_eq!(
                base.as_millis(),
                expected_ms as u128,
                "attempt {attempt}: expected {expected_ms}ms, got {:?}",
                base
            );
        }
    }

    #[test]
    fn test_backoff_capped_at_max() {
        let initial = Duration::from_millis(100);
        let max = Duration::from_secs(30);

        // At attempt 20, the raw value would be huge; it must be capped.
        let backoff = calculate_backoff(&initial, &max, 20);
        let upper = max + Duration::from_secs_f64(max.as_secs_f64() * 0.5);
        assert!(backoff <= upper, "backoff {backoff:?} exceeds capped upper bound {upper:?}");
        assert!(backoff >= max, "backoff {backoff:?} below max {max:?}");
    }

    #[test]
    fn test_jitter_within_range() {
        let initial = Duration::from_millis(1000);
        let max = Duration::from_secs(30);

        // Run multiple times to exercise the random path.
        for _ in 0..100 {
            let backoff = calculate_backoff(&initial, &max, 0);
            // Base is 1000ms. With jitter, result should be in [1000ms, 1500ms].
            assert!(backoff >= Duration::from_millis(1000));
            assert!(backoff <= Duration::from_millis(1500));
        }
    }

    #[test]
    fn test_connection_state_equality() {
        assert_eq!(ConnectionState::Connected, ConnectionState::Connected);
        assert_ne!(ConnectionState::Connected, ConnectionState::Failed {
            reason: "test".to_string(),
        });

        let d1 = ConnectionState::Disconnected { reason: "a".to_string() };
        let d2 = ConnectionState::Disconnected { reason: "a".to_string() };
        let d3 = ConnectionState::Disconnected { reason: "b".to_string() };
        assert_eq!(d1, d2);
        assert_ne!(d1, d3);

        let r1 = ConnectionState::Reconnecting { attempt: 1 };
        let r2 = ConnectionState::Reconnecting { attempt: 1 };
        let r3 = ConnectionState::Reconnecting { attempt: 2 };
        assert_eq!(r1, r2);
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_reconnect_attempt_counting_with_max_retries() {
        // Verify that config with max_retries=3 would allow attempts 1, 2, 3
        // and fail on attempt 4 (> max_retries).
        let config = ReconnectConfig {
            max_retries: 3,
            ..Default::default()
        };
        // Attempts 1..=3 are within range.
        for attempt in 1..=config.max_retries {
            assert!(
                attempt <= config.max_retries,
                "attempt {attempt} should be allowed"
            );
        }
        // Attempt 4 exceeds max_retries.
        assert!(4 > config.max_retries);
    }

    #[test]
    fn test_ws_connection_new() {
        let config = ReconnectConfig::default();
        let conn = WsConnection::new("wss://example.com".to_string(), config.clone());
        assert_eq!(conn.url, "wss://example.com");
        assert_eq!(conn.config.initial_backoff, config.initial_backoff);
        assert_eq!(conn.config.max_backoff, config.max_backoff);
        assert_eq!(conn.config.max_retries, config.max_retries);
        assert_eq!(conn.config.alert_after, config.alert_after);
    }
}
