//! Binance REST API client for order management.
//!
//! Handles request signing, rate limiting, and JSON serialization for
//! the Binance spot trading API (`/api/v3/`).

use anyhow::{bail, Context, Result};
use reqwest::Client;
use tracing::debug;

use crate::rate_limiter::RateLimiter;
use crate::signing::sign_binance_request;

/// Binance REST API client.
///
/// Reuses a single `reqwest::Client` for connection pooling across requests.
/// All authenticated requests are signed with HMAC-SHA256 and pass through
/// the rate limiter before dispatch.
pub struct BinanceRestClient {
    base_url: String,
    api_key: String,
    api_secret: String,
    client: Client,
    rate_limiter: RateLimiter,
}

/// Binance order request payload.
#[derive(Debug, serde::Serialize)]
pub struct BinanceOrderRequest {
    /// Trading pair (e.g., "BTCUSDT").
    pub symbol: String,
    /// Order side: "BUY" or "SELL".
    pub side: String,
    /// Order type: "LIMIT", "MARKET", "LIMIT_MAKER".
    #[serde(rename = "type")]
    pub order_type: String,
    /// Order quantity as a decimal string.
    pub quantity: String,
    /// Limit price (omitted for market orders).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    /// Time-in-force: "GTC", "IOC", "FOK".
    #[serde(rename = "timeInForce", skip_serializing_if = "Option::is_none")]
    pub time_in_force: Option<String>,
    /// Client-assigned order ID for tracking.
    #[serde(rename = "newClientOrderId", skip_serializing_if = "Option::is_none")]
    pub client_order_id: Option<String>,
    /// Millisecond timestamp of the request.
    pub timestamp: u64,
    /// Receive window in milliseconds.
    #[serde(rename = "recvWindow")]
    pub recv_window: u64,
}

/// Binance order response from the API.
#[derive(Debug, serde::Deserialize)]
pub struct BinanceOrderResponse {
    /// Trading pair.
    pub symbol: String,
    /// Exchange-assigned order ID.
    #[serde(rename = "orderId")]
    pub order_id: u64,
    /// Client-assigned order ID.
    #[serde(rename = "clientOrderId")]
    pub client_order_id: String,
    /// Limit price.
    pub price: String,
    /// Original requested quantity.
    #[serde(rename = "origQty")]
    pub orig_qty: String,
    /// Quantity already executed.
    #[serde(rename = "executedQty")]
    pub executed_qty: String,
    /// Order status (NEW, FILLED, PARTIALLY_FILLED, CANCELED, etc.).
    pub status: String,
    /// Order type.
    #[serde(rename = "type")]
    pub order_type: String,
    /// Order side.
    pub side: String,
}

/// Binance account information response.
#[derive(Debug, serde::Deserialize)]
pub struct BinanceAccountInfo {
    /// Account balances.
    pub balances: Vec<BinanceBalance>,
    /// Whether the account can trade.
    #[serde(rename = "canTrade")]
    pub can_trade: bool,
}

/// A single asset balance entry.
#[derive(Debug, serde::Deserialize)]
pub struct BinanceBalance {
    /// Asset ticker (e.g., "BTC").
    pub asset: String,
    /// Free (available) balance.
    pub free: String,
    /// Locked balance (in open orders).
    pub locked: String,
}

/// Binance API error response.
#[derive(Debug, serde::Deserialize)]
pub struct BinanceApiError {
    /// Numeric error code.
    pub code: i32,
    /// Human-readable error message.
    pub msg: String,
}

impl BinanceRestClient {
    /// Create a new Binance REST client.
    ///
    /// Uses connection pooling via `reqwest::Client` for efficient HTTP reuse.
    pub fn new(base_url: impl Into<String>, api_key: impl Into<String>, api_secret: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            client: Client::new(),
            rate_limiter: RateLimiter::binance_default(),
        }
    }

    /// Place a new order.
    ///
    /// POST `/api/v3/order` with signed query string.
    pub async fn place_order(&self, req: &BinanceOrderRequest) -> Result<BinanceOrderResponse> {
        let query = serde_urlencoded::to_string(req)
            .context("failed to serialize order request")?;
        self.sign_and_send("POST", "/api/v3/order", &query, 1).await
    }

    /// Cancel an existing order.
    ///
    /// DELETE `/api/v3/order` with symbol and orderId.
    pub async fn cancel_order(&self, symbol: &str, order_id: u64) -> Result<BinanceOrderResponse> {
        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        let query = format!(
            "symbol={}&orderId={}&recvWindow=5000&timestamp={}",
            symbol, order_id, timestamp
        );
        self.sign_and_send("DELETE", "/api/v3/order", &query, 1).await
    }

    /// Get all open orders for a symbol.
    ///
    /// GET `/api/v3/openOrders`.
    pub async fn get_open_orders(&self, symbol: &str) -> Result<Vec<BinanceOrderResponse>> {
        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        let query = format!(
            "symbol={}&recvWindow=5000&timestamp={}",
            symbol, timestamp
        );
        self.sign_and_send_vec("GET", "/api/v3/openOrders", &query, 3).await
    }

    /// Get account information including balances.
    ///
    /// GET `/api/v3/account`.
    pub async fn get_account(&self) -> Result<BinanceAccountInfo> {
        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        let query = format!("recvWindow=5000&timestamp={}", timestamp);
        let signed_query = self.build_signed_query(&query);
        let url = format!("{}/api/v3/account?{}", self.base_url, signed_query);

        if !self.rate_limiter.try_acquire(10) {
            bail!("Binance rate limit exceeded for GET /api/v3/account");
        }

        debug!(endpoint = "/api/v3/account", "Binance GET request");

        let resp = self.client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .send()
            .await
            .context("Binance GET /api/v3/account request failed")?;

        // Sync rate limiter from response headers.
        if let Some(weight) = resp.headers().get("x-mbx-used-weight-1m") {
            if let Ok(w) = weight.to_str().unwrap_or("0").parse::<u32>() {
                self.rate_limiter.update_from_header(w);
            }
        }

        let status = resp.status();
        let body = resp.text().await.context("failed to read response body")?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<BinanceApiError>(&body) {
                bail!("Binance API error {}: {}", err.code, err.msg);
            }
            bail!("Binance HTTP {}: {}", status, body);
        }

        serde_json::from_str(&body).context("failed to deserialize account info")
    }

    /// Build a signed query string by appending `&signature=...`.
    fn build_signed_query(&self, query: &str) -> String {
        let signature = sign_binance_request(&self.api_secret, query);
        format!("{}&signature={}", query, signature)
    }

    /// Internal helper: sign query, check rate limit, send request, parse response.
    async fn sign_and_send<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        query: &str,
        weight: u32,
    ) -> Result<T> {
        if !self.rate_limiter.try_acquire(weight) {
            bail!("Binance rate limit exceeded for {} {}", method, path);
        }

        let signed_query = self.build_signed_query(query);
        let url = format!("{}{}?{}", self.base_url, path, signed_query);

        debug!(method, path, "Binance signed request");

        let resp = match method {
            "POST" => {
                self.client
                    .post(&url)
                    .header("X-MBX-APIKEY", &self.api_key)
                    .send()
                    .await
            }
            "DELETE" => {
                self.client
                    .delete(&url)
                    .header("X-MBX-APIKEY", &self.api_key)
                    .send()
                    .await
            }
            "GET" => {
                self.client
                    .get(&url)
                    .header("X-MBX-APIKEY", &self.api_key)
                    .send()
                    .await
            }
            _ => bail!("unsupported HTTP method: {}", method),
        }
        .with_context(|| format!("Binance {} {} request failed", method, path))?;

        // Sync rate limiter from response headers.
        if let Some(weight_hdr) = resp.headers().get("x-mbx-used-weight-1m") {
            if let Ok(w) = weight_hdr.to_str().unwrap_or("0").parse::<u32>() {
                self.rate_limiter.update_from_header(w);
            }
        }

        let status = resp.status();
        let body = resp.text().await.context("failed to read response body")?;

        if !status.is_success() {
            if let Ok(err) = serde_json::from_str::<BinanceApiError>(&body) {
                bail!("Binance API error {}: {}", err.code, err.msg);
            }
            bail!("Binance HTTP {}: {}", status, body);
        }

        serde_json::from_str(&body)
            .with_context(|| format!("failed to deserialize {} {} response", method, path))
    }

    /// Internal helper for endpoints that return a JSON array.
    async fn sign_and_send_vec<T: serde::de::DeserializeOwned>(
        &self,
        method: &str,
        path: &str,
        query: &str,
        weight: u32,
    ) -> Result<Vec<T>> {
        self.sign_and_send::<Vec<T>>(method, path, query, weight).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_request_serialization() {
        let req = BinanceOrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            quantity: "0.001".to_string(),
            price: Some("50000.00".to_string()),
            time_in_force: Some("GTC".to_string()),
            client_order_id: Some("test-123".to_string()),
            timestamp: 1706000000000,
            recv_window: 5000,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["symbol"], "BTCUSDT");
        assert_eq!(json["side"], "BUY");
        assert_eq!(json["type"], "LIMIT");
        assert_eq!(json["quantity"], "0.001");
        assert_eq!(json["price"], "50000.00");
        assert_eq!(json["timeInForce"], "GTC");
        assert_eq!(json["newClientOrderId"], "test-123");
        assert_eq!(json["timestamp"], 1706000000000u64);
        assert_eq!(json["recvWindow"], 5000);
    }

    #[test]
    fn test_order_request_skip_none_fields() {
        let req = BinanceOrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            order_type: "MARKET".to_string(),
            quantity: "0.001".to_string(),
            price: None,
            time_in_force: None,
            client_order_id: None,
            timestamp: 1706000000000,
            recv_window: 5000,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("price").is_none());
        assert!(json.get("timeInForce").is_none());
        assert!(json.get("newClientOrderId").is_none());
    }

    #[test]
    fn test_order_request_url_encoding() {
        let req = BinanceOrderRequest {
            symbol: "BTCUSDT".to_string(),
            side: "BUY".to_string(),
            order_type: "LIMIT".to_string(),
            quantity: "0.001".to_string(),
            price: Some("50000.00".to_string()),
            time_in_force: Some("GTC".to_string()),
            client_order_id: None,
            timestamp: 1706000000000,
            recv_window: 5000,
        };

        let encoded = serde_urlencoded::to_string(&req).unwrap();
        assert!(encoded.contains("symbol=BTCUSDT"));
        assert!(encoded.contains("side=BUY"));
        assert!(encoded.contains("type=LIMIT"));
        assert!(encoded.contains("timestamp=1706000000000"));
    }

    #[test]
    fn test_order_response_deserialization() {
        let json = r#"{
            "symbol": "BTCUSDT",
            "orderId": 28,
            "clientOrderId": "6gCrw2kRUAF9CvJDGP16IP",
            "price": "50000.00",
            "origQty": "0.001",
            "executedQty": "0.000",
            "status": "NEW",
            "type": "LIMIT",
            "side": "BUY"
        }"#;

        let resp: BinanceOrderResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.symbol, "BTCUSDT");
        assert_eq!(resp.order_id, 28);
        assert_eq!(resp.client_order_id, "6gCrw2kRUAF9CvJDGP16IP");
        assert_eq!(resp.price, "50000.00");
        assert_eq!(resp.orig_qty, "0.001");
        assert_eq!(resp.executed_qty, "0.000");
        assert_eq!(resp.status, "NEW");
        assert_eq!(resp.order_type, "LIMIT");
        assert_eq!(resp.side, "BUY");
    }

    #[test]
    fn test_order_response_filled() {
        let json = r#"{
            "symbol": "BTCUSDT",
            "orderId": 42,
            "clientOrderId": "test-fill",
            "price": "50000.00",
            "origQty": "0.001",
            "executedQty": "0.001",
            "status": "FILLED",
            "type": "LIMIT",
            "side": "SELL"
        }"#;

        let resp: BinanceOrderResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "FILLED");
        assert_eq!(resp.executed_qty, "0.001");
    }

    #[test]
    fn test_account_info_deserialization() {
        let json = r#"{
            "canTrade": true,
            "balances": [
                {"asset": "BTC", "free": "0.10000000", "locked": "0.01000000"},
                {"asset": "USDT", "free": "10000.00", "locked": "5000.00"}
            ]
        }"#;

        let info: BinanceAccountInfo = serde_json::from_str(json).unwrap();
        assert!(info.can_trade);
        assert_eq!(info.balances.len(), 2);
        assert_eq!(info.balances[0].asset, "BTC");
        assert_eq!(info.balances[0].free, "0.10000000");
        assert_eq!(info.balances[1].asset, "USDT");
        assert_eq!(info.balances[1].locked, "5000.00");
    }

    #[test]
    fn test_api_error_deserialization() {
        let json = r#"{"code": -1021, "msg": "Timestamp for this request was 1000ms ahead of the server's time."}"#;
        let err: BinanceApiError = serde_json::from_str(json).unwrap();
        assert_eq!(err.code, -1021);
        assert!(err.msg.contains("Timestamp"));
    }

    #[test]
    fn test_api_error_insufficient_balance() {
        let json = r#"{"code": -2010, "msg": "Account has insufficient balance for requested action."}"#;
        let err: BinanceApiError = serde_json::from_str(json).unwrap();
        assert_eq!(err.code, -2010);
        assert!(err.msg.contains("insufficient balance"));
    }

    #[test]
    fn test_client_construction() {
        let client = BinanceRestClient::new(
            "https://testnet.binance.vision",
            "test_key",
            "test_secret",
        );
        assert_eq!(client.base_url, "https://testnet.binance.vision");
        assert_eq!(client.api_key, "test_key");
    }

    #[test]
    fn test_build_signed_query() {
        let client = BinanceRestClient::new(
            "https://testnet.binance.vision",
            "key",
            "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j",
        );
        let query = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";
        let signed = client.build_signed_query(query);
        assert!(signed.starts_with(query));
        assert!(signed.contains("&signature="));
        assert!(signed.ends_with("c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"));
    }
}
