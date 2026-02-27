//! Bybit REST API v5 client for order management.
//!
//! Handles request signing, rate limiting, and JSON serialization for
//! the Bybit unified trading API (`/v5/`). All authenticated requests
//! include `X-BAPI-*` headers for HMAC-SHA256 authentication.

use anyhow::{bail, Context, Result};
use reqwest::Client;
use tracing::debug;

use crate::rate_limiter::RateLimiter;
use crate::signing::sign_bybit_request;

/// Bybit REST API v5 client.
///
/// Reuses a single `reqwest::Client` for connection pooling. All authenticated
/// requests are signed with HMAC-SHA256 via `X-BAPI-SIGN` headers.
pub struct BybitRestClient {
    base_url: String,
    api_key: String,
    api_secret: String,
    client: Client,
    rate_limiter: RateLimiter,
    recv_window: u64,
}

/// Bybit v5 order request payload.
#[derive(Debug, serde::Serialize)]
pub struct BybitOrderRequest {
    /// Product category: "linear", "inverse", "spot".
    pub category: String,
    /// Trading pair (e.g., "BTCUSDT").
    pub symbol: String,
    /// Order side: "Buy" or "Sell".
    pub side: String,
    /// Order type: "Limit" or "Market".
    #[serde(rename = "orderType")]
    pub order_type: String,
    /// Order quantity as a decimal string.
    pub qty: String,
    /// Limit price (omitted for market orders).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub price: Option<String>,
    /// Client-assigned order link ID for tracking.
    #[serde(rename = "orderLinkId", skip_serializing_if = "Option::is_none")]
    pub order_link_id: Option<String>,
    /// Time-in-force: "GTC", "IOC", "FOK", "PostOnly".
    #[serde(rename = "timeInForce", skip_serializing_if = "Option::is_none")]
    pub time_in_force: Option<String>,
}

/// Bybit v5 generic response wrapper.
#[derive(Debug, serde::Deserialize)]
pub struct BybitResponse<T> {
    /// Return code (0 = success).
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    /// Return message.
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    /// Response payload (present on success).
    pub result: Option<T>,
}

/// Bybit order placement/amendment/cancellation result.
#[derive(Debug, serde::Deserialize)]
pub struct BybitOrderResult {
    /// Exchange-assigned order ID.
    #[serde(rename = "orderId")]
    pub order_id: String,
    /// Client-assigned order link ID.
    #[serde(rename = "orderLinkId")]
    pub order_link_id: String,
}

/// Bybit position entry.
#[derive(Debug, serde::Deserialize)]
pub struct BybitPosition {
    /// Trading pair.
    pub symbol: String,
    /// Position side: "Buy", "Sell", "None".
    pub side: String,
    /// Position size.
    pub size: String,
    /// Average entry price.
    #[serde(rename = "avgPrice")]
    pub avg_price: String,
    /// Unrealized PnL.
    #[serde(rename = "unrealisedPnl")]
    pub unrealised_pnl: String,
}

/// Bybit position list wrapper.
#[derive(Debug, serde::Deserialize)]
pub struct BybitPositionList {
    /// List of positions.
    pub list: Vec<BybitPosition>,
}

/// Bybit wallet balance entry.
#[derive(Debug, serde::Deserialize)]
pub struct BybitWalletBalance {
    /// Account type.
    #[serde(rename = "accountType")]
    pub account_type: String,
    /// Coin balances.
    pub coin: Vec<BybitCoinBalance>,
}

/// Bybit single coin balance.
#[derive(Debug, serde::Deserialize)]
pub struct BybitCoinBalance {
    /// Coin name (e.g., "BTC").
    pub coin: String,
    /// Available to withdraw.
    #[serde(rename = "availableToWithdraw")]
    pub available_to_withdraw: String,
    /// Wallet balance.
    #[serde(rename = "walletBalance")]
    pub wallet_balance: String,
}

/// Bybit wallet balance list wrapper.
#[derive(Debug, serde::Deserialize)]
pub struct BybitWalletList {
    /// List of wallet balances.
    pub list: Vec<BybitWalletBalance>,
}

impl BybitRestClient {
    /// Create a new Bybit REST v5 client.
    ///
    /// Uses connection pooling via `reqwest::Client` for efficient HTTP reuse.
    pub fn new(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        api_secret: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            client: Client::new(),
            rate_limiter: RateLimiter::bybit_default(),
            recv_window: 5000,
        }
    }

    /// Place a new order.
    ///
    /// POST `/v5/order/create`.
    pub async fn place_order(&self, req: &BybitOrderRequest) -> Result<BybitOrderResult> {
        let body = serde_json::to_string(req).context("failed to serialize order request")?;
        self.sign_and_send_post("/v5/order/create", &body).await
    }

    /// Cancel an existing order.
    ///
    /// POST `/v5/order/cancel`.
    pub async fn cancel_order(
        &self,
        category: &str,
        symbol: &str,
        order_id: &str,
    ) -> Result<BybitOrderResult> {
        let body = serde_json::json!({
            "category": category,
            "symbol": symbol,
            "orderId": order_id,
        })
        .to_string();
        self.sign_and_send_post("/v5/order/cancel", &body).await
    }

    /// Amend an existing order's price and/or quantity.
    ///
    /// POST `/v5/order/amend`.
    pub async fn amend_order(
        &self,
        category: &str,
        symbol: &str,
        order_id: &str,
        new_price: Option<&str>,
        new_qty: Option<&str>,
    ) -> Result<BybitOrderResult> {
        let mut payload = serde_json::json!({
            "category": category,
            "symbol": symbol,
            "orderId": order_id,
        });

        if let Some(price) = new_price {
            payload["price"] = serde_json::Value::String(price.to_string());
        }
        if let Some(qty) = new_qty {
            payload["qty"] = serde_json::Value::String(qty.to_string());
        }

        self.sign_and_send_post("/v5/order/amend", &payload.to_string())
            .await
    }

    /// Get open positions.
    ///
    /// GET `/v5/position/list`.
    pub async fn get_positions(&self, category: &str, symbol: &str) -> Result<BybitPositionList> {
        let query = format!("category={}&symbol={}", category, symbol);
        self.sign_and_send_get("/v5/position/list", &query).await
    }

    /// Get wallet balance.
    ///
    /// GET `/v5/account/wallet-balance`.
    pub async fn get_wallet_balance(&self, account_type: &str) -> Result<BybitWalletList> {
        let query = format!("accountType={}", account_type);
        self.sign_and_send_get("/v5/account/wallet-balance", &query)
            .await
    }

    /// Internal helper: sign and send a POST request with JSON body.
    async fn sign_and_send_post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        body: &str,
    ) -> Result<T> {
        if !self.rate_limiter.try_acquire(1) {
            bail!("Bybit rate limit exceeded for POST {}", path);
        }

        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        let signature = sign_bybit_request(
            &self.api_secret,
            timestamp,
            &self.api_key,
            self.recv_window,
            body,
        );

        let url = format!("{}{}", self.base_url, path);
        debug!(path, "Bybit POST request");

        let resp = self
            .client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", &signature)
            .header("X-BAPI-RECV-WINDOW", self.recv_window.to_string())
            .header("Content-Type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .with_context(|| format!("Bybit POST {} request failed", path))?;

        let status = resp.status();
        let resp_body = resp.text().await.context("failed to read response body")?;

        if !status.is_success() {
            bail!("Bybit HTTP {}: {}", status, resp_body);
        }

        let wrapper: BybitResponse<T> = serde_json::from_str(&resp_body)
            .with_context(|| format!("failed to deserialize Bybit {} response", path))?;

        if wrapper.ret_code != 0 {
            bail!("Bybit API error {}: {}", wrapper.ret_code, wrapper.ret_msg);
        }

        wrapper
            .result
            .with_context(|| format!("Bybit {} returned null result", path))
    }

    /// Internal helper: sign and send a GET request with query string.
    async fn sign_and_send_get<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        query: &str,
    ) -> Result<T> {
        if !self.rate_limiter.try_acquire(1) {
            bail!("Bybit rate limit exceeded for GET {}", path);
        }

        let timestamp = chrono::Utc::now().timestamp_millis() as u64;
        // For GET requests, Bybit signs: timestamp + api_key + recv_window + query_string
        let signature = sign_bybit_request(
            &self.api_secret,
            timestamp,
            &self.api_key,
            self.recv_window,
            query,
        );

        let url = format!("{}{}?{}", self.base_url, path, query);
        debug!(path, "Bybit GET request");

        let resp = self
            .client
            .get(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", &signature)
            .header("X-BAPI-RECV-WINDOW", self.recv_window.to_string())
            .send()
            .await
            .with_context(|| format!("Bybit GET {} request failed", path))?;

        let status = resp.status();
        let resp_body = resp.text().await.context("failed to read response body")?;

        if !status.is_success() {
            bail!("Bybit HTTP {}: {}", status, resp_body);
        }

        let wrapper: BybitResponse<T> = serde_json::from_str(&resp_body)
            .with_context(|| format!("failed to deserialize Bybit {} response", path))?;

        if wrapper.ret_code != 0 {
            bail!("Bybit API error {}: {}", wrapper.ret_code, wrapper.ret_msg);
        }

        wrapper
            .result
            .with_context(|| format!("Bybit {} returned null result", path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_request_serialization() {
        let req = BybitOrderRequest {
            category: "linear".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "Buy".to_string(),
            order_type: "Limit".to_string(),
            qty: "0.001".to_string(),
            price: Some("50000".to_string()),
            order_link_id: Some("test-link-001".to_string()),
            time_in_force: Some("GTC".to_string()),
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["category"], "linear");
        assert_eq!(json["symbol"], "BTCUSDT");
        assert_eq!(json["side"], "Buy");
        assert_eq!(json["orderType"], "Limit");
        assert_eq!(json["qty"], "0.001");
        assert_eq!(json["price"], "50000");
        assert_eq!(json["orderLinkId"], "test-link-001");
        assert_eq!(json["timeInForce"], "GTC");
    }

    #[test]
    fn test_order_request_skip_none_fields() {
        let req = BybitOrderRequest {
            category: "linear".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: "Buy".to_string(),
            order_type: "Market".to_string(),
            qty: "0.001".to_string(),
            price: None,
            order_link_id: None,
            time_in_force: None,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("price").is_none());
        assert!(json.get("orderLinkId").is_none());
        assert!(json.get("timeInForce").is_none());
    }

    #[test]
    fn test_order_result_deserialization() {
        let json = r#"{
            "orderId": "1234567890",
            "orderLinkId": "test-link-001"
        }"#;

        let result: BybitOrderResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.order_id, "1234567890");
        assert_eq!(result.order_link_id, "test-link-001");
    }

    #[test]
    fn test_response_wrapper_success() {
        let json = r#"{
            "retCode": 0,
            "retMsg": "OK",
            "result": {
                "orderId": "1234567890",
                "orderLinkId": "test-link-001"
            }
        }"#;

        let resp: BybitResponse<BybitOrderResult> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.ret_code, 0);
        assert_eq!(resp.ret_msg, "OK");
        let result = resp.result.unwrap();
        assert_eq!(result.order_id, "1234567890");
    }

    #[test]
    fn test_response_wrapper_error() {
        let json = r#"{
            "retCode": 10001,
            "retMsg": "parameter error",
            "result": null
        }"#;

        let resp: BybitResponse<BybitOrderResult> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.ret_code, 10001);
        assert_eq!(resp.ret_msg, "parameter error");
        assert!(resp.result.is_none());
    }

    #[test]
    fn test_position_deserialization() {
        let json = r#"{
            "list": [
                {
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.001",
                    "avgPrice": "50000.00",
                    "unrealisedPnl": "5.50"
                }
            ]
        }"#;

        let positions: BybitPositionList = serde_json::from_str(json).unwrap();
        assert_eq!(positions.list.len(), 1);
        assert_eq!(positions.list[0].symbol, "BTCUSDT");
        assert_eq!(positions.list[0].side, "Buy");
        assert_eq!(positions.list[0].size, "0.001");
        assert_eq!(positions.list[0].avg_price, "50000.00");
    }

    #[test]
    fn test_wallet_balance_deserialization() {
        let json = r#"{
            "list": [
                {
                    "accountType": "UNIFIED",
                    "coin": [
                        {
                            "coin": "BTC",
                            "availableToWithdraw": "0.5",
                            "walletBalance": "1.0"
                        },
                        {
                            "coin": "USDT",
                            "availableToWithdraw": "10000",
                            "walletBalance": "15000"
                        }
                    ]
                }
            ]
        }"#;

        let wallet: BybitWalletList = serde_json::from_str(json).unwrap();
        assert_eq!(wallet.list.len(), 1);
        assert_eq!(wallet.list[0].account_type, "UNIFIED");
        assert_eq!(wallet.list[0].coin.len(), 2);
        assert_eq!(wallet.list[0].coin[0].coin, "BTC");
        assert_eq!(wallet.list[0].coin[0].wallet_balance, "1.0");
        assert_eq!(wallet.list[0].coin[1].available_to_withdraw, "10000");
    }

    #[test]
    fn test_client_construction() {
        let client =
            BybitRestClient::new("https://api-testnet.bybit.com", "test_key", "test_secret");
        assert_eq!(client.base_url, "https://api-testnet.bybit.com");
        assert_eq!(client.api_key, "test_key");
        assert_eq!(client.recv_window, 5000);
    }

    #[test]
    fn test_response_wrapper_with_no_result_field() {
        // Some endpoints may omit result entirely
        let json = r#"{
            "retCode": 0,
            "retMsg": "OK"
        }"#;

        let resp: BybitResponse<BybitOrderResult> = serde_json::from_str(json).unwrap();
        assert_eq!(resp.ret_code, 0);
        assert!(resp.result.is_none());
    }
}
