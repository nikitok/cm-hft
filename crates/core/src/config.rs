//! Layered configuration for the CM.HFT trading platform.
//!
//! Configuration is loaded in layers with increasing priority:
//! 1. Compiled-in defaults (testnet URLs, conservative risk parameters)
//! 2. TOML configuration file (if provided)
//! 3. Environment variable overrides (prefix `CM_HFT_`, nested with `__`)
//! 4. Specific env vars for API secrets (`BINANCE_API_KEY`, etc.)
//!
//! API keys and secrets **must** come from environment variables, never from
//! configuration files, to prevent accidental check-in of credentials.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use config::{Config, Environment, File};
use serde::Deserialize;

// ── Default value functions ────────────────────────────────────────────

/// Default request timeout: 5 000 ms.
fn default_timeout_ms() -> u64 {
    5_000
}

/// Default order book depth: 20 levels.
fn default_depth_levels() -> usize {
    20
}

/// Default initial reconnect backoff: 1 000 ms.
fn default_initial_backoff_ms() -> u64 {
    1_000
}

/// Default maximum reconnect backoff: 30 000 ms.
fn default_max_backoff_ms() -> u64 {
    30_000
}

/// Default maximum reconnect retries: 10.
fn default_max_retries() -> u32 {
    10
}

/// Default maximum orders per second: 5.
fn default_max_orders_per_second() -> u32 {
    5
}

/// Default fat-finger threshold: 50 basis points.
fn default_fat_finger_bps() -> u32 {
    50
}

/// Default recorder batch size: 1 000 rows.
fn default_batch_size() -> usize {
    1_000
}

/// Default recorder flush interval: 1 000 ms.
fn default_flush_interval_ms() -> u64 {
    1_000
}

/// Default paper-trading simulated latency: 1 ms.
fn default_paper_latency_ms() -> u64 {
    1
}

/// Default paper-trading maker fee: -0.01 % (rebate).
fn default_paper_maker_fee() -> f64 {
    -0.0001
}

/// Default paper-trading taker fee: 0.04 %.
fn default_paper_taker_fee() -> f64 {
    0.0004
}

/// Default paper-trading max fill fraction: 100 %.
fn default_paper_max_fill_fraction() -> f64 {
    1.0
}

// ── Configuration structs ──────────────────────────────────────────────

/// Top-level application configuration.
///
/// Aggregates exchange connections, market data subscriptions, risk limits,
/// recording settings, and trading mode into a single loadable unit.
#[derive(Debug, Clone, Deserialize)]
pub struct AppConfig {
    /// Binance exchange connection settings.
    pub binance: ExchangeConfig,
    /// Bybit exchange connection settings.
    pub bybit: ExchangeConfig,
    /// Market data subscription settings.
    pub market_data: MarketDataConfig,
    /// Risk management limits.
    pub risk: RiskConfig,
    /// QuestDB recorder settings.
    pub recorder: RecorderConfig,
    /// Trading mode and strategy selection.
    pub trading: TradingConfig,
    /// Paper-trading simulation parameters.
    #[serde(default)]
    pub paper: PaperConfig,
}

/// Exchange connection configuration.
///
/// API key and secret **must** come from environment variables, never config
/// files. The `#[serde(default)]` annotation ensures deserialization does not
/// require them in the TOML source.
#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeConfig {
    /// API key — loaded from env var (e.g., `BINANCE_API_KEY`).
    #[serde(default)]
    pub api_key: String,
    /// API secret — loaded from env var (e.g., `BINANCE_API_SECRET`).
    #[serde(default)]
    pub api_secret: String,
    /// Use testnet endpoints.
    #[serde(default)]
    pub testnet: bool,
    /// WebSocket base URL.
    pub ws_url: String,
    /// REST API base URL.
    pub rest_url: String,
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

/// Market data subscription configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MarketDataConfig {
    /// Symbols to subscribe to (e.g., `["BTCUSDT", "ETHUSDT"]`).
    pub symbols: Vec<String>,
    /// Order book depth levels to maintain.
    #[serde(default = "default_depth_levels")]
    pub depth_levels: usize,
    /// Reconnect parameters.
    pub reconnect: ReconnectConfig,
}

/// WebSocket reconnect parameters with exponential backoff.
#[derive(Debug, Clone, Deserialize)]
pub struct ReconnectConfig {
    /// Initial backoff delay in milliseconds.
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds.
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
    /// Maximum number of consecutive reconnect attempts before giving up.
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

/// Risk management configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RiskConfig {
    /// Maximum position size per symbol (in base currency units, e.g., BTC).
    pub max_position_size: f64,
    /// Maximum single order size.
    pub max_order_size: f64,
    /// Maximum orders per second.
    #[serde(default = "default_max_orders_per_second")]
    pub max_orders_per_second: u32,
    /// Daily loss limit in USD.
    pub daily_loss_limit_usd: f64,
    /// Fat-finger threshold — maximum deviation from mid price, in basis points.
    #[serde(default = "default_fat_finger_bps")]
    pub fat_finger_bps: u32,
}

/// QuestDB recorder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct RecorderConfig {
    /// QuestDB ILP endpoint (`host:port`).
    pub questdb_ilp_addr: String,
    /// Batch size before flush.
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    /// Flush interval in milliseconds.
    #[serde(default = "default_flush_interval_ms")]
    pub flush_interval_ms: u64,
}

/// Trading mode configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TradingConfig {
    /// `paper` for simulated trading, `live` for real money.
    pub mode: TradingMode,
    /// Strategy identifier to run.
    pub strategy: String,
}

/// Paper-trading simulation parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct PaperConfig {
    /// Simulated order latency in milliseconds.
    #[serde(default = "default_paper_latency_ms")]
    pub latency_ms: u64,
    /// Maker fee rate (negative = rebate). E.g., -0.0001 for -1 bps.
    #[serde(default = "default_paper_maker_fee")]
    pub maker_fee: f64,
    /// Taker fee rate. E.g., 0.0004 for 4 bps.
    #[serde(default = "default_paper_taker_fee")]
    pub taker_fee: f64,
    /// Maximum fraction of available book liquidity to fill against.
    #[serde(default = "default_paper_max_fill_fraction")]
    pub max_fill_fraction: f64,
}

/// Trading mode selector.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TradingMode {
    /// Simulated / paper trading.
    Paper,
    /// Live trading with real funds.
    Live,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            latency_ms: default_paper_latency_ms(),
            maker_fee: default_paper_maker_fee(),
            taker_fee: default_paper_taker_fee(),
            max_fill_fraction: default_paper_max_fill_fraction(),
        }
    }
}

impl AppConfig {
    /// Load configuration using layered sources.
    ///
    /// 1. Compiled-in sensible defaults (testnet URLs, conservative risk).
    /// 2. TOML file at `config_path` (if `Some`).
    /// 3. Environment variable overrides with prefix `CM_HFT_` and `__` as
    ///    the nesting separator (e.g., `CM_HFT_RISK__MAX_ORDER_SIZE=0.5`).
    /// 4. API keys from dedicated env vars: `BINANCE_API_KEY`,
    ///    `BINANCE_API_SECRET`, `BYBIT_API_KEY`, `BYBIT_API_SECRET`.
    ///
    /// After loading, validates that API keys are present when trading mode
    /// is `Live`.
    pub fn load(config_path: Option<PathBuf>) -> Result<Self> {
        let mut builder = Config::builder()
            // ── Layer 1: compiled-in defaults ───────────────────────
            // Binance testnet
            .set_default("binance.testnet", true)?
            .set_default("binance.ws_url", "wss://testnet.binance.vision/ws")?
            .set_default("binance.rest_url", "https://testnet.binance.vision")?
            .set_default("binance.timeout_ms", 5000i64)?
            .set_default("binance.api_key", "")?
            .set_default("binance.api_secret", "")?
            // Bybit testnet
            .set_default("bybit.testnet", true)?
            .set_default("bybit.ws_url", "wss://stream-testnet.bybit.com/v5/public/spot")?
            .set_default("bybit.rest_url", "https://api-testnet.bybit.com")?
            .set_default("bybit.timeout_ms", 5000i64)?
            .set_default("bybit.api_key", "")?
            .set_default("bybit.api_secret", "")?
            // Market data
            .set_default("market_data.symbols", vec!["BTCUSDT"])?
            .set_default("market_data.depth_levels", 20i64)?
            .set_default("market_data.reconnect.initial_backoff_ms", 1000i64)?
            .set_default("market_data.reconnect.max_backoff_ms", 30000i64)?
            .set_default("market_data.reconnect.max_retries", 10i64)?
            // Risk (conservative defaults)
            .set_default("risk.max_position_size", 0.1)?
            .set_default("risk.max_order_size", 0.01)?
            .set_default("risk.max_orders_per_second", 5i64)?
            .set_default("risk.daily_loss_limit_usd", 100.0)?
            .set_default("risk.fat_finger_bps", 50i64)?
            // Recorder
            .set_default("recorder.questdb_ilp_addr", "localhost:9009")?
            .set_default("recorder.batch_size", 1000i64)?
            .set_default("recorder.flush_interval_ms", 1000i64)?
            // Trading
            .set_default("trading.mode", "paper")?
            .set_default("trading.strategy", "simple_mm")?
            // Paper trading
            .set_default("paper.latency_ms", 1i64)?
            .set_default("paper.maker_fee", -0.0001)?
            .set_default("paper.taker_fee", 0.0004)?
            .set_default("paper.max_fill_fraction", 1.0)?;

        // ── Layer 2: TOML file ─────────────────────────────────────
        if let Some(path) = config_path {
            let path_str = path
                .to_str()
                .context("config path is not valid UTF-8")?;
            builder = builder.add_source(File::with_name(path_str).required(true));
        }

        // ── Layer 3: env var overrides (CM_HFT_ prefix) ───────────
        // The prefix separator must be set explicitly to `_` because the
        // `config` crate defaults it to the nesting separator when one is
        // provided.  Without this, `CM_HFT_RISK__MAX_ORDER_SIZE` would be
        // matched against prefix `cm_hft__` (double underscore) instead of
        // `cm_hft_` (single underscore).
        builder = builder.add_source(
            Environment::with_prefix("CM_HFT")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        let mut cfg: AppConfig = builder
            .build()
            .context("failed to build configuration")?
            .try_deserialize()
            .context("failed to deserialize configuration")?;

        // ── Layer 4: dedicated API key env vars ────────────────────
        if let Ok(v) = std::env::var("BINANCE_API_KEY") {
            cfg.binance.api_key = v;
        }
        if let Ok(v) = std::env::var("BINANCE_API_SECRET") {
            cfg.binance.api_secret = v;
        }
        if let Ok(v) = std::env::var("BYBIT_API_KEY") {
            cfg.bybit.api_key = v;
        }
        if let Ok(v) = std::env::var("BYBIT_API_SECRET") {
            cfg.bybit.api_secret = v;
        }

        // ── Validation ─────────────────────────────────────────────
        cfg.validate()?;

        Ok(cfg)
    }

    /// Validate configuration invariants.
    ///
    /// In live trading mode, all API keys and secrets must be non-empty.
    fn validate(&self) -> Result<()> {
        if self.trading.mode == TradingMode::Live {
            if self.binance.api_key.is_empty() || self.binance.api_secret.is_empty() {
                bail!("Binance API key and secret are required in live trading mode");
            }
            if self.bybit.api_key.is_empty() || self.bybit.api_secret.is_empty() {
                bail!("Bybit API key and secret are required in live trading mode");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Mutex;

    /// Global mutex to serialize tests that manipulate environment variables.
    /// Uses `unwrap_or_else` to recover from poisoned state so a panic in one
    /// test does not cascade to all others.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn lock_env() -> std::sync::MutexGuard<'static, ()> {
        ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Clear all env vars that could interfere with config loading.
    fn clear_env() {
        std::env::remove_var("CM_HFT_TRADING__MODE");
        std::env::remove_var("CM_HFT_RISK__MAX_ORDER_SIZE");
        std::env::remove_var("CM_HFT_RISK__MAX_POSITION_SIZE");
        std::env::remove_var("BINANCE_API_KEY");
        std::env::remove_var("BINANCE_API_SECRET");
        std::env::remove_var("BYBIT_API_KEY");
        std::env::remove_var("BYBIT_API_SECRET");
    }

    /// Helper: create a temporary TOML config file and return its path.
    ///
    /// Uses `.toml` suffix so the `config` crate auto-detects the format.
    fn write_temp_toml(content: &str) -> (tempfile::NamedTempFile, PathBuf) {
        let mut f = tempfile::Builder::new()
            .suffix(".toml")
            .tempfile()
            .expect("create temp file");
        write!(f, "{}", content).expect("write temp file");
        let path = f.path().to_path_buf();
        (f, path)
    }

    #[test]
    fn test_load_defaults_only() {
        let _lock = lock_env();
        clear_env();

        let cfg = AppConfig::load(None).expect("load defaults");
        assert!(cfg.binance.testnet);
        assert!(cfg.bybit.testnet);
        assert_eq!(cfg.trading.mode, TradingMode::Paper);
        assert_eq!(cfg.trading.strategy, "simple_mm");
        assert_eq!(cfg.market_data.symbols, vec!["BTCUSDT"]);
        assert_eq!(cfg.market_data.depth_levels, 20);
        assert_eq!(cfg.risk.max_orders_per_second, 5);
        assert_eq!(cfg.risk.fat_finger_bps, 50);
        assert_eq!(cfg.recorder.batch_size, 1000);
    }

    #[test]
    fn test_load_from_toml() {
        let _lock = lock_env();
        clear_env();

        let toml_content = r#"
[binance]
ws_url = "wss://custom.binance.com/ws"
rest_url = "https://custom.binance.com"
testnet = false

[bybit]
ws_url = "wss://custom.bybit.com/ws"
rest_url = "https://custom.bybit.com"

[market_data]
symbols = ["BTCUSDT", "ETHUSDT"]
depth_levels = 50

[market_data.reconnect]
initial_backoff_ms = 500
max_backoff_ms = 60000
max_retries = 20

[risk]
max_position_size = 1.0
max_order_size = 0.5
daily_loss_limit_usd = 5000.0

[recorder]
questdb_ilp_addr = "db.example.com:9009"

[trading]
mode = "paper"
strategy = "arb_v2"
"#;
        let (_f, path) = write_temp_toml(toml_content);
        let cfg = AppConfig::load(Some(path)).expect("load from toml");

        assert_eq!(cfg.binance.ws_url, "wss://custom.binance.com/ws");
        assert!(!cfg.binance.testnet);
        assert_eq!(cfg.market_data.symbols, vec!["BTCUSDT", "ETHUSDT"]);
        assert_eq!(cfg.market_data.depth_levels, 50);
        assert_eq!(cfg.market_data.reconnect.max_retries, 20);
        assert_eq!(cfg.risk.max_position_size, 1.0);
        assert_eq!(cfg.recorder.questdb_ilp_addr, "db.example.com:9009");
        assert_eq!(cfg.trading.strategy, "arb_v2");
    }

    #[test]
    fn test_env_var_overrides() {
        let _lock = lock_env();
        clear_env();
        std::env::set_var("CM_HFT_RISK__MAX_ORDER_SIZE", "0.25");

        let cfg = AppConfig::load(None).expect("load with env override");
        assert_eq!(cfg.risk.max_order_size, 0.25);

        std::env::remove_var("CM_HFT_RISK__MAX_ORDER_SIZE");
    }

    #[test]
    fn test_default_values() {
        let _lock = lock_env();
        clear_env();

        let cfg = AppConfig::load(None).expect("load defaults");

        // timeout_ms
        assert_eq!(cfg.binance.timeout_ms, 5000);
        assert_eq!(cfg.bybit.timeout_ms, 5000);
        // reconnect
        assert_eq!(cfg.market_data.reconnect.initial_backoff_ms, 1000);
        assert_eq!(cfg.market_data.reconnect.max_backoff_ms, 30000);
        assert_eq!(cfg.market_data.reconnect.max_retries, 10);
        // risk
        assert_eq!(cfg.risk.max_orders_per_second, 5);
        assert_eq!(cfg.risk.fat_finger_bps, 50);
        // recorder
        assert_eq!(cfg.recorder.batch_size, 1000);
        assert_eq!(cfg.recorder.flush_interval_ms, 1000);
    }

    #[test]
    fn test_live_mode_without_api_keys_fails() {
        let _lock = lock_env();
        clear_env();

        let toml_content = r#"
[trading]
mode = "live"
strategy = "arb_v2"
"#;
        let (_f, path) = write_temp_toml(toml_content);
        let result = AppConfig::load(Some(path));
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("API key"));
    }

    #[test]
    fn test_live_mode_with_api_keys_succeeds() {
        let _lock = lock_env();
        clear_env();
        std::env::set_var("BINANCE_API_KEY", "test_binance_key");
        std::env::set_var("BINANCE_API_SECRET", "test_binance_secret");
        std::env::set_var("BYBIT_API_KEY", "test_bybit_key");
        std::env::set_var("BYBIT_API_SECRET", "test_bybit_secret");

        let toml_content = r#"
[trading]
mode = "live"
strategy = "arb_v2"
"#;
        let (_f, path) = write_temp_toml(toml_content);
        let cfg = AppConfig::load(Some(path)).expect("load live mode with keys");

        assert_eq!(cfg.trading.mode, TradingMode::Live);
        assert_eq!(cfg.binance.api_key, "test_binance_key");
        assert_eq!(cfg.bybit.api_secret, "test_bybit_secret");

        clear_env();
    }

    #[test]
    fn test_api_keys_from_env() {
        let _lock = lock_env();
        clear_env();
        std::env::set_var("BINANCE_API_KEY", "bn_key_123");
        std::env::set_var("BINANCE_API_SECRET", "bn_sec_456");

        let cfg = AppConfig::load(None).expect("load with api key env");
        assert_eq!(cfg.binance.api_key, "bn_key_123");
        assert_eq!(cfg.binance.api_secret, "bn_sec_456");

        clear_env();
    }
}
