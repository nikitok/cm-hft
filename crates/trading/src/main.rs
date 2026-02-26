//! CM.HFT Trading Binary
//!
//! Entry point for the paper and live trading engine. Loads configuration,
//! initializes tracing, and starts the [`TradingEngine`].

use std::path::PathBuf;

use clap::Parser;

use cm_core::config::AppConfig;
use cm_trading::engine::TradingEngine;

/// CM.HFT Trading Engine
#[derive(Parser, Debug)]
#[command(name = "cm-trading", about = "CM.HFT trading engine")]
struct Args {
    /// Path to TOML configuration file.
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = AppConfig::load(args.config)?;

    cm_core::logging::init_tracing(true);

    tracing::info!(
        mode = ?config.trading.mode,
        strategy = %config.trading.strategy,
        symbols = ?config.market_data.symbols,
        "starting cm-trading"
    );

    let engine = TradingEngine::new(config).await?;
    engine.run().await
}
