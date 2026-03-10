//! Integration tests for the symbol analysis pipeline.
//!
//! Tests the full SymbolAnalyzer pipeline with synthetic data — no live WebSocket
//! connection or network access required.

use std::time::Duration;

use cm_core::types::{Exchange, Price, Quantity, Side, Symbol, Timestamp, Trade};
use cm_market_data::orderbook::OrderBook;
use cm_trading::analyzer::{AnalysisReport, AnalysisVerdict, SymbolAnalyzer, Thresholds};

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn make_book(bid: f64, ask: f64, qty: f64) -> OrderBook {
    let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
    let bids: Vec<_> = (0..5)
        .map(|i| (Price::from(bid - i as f64), Quantity::from(qty)))
        .collect();
    let asks: Vec<_> = (0..5)
        .map(|i| (Price::from(ask + i as f64), Quantity::from(qty)))
        .collect();
    book.apply_snapshot(&bids, &asks, 1);
    book
}

fn make_trade(price: f64, qty: f64) -> Trade {
    Trade {
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        timestamp: Timestamp::from_millis(0),
        price: Price::from(price),
        quantity: Quantity::from(qty),
        side: Side::Buy,
        trade_id: "1".to_string(),
    }
}

/// Simulate 30 seconds of market data: 300 book updates and 30 trades.
fn simulate_30s(analyzer: &mut SymbolAnalyzer) {
    let book = make_book(50000.0, 50001.0, 1.0);
    for _ in 0..300 {
        analyzer.on_book_update(&book);
    }
    for _ in 0..30 {
        analyzer.on_trade(&make_trade(50000.5, 0.01));
    }
}

// ─── Integration Test 1: Full pipeline with synthetic data ───────────────────

#[test]
fn test_full_pipeline_synthetic_data() {
    let mut analyzer = SymbolAnalyzer::new();
    simulate_30s(&mut analyzer);

    let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());

    // Verify report structure has all expected fields populated.
    assert_eq!(report.book_samples, 300, "book samples");
    assert_eq!(report.trade_count, 30, "trade count");
    assert!(report.spread_bps_mean > 0.0, "spread should be positive");
    assert!(
        report.bid_depth_usd_mean > 0.0,
        "bid depth should be positive"
    );
    assert!(
        report.ask_depth_usd_mean > 0.0,
        "ask depth should be positive"
    );
    assert!(
        report.bid_ask_imbalance_mean >= 0.0 && report.bid_ask_imbalance_mean <= 1.0,
        "imbalance should be in [0, 1]"
    );
    assert!(report.duration_secs > 0.0, "duration should be positive");
    assert!(
        report.trade_rate_per_sec > 0.0,
        "trade rate should be positive"
    );
    assert!(report.trade_volume_usd > 0.0, "volume should be positive");
}

// ─── Integration Test 2: JSON roundtrip ──────────────────────────────────────

#[test]
fn test_report_json_roundtrip() {
    let mut analyzer = SymbolAnalyzer::new();
    simulate_30s(&mut analyzer);

    let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());

    // Serialize to JSON.
    let json = serde_json::to_string_pretty(&report).expect("serialize report to JSON");
    assert!(!json.is_empty());

    // Verify JSON contains expected fields.
    assert!(
        json.contains("\"book_samples\""),
        "JSON missing book_samples"
    );
    assert!(json.contains("\"trade_count\""), "JSON missing trade_count");
    assert!(json.contains("\"verdict\""), "JSON missing verdict");
    assert!(
        json.contains("\"spread_bps_mean\""),
        "JSON missing spread_bps_mean"
    );
    assert!(
        json.contains("\"bid_depth_usd_mean\""),
        "JSON missing bid_depth_usd_mean"
    );

    // Deserialize back and verify numeric fields match.
    let deserialized: AnalysisReport = serde_json::from_str(&json).expect("deserialize report");
    assert_eq!(deserialized.book_samples, report.book_samples);
    assert_eq!(deserialized.trade_count, report.trade_count);
    assert!((deserialized.spread_bps_mean - report.spread_bps_mean).abs() < 1e-10);
    assert!((deserialized.bid_depth_usd_mean - report.bid_depth_usd_mean).abs() < 1e-10);
    assert!((deserialized.trade_rate_per_sec - report.trade_rate_per_sec).abs() < 1e-10);
}

// ─── Integration Test 3: Only books, no trades → Insufficient ────────────────

#[test]
fn test_only_books_no_trades() {
    let mut analyzer = SymbolAnalyzer::new();
    let book = make_book(50000.0, 50001.0, 1.0);
    for _ in 0..50 {
        analyzer.on_book_update(&book);
    }
    // No trades at all.
    let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());
    assert!(
        matches!(report.verdict, AnalysisVerdict::Insufficient { .. }),
        "no trades should be Insufficient, got {:?}",
        report.verdict
    );
    assert_eq!(report.trade_count, 0);
}

// ─── Integration Test 4: Only trades, no books → Insufficient ────────────────

#[test]
fn test_only_trades_no_books() {
    let mut analyzer = SymbolAnalyzer::new();
    for _ in 0..20 {
        analyzer.on_trade(&make_trade(50000.5, 0.01));
    }
    // No book updates at all.
    let report = analyzer.report(Duration::from_secs(30), &Thresholds::default());
    assert!(
        matches!(report.verdict, AnalysisVerdict::Insufficient { .. }),
        "no book updates should be Insufficient, got {:?}",
        report.verdict
    );
    assert_eq!(report.book_samples, 0);
}

// ─── Integration Test 5: Very short duration ─────────────────────────────────

#[test]
fn test_very_short_duration_no_panic() {
    let mut analyzer = SymbolAnalyzer::new();
    let book = make_book(50000.0, 50001.0, 1.0);
    // Only 3 book updates — well below the minimum of 10.
    for _ in 0..3 {
        analyzer.on_book_update(&book);
    }
    analyzer.on_trade(&make_trade(50000.5, 0.01));

    // Should produce Insufficient without panicking.
    let report = analyzer.report(Duration::from_millis(100), &Thresholds::default());
    assert!(
        matches!(report.verdict, AnalysisVerdict::Insufficient { .. }),
        "insufficient data should not panic"
    );
}

// ─── Integration Test 6: Verdict correctness end-to-end ──────────────────────

#[test]
fn test_verdict_go_with_relaxed_thresholds() {
    let mut analyzer = SymbolAnalyzer::new();
    simulate_30s(&mut analyzer);

    let relaxed = Thresholds {
        max_spread_bps: 10000.0, // far above any realistic spread
        min_trade_rate: 0.0,
        min_depth_usd: 0.0,
        max_volatility_bps: 10000.0,
    };

    let report = analyzer.report(Duration::from_secs(30), &relaxed);
    assert_eq!(
        report.verdict,
        AnalysisVerdict::Go,
        "relaxed thresholds should produce Go"
    );
}

// ─── Integration Test 7: Consecutive book updates accumulate correctly ────────

#[test]
fn test_consecutive_book_updates_accumulation() {
    let mut analyzer = SymbolAnalyzer::new();

    // 200 updates of narrow spread, 100 of wide spread.
    let narrow = make_book(50000.0, 50001.0, 1.0);
    let wide = make_book(40000.0, 60000.0, 1.0);

    for _ in 0..200 {
        analyzer.on_book_update(&narrow);
    }
    for _ in 0..100 {
        analyzer.on_book_update(&wide);
    }
    for _ in 0..10 {
        analyzer.on_trade(&make_trade(50000.0, 0.1));
    }

    let report = analyzer.report(Duration::from_secs(60), &Thresholds::default());
    assert_eq!(report.book_samples, 300);

    // Mean spread should be between narrow and wide (weighted 2:1 toward narrow).
    let narrow_bps = narrow.spread_bps().unwrap();
    let wide_bps = wide.spread_bps().unwrap();
    let expected_mean = (200.0 * narrow_bps + 100.0 * wide_bps) / 300.0;
    assert!(
        (report.spread_bps_mean - expected_mean).abs() < 1.0,
        "mean spread {:.2} should be close to weighted mean {:.2}",
        report.spread_bps_mean,
        expected_mean
    );
}
