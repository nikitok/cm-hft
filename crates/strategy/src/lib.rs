//! # cm-strategy
//!
//! Strategy trait definitions and implementations. Strategies are compiled Rust
//! code that react to market data events and produce order actions through a
//! [`TradingContext`]. The hot path is synchronous, lock-free, and
//! allocation-free.

pub mod context;
pub mod loader;
pub mod strategies;
pub mod traits;

pub use context::{OrderAction, TradingContext};
pub use loader::{default_registry, StrategyRegistry};
pub use traits::{Fill, Strategy, StrategyParams};

#[cfg(test)]
mod tests {
    use super::*;
    use cm_core::types::*;
    use cm_market_data::orderbook::OrderBook;
    use cm_oms::{Order, OrderStatus, Position};

    // ── TradingContext tests ──

    #[test]
    fn test_context_submit_order_buffered() {
        let mut ctx = TradingContext::new(vec![], vec![], Timestamp::from_millis(1000));

        ctx.submit_order(
            Exchange::Binance,
            Symbol::new("BTCUSDT"),
            Side::Buy,
            OrderType::Limit,
            Price::from(50000.0),
            Quantity::from(0.001),
        );

        assert_eq!(ctx.action_count(), 1);
        let actions = ctx.drain_actions();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            OrderAction::Submit {
                exchange,
                symbol,
                side,
                order_type,
                ..
            } => {
                assert_eq!(*exchange, Exchange::Binance);
                assert_eq!(*symbol, Symbol::new("BTCUSDT"));
                assert_eq!(*side, Side::Buy);
                assert_eq!(*order_type, OrderType::Limit);
            }
            _ => panic!("expected Submit action"),
        }
    }

    #[test]
    fn test_context_cancel_order_buffered() {
        let mut ctx = TradingContext::new(vec![], vec![], Timestamp::from_millis(1000));
        ctx.cancel_order(OrderId(42));

        assert_eq!(ctx.action_count(), 1);
        let actions = ctx.drain_actions();
        match &actions[0] {
            OrderAction::Cancel { order_id } => assert_eq!(*order_id, OrderId(42)),
            _ => panic!("expected Cancel action"),
        }
    }

    #[test]
    fn test_context_cancel_all_buffered() {
        let mut ctx = TradingContext::new(vec![], vec![], Timestamp::from_millis(1000));
        ctx.cancel_all(Some(Exchange::Binance), Some(Symbol::new("BTCUSDT")));

        assert_eq!(ctx.action_count(), 1);
        let actions = ctx.drain_actions();
        match &actions[0] {
            OrderAction::CancelAll { exchange, symbol } => {
                assert_eq!(*exchange, Some(Exchange::Binance));
                assert_eq!(*symbol, Some(Symbol::new("BTCUSDT")));
            }
            _ => panic!("expected CancelAll action"),
        }
    }

    #[test]
    fn test_context_drain_clears_actions() {
        let mut ctx = TradingContext::new(vec![], vec![], Timestamp::from_millis(1000));
        ctx.submit_order(
            Exchange::Binance,
            Symbol::new("BTCUSDT"),
            Side::Buy,
            OrderType::Limit,
            Price::from(50000.0),
            Quantity::from(0.001),
        );
        ctx.cancel_order(OrderId(1));

        let actions = ctx.drain_actions();
        assert_eq!(actions.len(), 2);
        assert_eq!(ctx.action_count(), 0);
        assert!(ctx.drain_actions().is_empty());
    }

    #[test]
    fn test_context_get_position() {
        let pos = Position {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            net_quantity: Quantity::from(1.0),
            avg_entry_price: Price::from(50000.0),
            realized_pnl: Price::zero(8),
            fill_count: 1,
        };
        let ctx = TradingContext::new(vec![pos], vec![], Timestamp::from_millis(1000));

        let found = ctx.get_position(&Exchange::Binance, &Symbol::new("BTCUSDT"));
        assert!(found.is_some());
        assert!((found.unwrap().net_quantity.to_f64() - 1.0).abs() < 1e-6);

        let missing = ctx.get_position(&Exchange::Bybit, &Symbol::new("BTCUSDT"));
        assert!(missing.is_none());
    }

    #[test]
    fn test_context_get_open_orders() {
        let order = Order {
            id: OrderId(1),
            client_order_id: "test_001".to_string(),
            exchange_order_id: None,
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.001),
            filled_quantity: Quantity::zero(8),
            status: OrderStatus::Acked,
            created_at: Timestamp::from_millis(1000),
            updated_at: Timestamp::from_millis(1000),
        };
        let ctx = TradingContext::new(vec![], vec![order], Timestamp::from_millis(1000));

        let orders = ctx.get_open_orders();
        assert_eq!(orders.len(), 1);
        assert_eq!(orders[0].id, OrderId(1));
    }

    #[test]
    fn test_context_net_position() {
        let pos = Position {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            net_quantity: Quantity::from(-0.5),
            avg_entry_price: Price::from(50000.0),
            realized_pnl: Price::zero(8),
            fill_count: 1,
        };
        let ctx = TradingContext::new(vec![pos], vec![], Timestamp::from_millis(1000));

        let net = ctx.net_position(&Exchange::Binance, &Symbol::new("BTCUSDT"));
        assert!((net - (-0.5)).abs() < 1e-6);

        let net_missing = ctx.net_position(&Exchange::Bybit, &Symbol::new("BTCUSDT"));
        assert!((net_missing).abs() < 1e-10);
    }

    // ── StrategyParams tests ──

    #[test]
    fn test_params_get_f64() {
        let params = StrategyParams {
            params: serde_json::json!({ "spread": 5.0 }),
        };
        assert!((params.get_f64("spread").unwrap() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_params_get_i64() {
        let params = StrategyParams {
            params: serde_json::json!({ "levels": 3 }),
        };
        assert_eq!(params.get_i64("levels").unwrap(), 3);
    }

    #[test]
    fn test_params_get_bool() {
        let params = StrategyParams {
            params: serde_json::json!({ "enabled": true }),
        };
        assert!(params.get_bool("enabled").unwrap());
    }

    #[test]
    fn test_params_missing_key() {
        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        assert!(params.get_f64("missing").is_none());
        assert!(params.get_i64("missing").is_none());
        assert!(params.get_bool("missing").is_none());
    }

    #[test]
    fn test_params_wrong_type() {
        let params = StrategyParams {
            params: serde_json::json!({ "value": "not_a_number" }),
        };
        assert!(params.get_f64("value").is_none());
        assert!(params.get_i64("value").is_none());
        assert!(params.get_bool("value").is_none());
    }

    // ── StrategyRegistry tests ──

    #[test]
    fn test_registry_register_and_create() {
        let mut registry = StrategyRegistry::new();
        registry.register("test_strategy", |_params| {
            Box::new(strategies::MarketMakingStrategy::from_params(_params))
        });

        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        let strat = registry.create("test_strategy", &params);
        assert!(strat.is_some());
        assert_eq!(strat.unwrap().name(), "market_making");
    }

    #[test]
    fn test_registry_unknown_strategy() {
        let registry = StrategyRegistry::new();
        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        assert!(registry.create("nonexistent", &params).is_none());
    }

    #[test]
    fn test_registry_available_strategies() {
        let registry = default_registry();
        let names = registry.available_strategies();
        assert!(names.contains(&"market_making".to_string()));
    }

    #[test]
    fn test_default_registry_creates_market_making() {
        let registry = default_registry();
        let params = StrategyParams {
            params: serde_json::json!({}),
        };
        let strat = registry.create("market_making", &params);
        assert!(strat.is_some());
        assert_eq!(strat.unwrap().name(), "market_making");
    }

    // ── Integration test: strategy through context ──

    #[test]
    fn test_strategy_produces_actions_through_context() {
        let mut strat = strategies::MarketMakingStrategy::from_params(&StrategyParams {
            params: serde_json::json!({}),
        });
        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        book.apply_snapshot(
            &[(Price::from(50000.0), Quantity::from(1.0))],
            &[(Price::from(50001.0), Quantity::from(1.0))],
            1,
        );

        let mut ctx = TradingContext::new(vec![], vec![], Timestamp::from_millis(1000));
        strat.on_book_update(&mut ctx, &book);

        let actions = ctx.drain_actions();
        assert!(!actions.is_empty());

        // Verify we got CancelAll + Submit actions
        let submits: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, OrderAction::Submit { .. }))
            .collect();
        assert!(
            submits.len() >= 2,
            "expected at least 2 Submit actions, got {}",
            submits.len()
        );
    }
}
