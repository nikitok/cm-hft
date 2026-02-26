//! Paper trading executor — simulates order execution against live market data.
//!
//! Implements the [`ExchangeGateway`] trait, providing a drop-in replacement
//! for real exchange connectivity during paper trading.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use crossbeam::channel::Sender;
use parking_lot::Mutex;

use cm_core::config::PaperConfig;
use cm_core::types::*;
use cm_execution::gateway::{AmendAck, CancelAck, ExchangeGateway, NewOrder, OrderAck};
use cm_market_data::orderbook::OrderBook;

/// A raw fill from the paper executor, carrying the `client_order_id` so the
/// engine can resolve the internal `OrderId`.
#[derive(Debug, Clone)]
pub struct RawFill {
    pub client_order_id: String,
    pub exchange: Exchange,
    pub symbol: Symbol,
    pub side: Side,
    pub price: Price,
    pub quantity: Quantity,
    pub timestamp: Timestamp,
    pub is_maker: bool,
}

/// A resting order in the paper executor.
#[derive(Debug, Clone)]
struct RestingOrder {
    client_order_id: String,
    exchange: Exchange,
    symbol: Symbol,
    side: Side,
    price: Price,
    remaining: Quantity,
}

/// Paper trading executor that simulates fills against live market data.
pub struct PaperExecutor {
    config: PaperConfig,
    fill_tx: Sender<RawFill>,
    next_id: AtomicU64,
    /// Resting bid orders keyed by exchange_id.
    resting_bids: Mutex<BTreeMap<String, RestingOrder>>,
    /// Resting ask orders keyed by exchange_id.
    resting_asks: Mutex<BTreeMap<String, RestingOrder>>,
    /// Current best bid from market data.
    best_bid: Mutex<Option<(Price, Quantity)>>,
    /// Current best ask from market data.
    best_ask: Mutex<Option<(Price, Quantity)>>,
}

impl PaperExecutor {
    /// Create a new paper executor.
    pub fn new(config: PaperConfig, fill_tx: Sender<RawFill>) -> Self {
        Self {
            config,
            fill_tx,
            next_id: AtomicU64::new(1),
            resting_bids: Mutex::new(BTreeMap::new()),
            resting_asks: Mutex::new(BTreeMap::new()),
            best_bid: Mutex::new(None),
            best_ask: Mutex::new(None),
        }
    }

    fn next_exchange_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("PAPER-{}", n)
    }

    /// Called on each book update to check resting orders for fills.
    pub fn update_market_data(&self, book: &OrderBook) {
        let bid = book.best_bid().map(|l| (l.price, l.quantity));
        let ask = book.best_ask().map(|l| (l.price, l.quantity));

        *self.best_bid.lock() = bid;
        *self.best_ask.lock() = ask;

        // Match resting bids against incoming best ask
        if let Some((ask_price, ask_qty)) = ask {
            let mut bids = self.resting_bids.lock();
            let mut to_remove = Vec::new();
            let max_fill_qty = ask_qty.to_f64() * self.config.max_fill_fraction;
            let mut available = max_fill_qty;

            for (id, order) in bids.iter_mut() {
                if available <= 0.0 {
                    break;
                }
                if order.price >= ask_price {
                    let remaining_f64 = order.remaining.to_f64();
                    let fill_qty = remaining_f64.min(available);
                    available -= fill_qty;

                    let _ = self.fill_tx.send(RawFill {
                        client_order_id: order.client_order_id.clone(),
                        exchange: order.exchange,
                        symbol: order.symbol.clone(),
                        side: Side::Buy,
                        price: order.price,
                        quantity: Quantity::from(fill_qty),
                        timestamp: Timestamp::now(),
                        is_maker: true,
                    });

                    if fill_qty >= remaining_f64 - 1e-12 {
                        to_remove.push(id.clone());
                    } else {
                        order.remaining = Quantity::from(remaining_f64 - fill_qty);
                    }
                }
            }
            for id in to_remove {
                bids.remove(&id);
            }
        }

        // Match resting asks against incoming best bid
        if let Some((bid_price, bid_qty)) = bid {
            let mut asks = self.resting_asks.lock();
            let mut to_remove = Vec::new();
            let max_fill_qty = bid_qty.to_f64() * self.config.max_fill_fraction;
            let mut available = max_fill_qty;

            for (id, order) in asks.iter_mut() {
                if available <= 0.0 {
                    break;
                }
                if order.price <= bid_price {
                    let remaining_f64 = order.remaining.to_f64();
                    let fill_qty = remaining_f64.min(available);
                    available -= fill_qty;

                    let _ = self.fill_tx.send(RawFill {
                        client_order_id: order.client_order_id.clone(),
                        exchange: order.exchange,
                        symbol: order.symbol.clone(),
                        side: Side::Sell,
                        price: order.price,
                        quantity: Quantity::from(fill_qty),
                        timestamp: Timestamp::now(),
                        is_maker: true,
                    });

                    if fill_qty >= remaining_f64 - 1e-12 {
                        to_remove.push(id.clone());
                    } else {
                        order.remaining = Quantity::from(remaining_f64 - fill_qty);
                    }
                }
            }
            for id in to_remove {
                asks.remove(&id);
            }
        }
    }
}

#[async_trait]
impl ExchangeGateway for PaperExecutor {
    async fn place_order(&self, order: &NewOrder) -> Result<OrderAck> {
        tokio::time::sleep(Duration::from_millis(self.config.latency_ms)).await;

        let exchange_id = self.next_exchange_id();

        let crosses = match order.side {
            Side::Buy => {
                let ask = self.best_ask.lock().clone();
                ask.map_or(false, |(ask_price, _)| order.price >= ask_price)
            }
            Side::Sell => {
                let bid = self.best_bid.lock().clone();
                bid.map_or(false, |(bid_price, _)| order.price <= bid_price)
            }
        };

        let immediate_fill = match order.order_type {
            OrderType::Market => true,
            OrderType::Limit => crosses,
            OrderType::PostOnly => {
                if crosses {
                    // PostOnly orders that would cross are rejected, not filled
                    return Err(anyhow::anyhow!(
                        "PostOnly order would cross the book (price={}, side={:?})",
                        order.price,
                        order.side
                    ));
                }
                false
            }
        };

        if immediate_fill {
            let fill_price = match order.side {
                Side::Buy => self
                    .best_ask
                    .lock()
                    .map(|(p, _)| p)
                    .unwrap_or(order.price),
                Side::Sell => self
                    .best_bid
                    .lock()
                    .map(|(p, _)| p)
                    .unwrap_or(order.price),
            };

            let _ = self.fill_tx.send(RawFill {
                client_order_id: order.client_order_id.clone(),
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: fill_price,
                quantity: order.quantity,
                timestamp: Timestamp::now(),
                is_maker: false,
            });
        } else {
            let resting = RestingOrder {
                client_order_id: order.client_order_id.clone(),
                exchange: order.exchange,
                symbol: order.symbol.clone(),
                side: order.side,
                price: order.price,
                remaining: order.quantity,
            };

            match order.side {
                Side::Buy => {
                    self.resting_bids
                        .lock()
                        .insert(exchange_id.clone(), resting);
                }
                Side::Sell => {
                    self.resting_asks
                        .lock()
                        .insert(exchange_id.clone(), resting);
                }
            }
        }

        Ok(OrderAck {
            exchange_order_id: exchange_id,
            client_order_id: order.client_order_id.clone(),
        })
    }

    async fn cancel_order(
        &self,
        _exchange: Exchange,
        _symbol: &str,
        order_id: &str,
    ) -> Result<CancelAck> {
        self.resting_bids.lock().remove(order_id);
        self.resting_asks.lock().remove(order_id);

        Ok(CancelAck {
            order_id: order_id.to_string(),
        })
    }

    async fn amend_order(
        &self,
        exchange: Exchange,
        symbol: &str,
        order_id: &str,
        _new_price: Option<Price>,
        _new_qty: Option<Quantity>,
    ) -> Result<AmendAck> {
        self.cancel_order(exchange, symbol, order_id).await?;
        Ok(AmendAck {
            order_id: order_id.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel;

    fn make_config() -> PaperConfig {
        PaperConfig {
            latency_ms: 0,
            maker_fee: -0.0001,
            taker_fee: 0.0004,
            max_fill_fraction: 1.0,
        }
    }

    #[tokio::test]
    async fn test_market_order_immediate_fill() {
        let (fill_tx, fill_rx) = channel::unbounded();
        let executor = PaperExecutor::new(make_config(), fill_tx);

        *executor.best_ask.lock() = Some((Price::from(50000.0), Quantity::from(1.0)));

        let order = NewOrder {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Market,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.1),
            client_order_id: "test-001".to_string(),
        };

        let ack = executor.place_order(&order).await.unwrap();
        assert!(ack.exchange_order_id.starts_with("PAPER-"));

        let fill = fill_rx.try_recv().unwrap();
        assert_eq!(fill.side, Side::Buy);
        assert!(!fill.is_maker);
        assert_eq!(fill.client_order_id, "test-001");
    }

    #[tokio::test]
    async fn test_limit_order_rests() {
        let (fill_tx, fill_rx) = channel::unbounded();
        let executor = PaperExecutor::new(make_config(), fill_tx);

        *executor.best_ask.lock() = Some((Price::from(50000.0), Quantity::from(1.0)));

        let order = NewOrder {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::from(49990.0),
            quantity: Quantity::from(0.1),
            client_order_id: "test-002".to_string(),
        };

        executor.place_order(&order).await.unwrap();
        assert!(fill_rx.try_recv().is_err());
        assert_eq!(executor.resting_bids.lock().len(), 1);
    }

    #[tokio::test]
    async fn test_resting_order_fills_on_book_update() {
        let (fill_tx, fill_rx) = channel::unbounded();
        let executor = PaperExecutor::new(make_config(), fill_tx);

        *executor.best_ask.lock() = Some((Price::from(50001.0), Quantity::from(1.0)));
        let order = NewOrder {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.1),
            client_order_id: "test-003".to_string(),
        };
        executor.place_order(&order).await.unwrap();
        assert!(fill_rx.try_recv().is_err());

        let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
        book.apply_snapshot(
            &[(Price::from(49999.0), Quantity::from(1.0))],
            &[(Price::from(50000.0), Quantity::from(1.0))],
            1,
        );
        executor.update_market_data(&book);

        let fill = fill_rx.try_recv().unwrap();
        assert_eq!(fill.side, Side::Buy);
        assert!(fill.is_maker);
        assert_eq!(fill.client_order_id, "test-003");
    }

    #[tokio::test]
    async fn test_cancel_order() {
        let (fill_tx, _fill_rx) = channel::unbounded();
        let executor = PaperExecutor::new(make_config(), fill_tx);

        *executor.best_ask.lock() = Some((Price::from(50001.0), Quantity::from(1.0)));
        let order = NewOrder {
            exchange: Exchange::Binance,
            symbol: Symbol::new("BTCUSDT"),
            side: Side::Buy,
            order_type: OrderType::Limit,
            price: Price::from(50000.0),
            quantity: Quantity::from(0.1),
            client_order_id: "test-004".to_string(),
        };
        let ack = executor.place_order(&order).await.unwrap();
        assert_eq!(executor.resting_bids.lock().len(), 1);

        executor
            .cancel_order(Exchange::Binance, "BTCUSDT", &ack.exchange_order_id)
            .await
            .unwrap();
        assert_eq!(executor.resting_bids.lock().len(), 0);
    }

    #[test]
    fn test_exchange_id_monotonic() {
        let (fill_tx, _) = channel::unbounded();
        let executor = PaperExecutor::new(make_config(), fill_tx);
        let id1 = executor.next_exchange_id();
        let id2 = executor.next_exchange_id();
        assert_eq!(id1, "PAPER-1");
        assert_eq!(id2, "PAPER-2");
    }
}
