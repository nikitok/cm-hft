//! Benchmarks for `OrderBook` operations using criterion.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use cm_core::types::{BookUpdate, Exchange, Price, Quantity, Symbol, Timestamp};
use cm_market_data::orderbook::OrderBook;

/// Build a pre-populated order book with `n` levels on each side.
fn populated_book(n: usize) -> OrderBook {
    let bids: Vec<(Price, Quantity)> = (0..n)
        .map(|i| {
            (
                Price::new(5000000 - (i as i64) * 100, 2),
                Quantity::new(100000 + (i as i64) * 1000, 8),
            )
        })
        .collect();

    let asks: Vec<(Price, Quantity)> = (0..n)
        .map(|i| {
            (
                Price::new(5000100 + (i as i64) * 100, 2),
                Quantity::new(100000 + (i as i64) * 1000, 8),
            )
        })
        .collect();

    let mut book = OrderBook::new(Exchange::Binance, Symbol::new("BTCUSDT"));
    book.apply_snapshot(&bids, &asks, 0);
    book
}

fn bench_apply_update(c: &mut Criterion) {
    let mut book = populated_book(100);
    let mut update_id = 1u64;

    // Typical update: 5 bid + 5 ask levels
    let update = BookUpdate {
        exchange: Exchange::Binance,
        symbol: Symbol::new("BTCUSDT"),
        timestamp: Timestamp::from_millis(1706000000000),
        bids: (0..5)
            .map(|i| {
                (
                    Price::new(4999500 + i * 100, 2),
                    Quantity::new(50000 + i * 1000, 8),
                )
            })
            .collect(),
        asks: (0..5)
            .map(|i| {
                (
                    Price::new(5000600 + i * 100, 2),
                    Quantity::new(50000 + i * 1000, 8),
                )
            })
            .collect(),
        is_snapshot: false,
    };

    c.bench_function("apply_update_5x5", |b| {
        b.iter(|| {
            update_id += 1;
            book.apply_update(black_box(&update), update_id).unwrap();
        })
    });
}

fn bench_best_bid(c: &mut Criterion) {
    let book = populated_book(100);

    c.bench_function("best_bid", |b| {
        b.iter(|| {
            black_box(book.best_bid());
        })
    });
}

fn bench_best_ask(c: &mut Criterion) {
    let book = populated_book(100);

    c.bench_function("best_ask", |b| {
        b.iter(|| {
            black_box(book.best_ask());
        })
    });
}

fn bench_mid_price(c: &mut Criterion) {
    let book = populated_book(100);

    c.bench_function("mid_price", |b| {
        b.iter(|| {
            black_box(book.mid_price());
        })
    });
}

fn bench_spread_bps(c: &mut Criterion) {
    let book = populated_book(100);

    c.bench_function("spread_bps", |b| {
        b.iter(|| {
            black_box(book.spread_bps());
        })
    });
}

fn bench_bid_depth_10(c: &mut Criterion) {
    let book = populated_book(100);

    c.bench_function("bid_depth_10", |b| {
        b.iter(|| {
            black_box(book.bid_depth(10));
        })
    });
}

criterion_group!(
    benches,
    bench_apply_update,
    bench_best_bid,
    bench_best_ask,
    bench_mid_price,
    bench_spread_bps,
    bench_bid_depth_10,
);
criterion_main!(benches);
