//! Token-bucket rate limiter for exchange API endpoints.
//!
//! Each exchange enforces request weight or rate limits. This module provides a
//! thread-safe token-bucket implementation that refills at a configurable rate
//! and supports both non-blocking (`try_acquire`) and blocking (`acquire`) modes.

use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Token-bucket rate limiter per endpoint category.
///
/// Tokens are consumed on each request (weighted by endpoint cost) and refill
/// continuously at `refill_rate` tokens per second. The bucket never exceeds
/// `max_tokens` (burst capacity).
pub struct RateLimiter {
    /// Maximum tokens (burst capacity).
    max_tokens: u32,
    /// Tokens refilled per second.
    refill_rate: f64,
    /// Current token count (fractional to allow smooth refill).
    tokens: Mutex<f64>,
    /// Last refill timestamp.
    last_refill: Mutex<Instant>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// - `max_tokens`: burst capacity (maximum tokens in the bucket).
    /// - `refill_rate`: tokens added per second.
    pub fn new(max_tokens: u32, refill_rate: f64) -> Self {
        Self {
            max_tokens,
            refill_rate,
            tokens: Mutex::new(max_tokens as f64),
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Try to consume `weight` tokens without blocking.
    ///
    /// Returns `true` if tokens were consumed, `false` if insufficient tokens
    /// are available. Always refills before checking.
    pub fn try_acquire(&self, weight: u32) -> bool {
        self.refill();
        let mut tokens = self.tokens.lock();
        let needed = weight as f64;
        if *tokens >= needed {
            *tokens -= needed;
            true
        } else {
            false
        }
    }

    /// Blocking wait until `weight` tokens are available, then consume them.
    ///
    /// Spins with a short sleep to avoid busy-waiting while keeping latency low.
    pub fn acquire(&self, weight: u32) {
        loop {
            if self.try_acquire(weight) {
                return;
            }
            // Sleep proportional to the deficit to avoid busy-waiting.
            let tokens = *self.tokens.lock();
            let deficit = weight as f64 - tokens;
            if deficit > 0.0 && self.refill_rate > 0.0 {
                let wait_secs = deficit / self.refill_rate;
                std::thread::sleep(Duration::from_secs_f64(wait_secs.min(0.1)));
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Return the current number of available tokens (truncated to integer).
    pub fn available_tokens(&self) -> u32 {
        self.refill();
        let tokens = self.tokens.lock();
        *tokens as u32
    }

    /// Current usage as a percentage (0.0 = empty bucket, 100.0 = fully consumed).
    ///
    /// Returns 100.0 when no tokens remain and 0.0 when the bucket is full.
    pub fn usage_percent(&self) -> f64 {
        self.refill();
        let tokens = *self.tokens.lock();
        let used = self.max_tokens as f64 - tokens;
        (used / self.max_tokens as f64 * 100.0).clamp(0.0, 100.0)
    }

    /// Synchronize token count from exchange response headers.
    ///
    /// Binance returns `X-MBX-USED-WEIGHT-1M` indicating how much weight has
    /// been used in the current minute. This method sets the remaining tokens
    /// to `max_tokens - used_weight`.
    pub fn update_from_header(&self, used_weight: u32) {
        let remaining = if used_weight >= self.max_tokens {
            0.0
        } else {
            (self.max_tokens - used_weight) as f64
        };
        let mut tokens = self.tokens.lock();
        *tokens = remaining;
        // Reset the refill timer so we don't double-count.
        let mut last = self.last_refill.lock();
        *last = Instant::now();
    }

    /// Preconfigured rate limiter for Binance spot API.
    ///
    /// Binance allows 1200 request weight per minute (= 20 weight/second).
    pub fn binance_default() -> Self {
        Self::new(1200, 20.0)
    }

    /// Preconfigured rate limiter for Bybit v5 API.
    ///
    /// Bybit allows 120 requests per second.
    pub fn bybit_default() -> Self {
        Self::new(120, 120.0)
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&self) {
        let mut last = self.last_refill.lock();
        let now = Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        if elapsed > 0.0 {
            let mut tokens = self.tokens.lock();
            *tokens = (*tokens + elapsed * self.refill_rate).min(self.max_tokens as f64);
            *last = now;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_starts_full() {
        let rl = RateLimiter::new(100, 10.0);
        assert_eq!(rl.available_tokens(), 100);
    }

    #[test]
    fn test_try_acquire_success() {
        let rl = RateLimiter::new(100, 10.0);
        assert!(rl.try_acquire(50));
        assert_eq!(rl.available_tokens(), 50);
    }

    #[test]
    fn test_try_acquire_exact_capacity() {
        let rl = RateLimiter::new(100, 10.0);
        assert!(rl.try_acquire(100));
        // Might have a tiny refill by now, so check it's near zero
        assert!(rl.available_tokens() <= 1);
    }

    #[test]
    fn test_try_acquire_insufficient() {
        let rl = RateLimiter::new(10, 1.0);
        assert!(!rl.try_acquire(11));
        // Tokens should not have been consumed
        assert!(rl.available_tokens() >= 9); // allow tiny refill
    }

    #[test]
    fn test_multiple_acquisitions() {
        let rl = RateLimiter::new(100, 10.0);
        assert!(rl.try_acquire(30));
        assert!(rl.try_acquire(30));
        assert!(rl.try_acquire(30));
        // ~10 tokens left (+ small refill)
        assert!(!rl.try_acquire(20));
    }

    #[test]
    fn test_refill_over_time() {
        let rl = RateLimiter::new(100, 1000.0); // 1000 tokens/sec
        assert!(rl.try_acquire(100)); // drain all
        // Sleep briefly to allow refill
        std::thread::sleep(Duration::from_millis(50));
        // Should have refilled ~50 tokens (1000/sec * 0.05s)
        let available = rl.available_tokens();
        assert!(available >= 30, "expected >=30, got {}", available);
        assert!(available <= 70, "expected <=70, got {}", available);
    }

    #[test]
    fn test_burst_capacity_not_exceeded() {
        let rl = RateLimiter::new(50, 1000.0);
        // Even after sleeping, tokens should not exceed max
        std::thread::sleep(Duration::from_millis(100));
        assert!(rl.available_tokens() <= 50);
    }

    #[test]
    fn test_usage_percent_full_bucket() {
        let rl = RateLimiter::new(100, 10.0);
        let usage = rl.usage_percent();
        assert!(usage < 1.0, "expected ~0%, got {:.1}%", usage);
    }

    #[test]
    fn test_usage_percent_half_consumed() {
        let rl = RateLimiter::new(100, 0.0); // no refill
        rl.try_acquire(50);
        let usage = rl.usage_percent();
        assert!(
            (usage - 50.0).abs() < 2.0,
            "expected ~50%, got {:.1}%",
            usage
        );
    }

    #[test]
    fn test_usage_percent_fully_consumed() {
        let rl = RateLimiter::new(100, 0.0); // no refill
        rl.try_acquire(100);
        let usage = rl.usage_percent();
        assert!(
            (usage - 100.0).abs() < 1.0,
            "expected ~100%, got {:.1}%",
            usage
        );
    }

    #[test]
    fn test_update_from_header() {
        let rl = RateLimiter::new(1200, 20.0);
        rl.update_from_header(600);
        assert_eq!(rl.available_tokens(), 600);
    }

    #[test]
    fn test_update_from_header_exceeds_max() {
        let rl = RateLimiter::new(1200, 20.0);
        rl.update_from_header(1500); // used more than max
        assert_eq!(rl.available_tokens(), 0);
    }

    #[test]
    fn test_acquire_blocking() {
        let rl = RateLimiter::new(10, 1000.0); // fast refill
        rl.try_acquire(10); // drain
        // acquire should block briefly then succeed
        rl.acquire(5);
        // If we got here, acquire succeeded
    }

    #[test]
    fn test_binance_default() {
        let rl = RateLimiter::binance_default();
        assert_eq!(rl.max_tokens, 1200);
        assert_eq!(rl.available_tokens(), 1200);
    }

    #[test]
    fn test_bybit_default() {
        let rl = RateLimiter::bybit_default();
        assert_eq!(rl.max_tokens, 120);
        assert_eq!(rl.available_tokens(), 120);
    }

    #[test]
    fn test_zero_weight_acquire() {
        let rl = RateLimiter::new(100, 10.0);
        assert!(rl.try_acquire(0));
        assert_eq!(rl.available_tokens(), 100);
    }
}
