//! Fixed-point decimal price type for the hot path.
//!
//! [`Price`] avoids floating-point arithmetic entirely by storing values as
//! `mantissa * 10^(-scale)`. For BTC prices with 2 decimal places:
//! `Price { mantissa: 5000050, scale: 2 }` represents `50000.50`.
//!
//! All arithmetic operations handle cross-scale operands by normalizing to the
//! higher (more precise) scale before operating.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

/// Fixed-point decimal representing a price.
///
/// `value = mantissa * 10^(-scale)`
///
/// # Examples
///
/// ```
/// use cm_core::types::Price;
///
/// let price = Price::new(5000050, 2); // 50000.50
/// assert_eq!(price.to_f64(), 50000.50);
/// ```
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Price {
    mantissa: i64,
    scale: u8,
}

impl Price {
    /// Create a new price from mantissa and scale.
    ///
    /// `value = mantissa * 10^(-scale)`
    #[inline]
    pub const fn new(mantissa: i64, scale: u8) -> Self {
        Self { mantissa, scale }
    }

    /// Create a zero price with the given scale.
    #[inline]
    pub const fn zero(scale: u8) -> Self {
        Self { mantissa: 0, scale }
    }

    /// Returns `true` if this price is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Returns the absolute value of this price.
    #[inline]
    pub const fn abs(&self) -> Self {
        Self {
            mantissa: self.mantissa.abs(),
            scale: self.scale,
        }
    }

    /// Returns the raw mantissa.
    #[inline]
    pub const fn mantissa(&self) -> i64 {
        self.mantissa
    }

    /// Returns the scale (number of decimal places).
    #[inline]
    pub const fn scale(&self) -> u8 {
        self.scale
    }

    /// Convert to `f64`. **Not for hot-path use** -- intended for logging,
    /// display, and serialization only.
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.mantissa as f64 / 10f64.powi(self.scale as i32)
    }

    /// Normalize two prices to the same (higher) scale, returning their
    /// aligned mantissas and the common scale.
    ///
    /// Returns `None` if the scale conversion would overflow `i64`.
    #[inline]
    fn normalize(a: Self, b: Self) -> Option<(i64, i64, u8)> {
        if a.scale == b.scale {
            return Some((a.mantissa, b.mantissa, a.scale));
        }

        let (lo, hi, lo_mantissa, hi_mantissa) = if a.scale < b.scale {
            (a.scale, b.scale, a.mantissa, b.mantissa)
        } else {
            (b.scale, a.scale, b.mantissa, a.mantissa)
        };

        let diff = (hi - lo) as u32;
        let factor = 10i64.checked_pow(diff)?;
        let scaled = lo_mantissa.checked_mul(factor)?;

        if a.scale < b.scale {
            Some((scaled, hi_mantissa, hi))
        } else {
            Some((hi_mantissa, scaled, hi))
        }
    }
}

impl fmt::Debug for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Price({}, scale={})", self.to_f64(), self.scale)
    }
}

impl fmt::Display for Price {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scale == 0 {
            write!(f, "{}", self.mantissa)
        } else {
            let divisor = 10i64.pow(self.scale as u32);
            let whole = self.mantissa / divisor;
            let frac = (self.mantissa % divisor).abs();
            write!(
                f,
                "{}.{:0>width$}",
                whole,
                frac,
                width = self.scale as usize
            )
        }
    }
}

impl PartialEq for Price {
    fn eq(&self, other: &Self) -> bool {
        match Self::normalize(*self, *other) {
            Some((a, b, _)) => a == b,
            None => false,
        }
    }
}

impl Eq for Price {}

impl PartialOrd for Price {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Price {
    fn cmp(&self, other: &Self) -> Ordering {
        match Self::normalize(*self, *other) {
            Some((a, b, _)) => a.cmp(&b),
            // If normalization overflows, fall back to f64 comparison.
            None => self
                .to_f64()
                .partial_cmp(&other.to_f64())
                .unwrap_or(Ordering::Equal),
        }
    }
}

impl Hash for Price {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Normalize to canonical form: strip trailing zeros.
        let mut m = self.mantissa;
        let mut s = self.scale;
        while s > 0 && m % 10 == 0 {
            m /= 10;
            s -= 1;
        }
        m.hash(state);
        s.hash(state);
    }
}

impl Add for Price {
    type Output = Self;

    /// Add two prices. Panics on overflow in debug builds.
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b, scale) =
            Self::normalize(self, rhs).expect("Price::add overflow during scale normalization");
        Self {
            mantissa: a.checked_add(b).expect("Price::add overflow"),
            scale,
        }
    }
}

impl Sub for Price {
    type Output = Self;

    /// Subtract two prices. Panics on overflow in debug builds.
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b, scale) =
            Self::normalize(self, rhs).expect("Price::sub overflow during scale normalization");
        Self {
            mantissa: a.checked_sub(b).expect("Price::sub overflow"),
            scale,
        }
    }
}

impl Mul<i64> for Price {
    type Output = Self;

    /// Multiply a price by an integer scalar.
    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            mantissa: self.mantissa.checked_mul(rhs).expect("Price::mul overflow"),
            scale: self.scale,
        }
    }
}

impl Div<i64> for Price {
    type Output = Self;

    /// Divide a price by an integer scalar (truncating division).
    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        assert!(rhs != 0, "Price::div division by zero");
        Self {
            mantissa: self.mantissa / rhs,
            scale: self.scale,
        }
    }
}

impl Neg for Price {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            mantissa: -self.mantissa,
            scale: self.scale,
        }
    }
}

impl From<f64> for Price {
    /// Convert an `f64` to a `Price` with scale 8 (8 decimal places).
    ///
    /// **Not for hot-path use.** Floating-point conversion is inherently
    /// imprecise; this is provided for convenience in tests and configuration.
    fn from(value: f64) -> Self {
        const DEFAULT_SCALE: u8 = 8;
        let factor = 10f64.powi(DEFAULT_SCALE as i32);
        Self {
            mantissa: (value * factor).round() as i64,
            scale: DEFAULT_SCALE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_new_and_accessors() {
        let p = Price::new(5000050, 2);
        assert_eq!(p.mantissa(), 5000050);
        assert_eq!(p.scale(), 2);
    }

    #[test]
    fn test_zero() {
        let z = Price::zero(4);
        assert!(z.is_zero());
        assert_eq!(z.mantissa(), 0);
        assert_eq!(z.scale(), 4);
    }

    #[test]
    fn test_is_zero_false() {
        let p = Price::new(1, 2);
        assert!(!p.is_zero());
    }

    #[test]
    fn test_abs_positive() {
        let p = Price::new(100, 2);
        assert_eq!(p.abs().mantissa(), 100);
    }

    #[test]
    fn test_abs_negative() {
        let p = Price::new(-100, 2);
        assert_eq!(p.abs().mantissa(), 100);
    }

    #[test]
    fn test_to_f64() {
        let p = Price::new(5000050, 2);
        assert!((p.to_f64() - 50000.50).abs() < 1e-10);
    }

    #[test]
    fn test_to_f64_zero_scale() {
        let p = Price::new(42, 0);
        assert!((p.to_f64() - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_display_basic() {
        let p = Price::new(5000050, 2);
        assert_eq!(format!("{}", p), "50000.50");
    }

    #[test]
    fn test_display_zero_scale() {
        let p = Price::new(42, 0);
        assert_eq!(format!("{}", p), "42");
    }

    #[test]
    fn test_display_negative() {
        let p = Price::new(-5000050, 2);
        assert_eq!(format!("{}", p), "-50000.50");
    }

    #[test]
    fn test_display_small_fraction() {
        let p = Price::new(1, 4);
        assert_eq!(format!("{}", p), "0.0001");
    }

    #[test]
    fn test_add_same_scale() {
        let a = Price::new(100, 2);
        let b = Price::new(200, 2);
        let c = a + b;
        assert_eq!(c.mantissa(), 300);
        assert_eq!(c.scale(), 2);
    }

    #[test]
    fn test_add_different_scale() {
        let a = Price::new(10, 1); // 1.0
        let b = Price::new(250, 2); // 2.50
        let c = a + b;
        assert_eq!(c.mantissa(), 350); // 3.50
        assert_eq!(c.scale(), 2);
    }

    #[test]
    fn test_sub_same_scale() {
        let a = Price::new(300, 2);
        let b = Price::new(100, 2);
        let c = a - b;
        assert_eq!(c.mantissa(), 200);
    }

    #[test]
    fn test_sub_different_scale() {
        let a = Price::new(350, 2); // 3.50
        let b = Price::new(10, 1); // 1.0
        let c = a - b;
        assert_eq!(c.mantissa(), 250); // 2.50
        assert_eq!(c.scale(), 2);
    }

    #[test]
    fn test_sub_result_negative() {
        let a = Price::new(100, 2);
        let b = Price::new(300, 2);
        let c = a - b;
        assert_eq!(c.mantissa(), -200);
    }

    #[test]
    fn test_mul_scalar() {
        let p = Price::new(100, 2);
        let r = p * 3;
        assert_eq!(r.mantissa(), 300);
        assert_eq!(r.scale(), 2);
    }

    #[test]
    fn test_mul_scalar_negative() {
        let p = Price::new(100, 2);
        let r = p * -2;
        assert_eq!(r.mantissa(), -200);
    }

    #[test]
    fn test_div_scalar() {
        let p = Price::new(300, 2);
        let r = p / 3;
        assert_eq!(r.mantissa(), 100);
    }

    #[test]
    fn test_div_truncation() {
        let p = Price::new(100, 2);
        let r = p / 3;
        assert_eq!(r.mantissa(), 33); // truncates
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_div_by_zero() {
        let p = Price::new(100, 2);
        let _ = p / 0;
    }

    #[test]
    fn test_neg() {
        let p = Price::new(100, 2);
        let n = -p;
        assert_eq!(n.mantissa(), -100);
    }

    #[test]
    fn test_neg_negative() {
        let p = Price::new(-100, 2);
        let n = -p;
        assert_eq!(n.mantissa(), 100);
    }

    #[test]
    fn test_eq_same_scale() {
        assert_eq!(Price::new(100, 2), Price::new(100, 2));
        assert_ne!(Price::new(100, 2), Price::new(200, 2));
    }

    #[test]
    fn test_eq_different_scale() {
        // 1.0 (scale 1) == 1.00 (scale 2)
        assert_eq!(Price::new(10, 1), Price::new(100, 2));
    }

    #[test]
    fn test_ord_same_scale() {
        assert!(Price::new(200, 2) > Price::new(100, 2));
        assert!(Price::new(100, 2) < Price::new(200, 2));
    }

    #[test]
    fn test_ord_different_scale() {
        // 2.50 > 2.0
        assert!(Price::new(250, 2) > Price::new(20, 1));
    }

    #[test]
    fn test_ord_negative() {
        assert!(Price::new(-100, 2) < Price::new(100, 2));
        assert!(Price::new(-100, 2) < Price::new(0, 2));
    }

    #[test]
    fn test_hash_consistency() {
        // Equal values must hash equally even with different scales.
        let mut set = HashSet::new();
        set.insert(Price::new(10, 1));
        assert!(set.contains(&Price::new(100, 2)));
    }

    #[test]
    fn test_from_f64() {
        let p = Price::from(50000.50);
        assert!((p.to_f64() - 50000.50).abs() < 1e-6);
    }

    #[test]
    fn test_from_f64_negative() {
        let p = Price::from(-123.456);
        assert!((p.to_f64() - (-123.456)).abs() < 1e-6);
    }

    #[test]
    fn test_from_f64_zero() {
        let p = Price::from(0.0);
        assert!(p.is_zero());
    }

    #[test]
    fn test_cross_scale_arithmetic_chain() {
        // (1.0 + 0.50) - 0.250 = 1.250
        let a = Price::new(10, 1); // 1.0
        let b = Price::new(50, 2); // 0.50
        let c = Price::new(250, 3); // 0.250
        let result = (a + b) - c;
        assert_eq!(result, Price::new(1250, 3));
    }

    #[test]
    fn test_zero_different_scales_equal() {
        assert_eq!(Price::zero(0), Price::zero(4));
    }

    #[test]
    fn test_large_mantissa() {
        let p = Price::new(i64::MAX / 2, 8);
        let q = Price::new(1, 8);
        let r = p + q;
        assert_eq!(r.mantissa(), i64::MAX / 2 + 1);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn test_add_overflow() {
        let a = Price::new(i64::MAX, 2);
        let b = Price::new(1, 2);
        let _ = a + b;
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn test_mul_overflow() {
        let p = Price::new(i64::MAX, 2);
        let _ = p * 2;
    }

    #[test]
    fn test_btc_price_example() {
        // BTC at $50,000.50 with 2 decimal precision
        let price = Price::new(5000050, 2);
        assert_eq!(format!("{}", price), "50000.50");
        assert!((price.to_f64() - 50000.50).abs() < 1e-10);
    }

    #[test]
    fn test_satoshi_precision() {
        // 0.00000001 BTC (1 satoshi) with 8 decimal places
        let sat = Price::new(1, 8);
        assert_eq!(format!("{}", sat), "0.00000001");
    }

    #[test]
    fn test_add_zero_identity() {
        let p = Price::new(12345, 3);
        let z = Price::zero(3);
        assert_eq!(p + z, p);
        assert_eq!(z + p, p);
    }

    #[test]
    fn test_sub_self_is_zero() {
        let p = Price::new(12345, 3);
        let r = p - p;
        assert!(r.is_zero());
    }

    #[test]
    fn test_display_large_scale() {
        let p = Price::new(123456789, 8);
        assert_eq!(format!("{}", p), "1.23456789");
    }
}
