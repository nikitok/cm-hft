//! Fixed-point decimal quantity type.
//!
//! [`Quantity`] uses the same fixed-point representation as [`super::Price`]:
//! `mantissa * 10^(-scale)`. The mantissa is signed (`i64`) to support short
//! positions expressed as negative quantities.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};

use serde::{Deserialize, Serialize};

/// Fixed-point decimal representing a quantity (e.g., BTC amount).
///
/// `value = mantissa * 10^(-scale)`
///
/// The mantissa is signed to represent short positions as negative values.
///
/// # Examples
///
/// ```
/// use cm_core::types::Quantity;
///
/// let qty = Quantity::new(100000, 8); // 0.001 BTC
/// assert!((qty.to_f64() - 0.001).abs() < 1e-12);
/// ```
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Quantity {
    mantissa: i64,
    scale: u8,
}

impl Quantity {
    /// Create a new quantity from mantissa and scale.
    #[inline]
    pub const fn new(mantissa: i64, scale: u8) -> Self {
        Self { mantissa, scale }
    }

    /// Create a zero quantity with the given scale.
    #[inline]
    pub const fn zero(scale: u8) -> Self {
        Self { mantissa: 0, scale }
    }

    /// Returns `true` if this quantity is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.mantissa == 0
    }

    /// Returns the absolute value of this quantity.
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

    /// Convert to `f64`. **Not for hot-path use.**
    #[inline]
    pub fn to_f64(&self) -> f64 {
        self.mantissa as f64 / 10f64.powi(self.scale as i32)
    }

    /// Returns `true` if this quantity is negative (short position).
    #[inline]
    pub const fn is_negative(&self) -> bool {
        self.mantissa < 0
    }

    /// Normalize two quantities to the same (higher) scale.
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

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({}, scale={})", self.to_f64(), self.scale)
    }
}

impl fmt::Display for Quantity {
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

impl PartialEq for Quantity {
    fn eq(&self, other: &Self) -> bool {
        match Self::normalize(*self, *other) {
            Some((a, b, _)) => a == b,
            None => false,
        }
    }
}

impl Eq for Quantity {}

impl PartialOrd for Quantity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Quantity {
    fn cmp(&self, other: &Self) -> Ordering {
        match Self::normalize(*self, *other) {
            Some((a, b, _)) => a.cmp(&b),
            None => self
                .to_f64()
                .partial_cmp(&other.to_f64())
                .unwrap_or(Ordering::Equal),
        }
    }
}

impl Hash for Quantity {
    fn hash<H: Hasher>(&self, state: &mut H) {
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

impl Add for Quantity {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b, scale) =
            Self::normalize(self, rhs).expect("Quantity::add overflow during scale normalization");
        Self {
            mantissa: a.checked_add(b).expect("Quantity::add overflow"),
            scale,
        }
    }
}

impl Sub for Quantity {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let (a, b, scale) =
            Self::normalize(self, rhs).expect("Quantity::sub overflow during scale normalization");
        Self {
            mantissa: a.checked_sub(b).expect("Quantity::sub overflow"),
            scale,
        }
    }
}

impl Mul<i64> for Quantity {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        Self {
            mantissa: self
                .mantissa
                .checked_mul(rhs)
                .expect("Quantity::mul overflow"),
            scale: self.scale,
        }
    }
}

impl Div<i64> for Quantity {
    type Output = Self;

    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        assert!(rhs != 0, "Quantity::div division by zero");
        Self {
            mantissa: self.mantissa / rhs,
            scale: self.scale,
        }
    }
}

impl Neg for Quantity {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            mantissa: -self.mantissa,
            scale: self.scale,
        }
    }
}

impl From<f64> for Quantity {
    /// Convert an `f64` to a `Quantity` with scale 8. **Not for hot-path use.**
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
        let q = Quantity::new(100000, 8);
        assert_eq!(q.mantissa(), 100000);
        assert_eq!(q.scale(), 8);
    }

    #[test]
    fn test_zero() {
        let z = Quantity::zero(8);
        assert!(z.is_zero());
    }

    #[test]
    fn test_is_negative() {
        assert!(Quantity::new(-1, 8).is_negative());
        assert!(!Quantity::new(1, 8).is_negative());
        assert!(!Quantity::new(0, 8).is_negative());
    }

    #[test]
    fn test_abs() {
        assert_eq!(Quantity::new(-500, 4).abs(), Quantity::new(500, 4));
    }

    #[test]
    fn test_to_f64() {
        let q = Quantity::new(100000, 8); // 0.001
        assert!((q.to_f64() - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_display() {
        let q = Quantity::new(100000, 8);
        assert_eq!(format!("{}", q), "0.00100000");
    }

    #[test]
    fn test_add_same_scale() {
        let a = Quantity::new(100, 4);
        let b = Quantity::new(200, 4);
        assert_eq!((a + b).mantissa(), 300);
    }

    #[test]
    fn test_add_different_scale() {
        let a = Quantity::new(10, 1);
        let b = Quantity::new(250, 3);
        let c = a + b;
        assert_eq!(c.mantissa(), 1250);
        assert_eq!(c.scale(), 3);
    }

    #[test]
    fn test_sub() {
        let a = Quantity::new(500, 4);
        let b = Quantity::new(200, 4);
        assert_eq!((a - b).mantissa(), 300);
    }

    #[test]
    fn test_sub_negative_result() {
        let a = Quantity::new(100, 4);
        let b = Quantity::new(300, 4);
        let r = a - b;
        assert!(r.is_negative());
        assert_eq!(r.mantissa(), -200);
    }

    #[test]
    fn test_mul_scalar() {
        let q = Quantity::new(100, 4);
        assert_eq!((q * 5).mantissa(), 500);
    }

    #[test]
    fn test_div_scalar() {
        let q = Quantity::new(300, 4);
        assert_eq!((q / 3).mantissa(), 100);
    }

    #[test]
    #[should_panic(expected = "division by zero")]
    fn test_div_by_zero() {
        let q = Quantity::new(100, 4);
        let _ = q / 0;
    }

    #[test]
    fn test_neg() {
        let q = Quantity::new(100, 4);
        assert_eq!((-q).mantissa(), -100);
    }

    #[test]
    fn test_eq_different_scale() {
        assert_eq!(Quantity::new(10, 1), Quantity::new(100, 2));
    }

    #[test]
    fn test_ord() {
        assert!(Quantity::new(200, 4) > Quantity::new(100, 4));
    }

    #[test]
    fn test_hash_consistency() {
        let mut set = HashSet::new();
        set.insert(Quantity::new(10, 1));
        assert!(set.contains(&Quantity::new(100, 2)));
    }

    #[test]
    fn test_from_f64() {
        let q = Quantity::from(0.001);
        assert!((q.to_f64() - 0.001).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "overflow")]
    fn test_add_overflow() {
        let a = Quantity::new(i64::MAX, 4);
        let b = Quantity::new(1, 4);
        let _ = a + b;
    }
}
