//! Nanosecond-precision timestamps.
//!
//! [`Timestamp`] wraps a `u64` representing nanoseconds since the Unix epoch.
//! The [`Timestamp::now`] method uses `CLOCK_MONOTONIC` for low-overhead,
//! NTP-drift-independent timing on the hot path.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Nanosecond-precision timestamp.
///
/// Internally stores nanoseconds since the Unix epoch as a `u64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Capture the current monotonic time as nanoseconds.
    ///
    /// Uses `clock_gettime(CLOCK_MONOTONIC)` on supported platforms for
    /// sub-microsecond overhead, independent of NTP adjustments.
    ///
    /// On platforms without `clock_gettime`, falls back to
    /// `std::time::SystemTime`.
    #[inline]
    pub fn now() -> Self {
        #[cfg(target_os = "linux")]
        {
            Self(linux_monotonic_nanos())
        }
        #[cfg(target_os = "macos")]
        {
            Self(macos_monotonic_nanos())
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            let dur = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock before UNIX epoch");
            Self(dur.as_nanos() as u64)
        }
    }

    /// Create a timestamp from milliseconds since the epoch.
    #[inline]
    pub const fn from_millis(ms: u64) -> Self {
        Self(ms * 1_000_000)
    }

    /// Convert to milliseconds since the epoch.
    #[inline]
    pub const fn to_millis(&self) -> u64 {
        self.0 / 1_000_000
    }

    /// Returns the raw nanosecond value.
    #[inline]
    pub const fn as_nanos(&self) -> u64 {
        self.0
    }

    /// Calculate the elapsed nanoseconds from `earlier` to `self`.
    ///
    /// Returns `0` if `self` is before `earlier` (monotonic timestamps
    /// should not go backwards, but this is defensive).
    #[inline]
    pub const fn elapsed_since(&self, earlier: &Timestamp) -> u64 {
        if self.0 >= earlier.0 {
            self.0 - earlier.0
        } else {
            0
        }
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let secs = self.0 / 1_000_000_000;
        let nanos = self.0 % 1_000_000_000;
        write!(f, "{}.{:09}", secs, nanos)
    }
}

#[cfg(target_os = "linux")]
fn linux_monotonic_nanos() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: passing a valid pointer to a stack-allocated timespec.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(target_os = "macos")]
fn macos_monotonic_nanos() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: passing a valid pointer to a stack-allocated timespec.
    unsafe {
        libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
    }
    ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_millis() {
        let ts = Timestamp::from_millis(1000);
        assert_eq!(ts.0, 1_000_000_000);
    }

    #[test]
    fn test_to_millis() {
        let ts = Timestamp(1_500_000_000);
        assert_eq!(ts.to_millis(), 1500);
    }

    #[test]
    fn test_as_nanos() {
        let ts = Timestamp(42);
        assert_eq!(ts.as_nanos(), 42);
    }

    #[test]
    fn test_elapsed_since() {
        let earlier = Timestamp(1_000_000_000);
        let later = Timestamp(2_500_000_000);
        assert_eq!(later.elapsed_since(&earlier), 1_500_000_000);
    }

    #[test]
    fn test_elapsed_since_backwards() {
        let earlier = Timestamp(2_000_000_000);
        let later = Timestamp(1_000_000_000);
        assert_eq!(later.elapsed_since(&earlier), 0);
    }

    #[test]
    fn test_now_is_nonzero() {
        let ts = Timestamp::now();
        assert!(ts.0 > 0);
    }

    #[test]
    fn test_now_monotonic() {
        let a = Timestamp::now();
        let b = Timestamp::now();
        assert!(b >= a);
    }

    #[test]
    fn test_display() {
        let ts = Timestamp(1_234_567_890_123_456_789);
        assert_eq!(format!("{}", ts), "1234567890.123456789");
    }

    #[test]
    fn test_ord() {
        assert!(Timestamp(100) < Timestamp(200));
        assert!(Timestamp(200) > Timestamp(100));
        assert_eq!(Timestamp(100), Timestamp(100));
    }

    #[test]
    fn test_from_millis_roundtrip() {
        let ms = 1706000000000u64; // ~2024-01-23
        let ts = Timestamp::from_millis(ms);
        assert_eq!(ts.to_millis(), ms);
    }
}
