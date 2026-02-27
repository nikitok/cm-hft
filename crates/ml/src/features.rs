/// Feature vector for mid-price prediction.
///
/// 7 inputs capturing the state visible to the adaptive market maker.
/// All features are raw (un-normalized); the [`RunningNormalizer`](crate::normalize::RunningNormalizer)
/// handles online z-score normalization before inference.
#[derive(Debug, Clone, Copy)]
pub struct MlFeatures {
    /// Book imbalance from top levels, in [-1, 1].
    pub book_imbalance: f64,
    /// Trade flow imbalance (buy vs sell pressure), in [-1, 1].
    pub trade_flow_imbalance: f64,
    /// Volume-synchronized probability of informed trading, in [0, 1].
    pub vpin: f64,
    /// Realized volatility in basis points (positive).
    pub volatility_bps: f64,
    /// Current spread in basis points (positive).
    pub spread_bps: f64,
    /// Recent mid-price return in basis points (signed).
    pub recent_return_bps: f64,
    /// Net position normalized by max_position, in [-1, 1].
    pub normalized_position: f64,
}

impl MlFeatures {
    pub const NUM_FEATURES: usize = 7;

    /// Convert to a fixed-size array for normalization and tensor creation.
    pub fn to_array(&self) -> [f64; Self::NUM_FEATURES] {
        [
            self.book_imbalance,
            self.trade_flow_imbalance,
            self.vpin,
            self.volatility_bps,
            self.spread_bps,
            self.recent_return_bps,
            self.normalized_position,
        ]
    }
}
