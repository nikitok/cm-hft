//! Strategy-facing inference wrapper.
//!
//! [`MlSignal`] combines the neural network with normalization.
//! It returns a directional signal in [-0.5, 0.5] that the strategy
//! can use to shift fair value.
//!
//! At load time, candle weights are extracted into [`RawWeights`] for
//! zero-allocation inference on the hot path.
//!
//! Normalization priority:
//! 1. Fixed stats from training (JSON sidecar) — preferred, consistent with training
//! 2. Online Welford normalization — fallback when no stats file

use std::path::Path;

use anyhow::Result;
use candle_core::Device;

use crate::features::MlFeatures;
use crate::model::{MidPredictor, RawWeights};
use crate::normalize::{NormStats, RunningNormalizer};

/// Normalization strategy: fixed (from training) or online (Welford).
enum Normalizer {
    Fixed(NormStats),
    Online(RunningNormalizer),
}

/// Inference wrapper for the mid-price predictor.
pub struct MlSignal {
    weights: RawWeights,
    normalizer: Normalizer,
}

impl MlSignal {
    /// Try to load a model from the given path.
    ///
    /// Also attempts to load normalization stats from a sidecar JSON file
    /// (same directory, same stem, `.norm.json` extension).
    ///
    /// Returns `None` if the weights file doesn't exist (graceful degradation).
    /// Returns `Err` if the file exists but is malformed.
    pub fn try_load(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            tracing::info!(?path, "ML weights not found, running without ML signal");
            return Ok(None);
        }

        let device = Device::Cpu;
        tracing::info!(?path, "loading ML model");

        let model = MidPredictor::load(path, &device)?;
        let weights = model.extract_weights()?;
        // candle model dropped here — only RawWeights kept

        // Try loading normalization stats from sidecar file.
        let norm_path = path.with_extension("norm.json");
        let normalizer = if norm_path.exists() {
            match NormStats::load(&norm_path) {
                Ok(stats) => {
                    tracing::info!(?norm_path, "loaded normalization stats from training");
                    Normalizer::Fixed(stats)
                }
                Err(e) => {
                    tracing::warn!("failed to load norm stats: {e}, using online normalization");
                    Normalizer::Online(RunningNormalizer::new(100))
                }
            }
        } else {
            tracing::info!("no norm stats found, using online normalization");
            Normalizer::Online(RunningNormalizer::new(100))
        };

        Ok(Some(Self {
            weights,
            normalizer,
        }))
    }

    /// Create from an already-loaded model (for testing).
    pub fn from_model(model: MidPredictor, _device: Device) -> Self {
        Self {
            weights: model.extract_weights().expect("weight extraction"),
            normalizer: Normalizer::Online(RunningNormalizer::new(100)),
        }
    }

    /// Create from model + pre-computed stats (for testing / explicit construction).
    pub fn from_model_with_stats(model: MidPredictor, stats: NormStats, _device: Device) -> Self {
        Self {
            weights: model.extract_weights().expect("weight extraction"),
            normalizer: Normalizer::Fixed(stats),
        }
    }

    /// Predict directional signal from features.
    ///
    /// Returns a value in [-0.5, 0.5]:
    /// - Positive: model expects mid to go up
    /// - Negative: model expects mid to go down
    /// - Zero: neutral (or normalizer still warming up)
    pub fn predict(&mut self, features: &MlFeatures) -> f64 {
        let raw = features.to_array();

        let normalized = match &mut self.normalizer {
            Normalizer::Fixed(stats) => stats.normalize(&raw),
            Normalizer::Online(norm) => {
                let out = norm.normalize(&raw);
                if !norm.is_warmed_up() {
                    return 0.0;
                }
                out
            }
        };

        let mut input = [0.0_f32; MlFeatures::NUM_FEATURES];
        for i in 0..MlFeatures::NUM_FEATURES {
            input[i] = normalized[i] as f32;
        }

        self.weights.predict(&input) as f64 - 0.5 // center around 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    fn make_signal() -> MlSignal {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();
        MlSignal::from_model(model, device)
    }

    #[test]
    fn test_returns_zero_before_warmup() {
        let mut signal = make_signal();
        let features = MlFeatures {
            book_imbalance: 0.5,
            trade_flow_imbalance: 0.3,
            vpin: 0.2,
            volatility_bps: 3.0,
            spread_bps: 2.0,
            recent_return_bps: 1.0,
            normalized_position: 0.0,
        };
        // Before 100 samples, should return 0.
        for _ in 0..50 {
            let val = signal.predict(&features);
            assert!(val.abs() < 1e-12, "should be 0 during warmup, got {val}");
        }
    }

    #[test]
    fn test_returns_signal_after_warmup() {
        let mut signal = make_signal();
        let features = MlFeatures {
            book_imbalance: 0.5,
            trade_flow_imbalance: 0.3,
            vpin: 0.2,
            volatility_bps: 3.0,
            spread_bps: 2.0,
            recent_return_bps: 1.0,
            normalized_position: 0.0,
        };
        // Warm up.
        for _ in 0..150 {
            signal.predict(&features);
        }
        // After warmup, should return a value in [-0.5, 0.5].
        let val = signal.predict(&features);
        assert!(
            (-0.5..=0.5).contains(&val),
            "signal should be in [-0.5, 0.5], got {val}"
        );
    }

    #[test]
    fn test_fixed_stats_no_warmup() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();
        let stats = NormStats {
            mean: vec![0.0; 7],
            std: vec![1.0; 7],
        };
        let mut signal = MlSignal::from_model_with_stats(model, stats, device);
        let features = MlFeatures {
            book_imbalance: 0.5,
            trade_flow_imbalance: 0.3,
            vpin: 0.2,
            volatility_bps: 3.0,
            spread_bps: 2.0,
            recent_return_bps: 1.0,
            normalized_position: 0.0,
        };
        // With fixed stats, should produce signal immediately (no warmup needed).
        let val = signal.predict(&features);
        assert!(
            (-0.5..=0.5).contains(&val),
            "signal should be in [-0.5, 0.5], got {val}"
        );
    }

    #[test]
    fn test_try_load_missing_file() {
        let result = MlSignal::try_load(Path::new("/nonexistent/model.safetensors")).unwrap();
        assert!(result.is_none());
    }
}
