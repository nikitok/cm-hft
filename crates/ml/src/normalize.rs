use crate::features::MlFeatures;

/// Pre-computed normalization statistics from training data.
///
/// Loaded from a JSON sidecar file alongside the model weights.
/// Ensures inference uses the exact same normalization as training.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NormStats {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

impl NormStats {
    /// Load normalization stats from a JSON file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let stats: NormStats = serde_json::from_str(&data)?;
        anyhow::ensure!(
            stats.mean.len() == MlFeatures::NUM_FEATURES,
            "norm stats mean has {} elements, expected {}",
            stats.mean.len(),
            MlFeatures::NUM_FEATURES
        );
        anyhow::ensure!(
            stats.std.len() == MlFeatures::NUM_FEATURES,
            "norm stats std has {} elements, expected {}",
            stats.std.len(),
            MlFeatures::NUM_FEATURES
        );
        Ok(stats)
    }

    /// Save normalization stats to a JSON file.
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Apply fixed normalization to a raw feature vector.
    pub fn normalize(
        &self,
        raw: &[f64; MlFeatures::NUM_FEATURES],
    ) -> [f64; MlFeatures::NUM_FEATURES] {
        let mut out = [0.0; MlFeatures::NUM_FEATURES];
        for i in 0..MlFeatures::NUM_FEATURES {
            if self.std[i] > 1e-12 {
                out[i] = ((raw[i] - self.mean[i]) / self.std[i]).clamp(-5.0, 5.0);
            }
        }
        out
    }
}

/// Online z-score normalizer using Welford's algorithm.
///
/// Fallback when no pre-computed stats are available.
/// Tracks per-feature running mean and variance. Returns raw values
/// until `min_samples` have been collected, then z-scores.
pub struct RunningNormalizer {
    count: u64,
    mean: [f64; MlFeatures::NUM_FEATURES],
    m2: [f64; MlFeatures::NUM_FEATURES],
    min_samples: u64,
}

impl RunningNormalizer {
    pub fn new(min_samples: u64) -> Self {
        Self {
            count: 0,
            mean: [0.0; MlFeatures::NUM_FEATURES],
            m2: [0.0; MlFeatures::NUM_FEATURES],
            min_samples,
        }
    }

    /// Feed a raw feature vector and return the normalized version.
    ///
    /// Before `min_samples` are collected, returns the raw values unchanged.
    /// After warm-up, returns `(x - mean) / std` for each feature, clamped to [-5, 5].
    pub fn normalize(
        &mut self,
        raw: &[f64; MlFeatures::NUM_FEATURES],
    ) -> [f64; MlFeatures::NUM_FEATURES] {
        self.count += 1;
        let n = self.count as f64;

        // Welford's online update.
        for (i, &r) in raw.iter().enumerate() {
            let delta = r - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = r - self.mean[i];
            self.m2[i] += delta * delta2;
        }

        if self.count < self.min_samples {
            return *raw;
        }

        let mut out = [0.0; MlFeatures::NUM_FEATURES];
        for i in 0..MlFeatures::NUM_FEATURES {
            let variance = self.m2[i] / n;
            let std = variance.sqrt();
            if std > 1e-12 {
                out[i] = ((raw[i] - self.mean[i]) / std).clamp(-5.0, 5.0);
            } else {
                out[i] = 0.0;
            }
        }
        out
    }

    /// Whether enough samples have been collected for meaningful normalization.
    pub fn is_warmed_up(&self) -> bool {
        self.count >= self.min_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_passthrough() {
        let mut norm = RunningNormalizer::new(3);
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert!(!norm.is_warmed_up());
        let out = norm.normalize(&raw);
        // Before min_samples, returns raw.
        assert_eq!(out, raw);
    }

    #[test]
    fn test_z_score_after_warmup() {
        let mut norm = RunningNormalizer::new(5);
        // Feed identical values → std=0 → output=0.
        for _ in 0..10 {
            norm.normalize(&[1.0; 7]);
        }
        assert!(norm.is_warmed_up());
        let out = norm.normalize(&[1.0; 7]);
        for v in &out {
            assert!(v.abs() < 1e-6, "constant input should give ~0 z-score");
        }
    }

    #[test]
    fn test_z_score_scaling() {
        let mut norm = RunningNormalizer::new(10);
        // Feed a range of values.
        for i in 0..100 {
            let v = i as f64;
            norm.normalize(&[v; 7]);
        }
        // A value near the mean should have z-score near 0.
        let mean_val = 49.5; // approx mean of 0..99
        let out = norm.normalize(&[mean_val; 7]);
        for v in &out {
            assert!(
                v.abs() < 0.5,
                "mean-ish value should have small z-score, got {v}"
            );
        }
    }

    #[test]
    fn test_clamp_extreme() {
        let mut norm = RunningNormalizer::new(5);
        for i in 0..100 {
            norm.normalize(&[i as f64; 7]);
        }
        // A very extreme value should be clamped to [-5, 5].
        let out = norm.normalize(&[99999.0; 7]);
        for v in &out {
            assert!(*v <= 5.0 && *v >= -5.0, "should be clamped, got {v}");
        }
    }

    #[test]
    fn test_norm_stats_fixed() {
        let stats = NormStats {
            mean: vec![0.0, 0.0, 0.3, 2.0, 3.0, 0.0, 0.0],
            std: vec![0.5, 0.4, 0.1, 1.0, 1.0, 5.0, 0.5],
        };
        let raw = [0.5, 0.4, 0.3, 2.0, 3.0, 0.0, 0.0];
        let out = stats.normalize(&raw);
        // (0.5-0.0)/0.5 = 1.0
        assert!((out[0] - 1.0).abs() < 1e-10);
        // (0.4-0.0)/0.4 = 1.0
        assert!((out[1] - 1.0).abs() < 1e-10);
        // (0.3-0.3)/0.1 = 0.0
        assert!(out[2].abs() < 1e-10);
    }

    #[test]
    fn test_norm_stats_save_load() {
        let stats = NormStats {
            mean: vec![1.0; 7],
            std: vec![2.0; 7],
        };
        let tmp = std::env::temp_dir().join("cm_ml_test_norm_stats.json");
        stats.save(&tmp).unwrap();
        let loaded = NormStats::load(&tmp).unwrap();
        assert_eq!(loaded.mean, stats.mean);
        assert_eq!(loaded.std, stats.std);
        std::fs::remove_file(&tmp).ok();
    }
}
