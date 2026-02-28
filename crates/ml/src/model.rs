//! Candle neural network for mid-price direction prediction.
//!
//! Architecture: 7 → 64 → 32 → 1 feedforward with ReLU activations
//! and sigmoid output. ~2,625 parameters total.
//!
//! [`RawWeights`] provides a zero-allocation inference path using flat arrays.
//! Use [`MidPredictor::extract_weights`] to convert once, then call
//! [`RawWeights::predict`] on the hot path.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::features::MlFeatures;

/// Network dimensions (shared by [`MidPredictor`] and [`RawWeights`]).
pub const IN: usize = MlFeatures::NUM_FEATURES; // 7
pub const H1: usize = 64;
pub const H2: usize = 32;

// ── RawWeights ──────────────────────────────────────────────────────────

/// Extracted weights for zero-allocation forward pass (~10.5 KB, fits in L1).
pub struct RawWeights {
    fc1_weight: [f32; H1 * IN], // 448
    fc1_bias: [f32; H1],        // 64
    fc2_weight: [f32; H2 * H1], // 2048
    fc2_bias: [f32; H2],        // 32
    fc3_weight: [f32; H2],      // 32 (output layer is H2 → 1)
    fc3_bias: f32,              // 1
}

impl RawWeights {
    /// Pure stack-allocated forward pass: matmul + bias + relu + sigmoid.
    ///
    /// Zero heap allocations, no candle dependency. ~2,600 FMA ops.
    #[inline]
    pub fn predict(&self, input: &[f32; IN]) -> f32 {
        // Layer 1: IN → H1, ReLU
        let mut h1 = [0.0_f32; H1];
        for (j, h1_j) in h1.iter_mut().enumerate() {
            let mut sum = self.fc1_bias[j];
            let row = j * IN;
            for (i, &inp) in input.iter().enumerate() {
                sum += self.fc1_weight[row + i] * inp;
            }
            *h1_j = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Layer 2: H1 → H2, ReLU
        let mut h2 = [0.0_f32; H2];
        for (j, h2_j) in h2.iter_mut().enumerate() {
            let mut sum = self.fc2_bias[j];
            let row = j * H1;
            for (i, &h) in h1.iter().enumerate() {
                sum += self.fc2_weight[row + i] * h;
            }
            *h2_j = if sum > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // Layer 3: H2 → 1, sigmoid
        let mut logit = self.fc3_bias;
        for (&w, &h) in self.fc3_weight.iter().zip(h2.iter()) {
            logit += w * h;
        }

        // Sigmoid: 1 / (1 + exp(-x))
        1.0 / (1.0 + (-logit).exp())
    }
}

// ── MidPredictor ────────────────────────────────────────────────────────

/// Feedforward network predicting P(mid goes up) ∈ [0, 1].
pub struct MidPredictor {
    fc1: Linear, // 7 → 64
    fc2: Linear, // 64 → 32
    fc3: Linear, // 32 → 1
}

impl MidPredictor {
    // Re-export module-level constants for backward compat with internal usage.
    const IN: usize = IN;
    const H1: usize = H1;
    const H2: usize = H2;

    /// Create a new model with trainable weights (for training).
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = linear(Self::IN, Self::H1, vb.pp("fc1"))?;
        let fc2 = linear(Self::H1, Self::H2, vb.pp("fc2"))?;
        let fc3 = linear(Self::H2, 1, vb.pp("fc3"))?;
        Ok(Self { fc1, fc2, fc3 })
    }

    /// Load a trained model from a safetensors file.
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)? };
        Self::new(vb)
    }

    /// Forward pass returning raw logits (before sigmoid).
    /// Use this for training with `binary_cross_entropy_with_logit`.
    pub fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(&x)?.relu()?;
        let x = self.fc3.forward(&x)?;
        Ok(x)
    }

    /// Forward pass: features → P(mid goes up) ∈ [0, 1].
    ///
    /// Applies sigmoid to logits. For training, use `forward_logits` instead.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(x)?;
        // Sigmoid: 1 / (1 + exp(-x))
        // Using basic ops for Metal compatibility.
        let ones = logits.ones_like()?;
        let sigmoid = ones.broadcast_div(&(logits.neg()?.exp()? + 1.0)?)?;
        Ok(sigmoid)
    }

    /// Predict P(up) for a single feature vector.
    pub fn predict_one(
        &self,
        features: &[f32; MlFeatures::NUM_FEATURES],
        device: &Device,
    ) -> Result<f32> {
        let input = Tensor::from_slice(features.as_slice(), (1, Self::IN), device)?;
        let output = self.forward(&input)?;
        let val = output.flatten_all()?.to_vec1::<f32>()?;
        Ok(val[0])
    }

    /// Extract weights into flat arrays for zero-allocation inference.
    ///
    /// Call once at startup, then use [`RawWeights::predict`] on the hot path.
    pub fn extract_weights(&self) -> Result<RawWeights> {
        // fc1: weight shape [H1, IN], bias shape [H1]
        let w1_2d = self.fc1.weight().to_vec2::<f32>()?;
        let b1 = self.fc1.bias().expect("fc1 bias").to_vec1::<f32>()?;

        let mut fc1_weight = [0.0_f32; H1 * IN];
        for (j, row) in w1_2d.iter().enumerate() {
            fc1_weight[j * IN..(j + 1) * IN].copy_from_slice(row);
        }
        let mut fc1_bias = [0.0_f32; H1];
        fc1_bias.copy_from_slice(&b1);

        // fc2: weight shape [H2, H1], bias shape [H2]
        let w2_2d = self.fc2.weight().to_vec2::<f32>()?;
        let b2 = self.fc2.bias().expect("fc2 bias").to_vec1::<f32>()?;

        let mut fc2_weight = [0.0_f32; H2 * H1];
        for (j, row) in w2_2d.iter().enumerate() {
            fc2_weight[j * H1..(j + 1) * H1].copy_from_slice(row);
        }
        let mut fc2_bias = [0.0_f32; H2];
        fc2_bias.copy_from_slice(&b2);

        // fc3: weight shape [1, H2], bias shape [1]
        let w3_2d = self.fc3.weight().to_vec2::<f32>()?;
        let b3 = self.fc3.bias().expect("fc3 bias").to_vec1::<f32>()?;

        let mut fc3_weight = [0.0_f32; H2];
        fc3_weight.copy_from_slice(&w3_2d[0]);
        let fc3_bias = b3[0];

        Ok(RawWeights {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            fc3_weight,
            fc3_bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn test_forward_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();

        let n = MlFeatures::NUM_FEATURES;
        // Single sample
        let input = Tensor::zeros((1, n), DType::F32, &device).unwrap();
        let output = model.forward(&input).unwrap();
        assert_eq!(output.dims(), &[1, 1]);

        // Batch
        let batch = Tensor::zeros((16, n), DType::F32, &device).unwrap();
        let output = model.forward(&batch).unwrap();
        assert_eq!(output.dims(), &[16, 1]);
    }

    #[test]
    fn test_logits_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();

        let n = MlFeatures::NUM_FEATURES;
        let input = Tensor::zeros((4, n), DType::F32, &device).unwrap();
        let logits = model.forward_logits(&input).unwrap();
        assert_eq!(logits.dims(), &[4, 1]);
    }

    #[test]
    fn test_predict_one() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();

        let features = [0.0_f32; MlFeatures::NUM_FEATURES];
        let prob = model.predict_one(&features, &device).unwrap();
        assert!((0.0..=1.0).contains(&prob), "got {prob}");
    }

    #[test]
    fn test_output_range() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();

        for val in [-100.0_f32, -1.0, 0.0, 1.0, 100.0] {
            let features = [val; MlFeatures::NUM_FEATURES];
            let prob = model.predict_one(&features, &device).unwrap();
            assert!(
                (0.0..=1.0).contains(&prob),
                "val={val}, prob={prob} out of range"
            );
        }
    }

    #[test]
    fn test_raw_weights_matches_candle() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = MidPredictor::new(vb).unwrap();
        let weights = model.extract_weights().unwrap();

        let test_inputs: &[[f32; IN]] = &[
            [0.0; IN],
            [1.0; IN],
            [-1.0; IN],
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
            [-100.0, 50.0, 0.0, 1.0, -1.0, 0.01, -0.01],
        ];

        for input in test_inputs {
            let candle_prob = model.predict_one(input, &device).unwrap();
            let raw_prob = weights.predict(input);
            let diff = (candle_prob - raw_prob).abs();
            assert!(
                diff < 1e-5,
                "mismatch: candle={candle_prob}, raw={raw_prob}, diff={diff}, \
                 input={input:?}"
            );
        }
    }
}
