//! Candle neural network for mid-price direction prediction.
//!
//! Architecture: 7 → 64 → 32 → 1 feedforward with ReLU activations
//! and sigmoid output. ~2,625 parameters total.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

use crate::features::MlFeatures;

/// Feedforward network predicting P(mid goes up) ∈ [0, 1].
pub struct MidPredictor {
    fc1: Linear, // 7 → 64
    fc2: Linear, // 64 → 32
    fc3: Linear, // 32 → 1
}

impl MidPredictor {
    const IN: usize = MlFeatures::NUM_FEATURES;
    const H1: usize = 64;
    const H2: usize = 32;

    /// Create a new model with trainable weights (for training).
    pub fn new(vb: VarBuilder<'_>) -> Result<Self> {
        let fc1 = linear(Self::IN, Self::H1, vb.pp("fc1"))?;
        let fc2 = linear(Self::H1, Self::H2, vb.pp("fc2"))?;
        let fc3 = linear(Self::H2, 1, vb.pp("fc3"))?;
        Ok(Self { fc1, fc2, fc3 })
    }

    /// Load a trained model from a safetensors file.
    pub fn load(path: &std::path::Path, device: &Device) -> Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[path], DType::F32, device)?
        };
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
    pub fn predict_one(&self, features: &[f32; MlFeatures::NUM_FEATURES], device: &Device) -> Result<f32> {
        let input = Tensor::from_slice(features.as_slice(), (1, Self::IN), device)?;
        let output = self.forward(&input)?;
        let val = output.flatten_all()?.to_vec1::<f32>()?;
        Ok(val[0])
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
        assert!(prob >= 0.0 && prob <= 1.0, "got {prob}");
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
                prob >= 0.0 && prob <= 1.0,
                "val={val}, prob={prob} out of range"
            );
        }
    }
}
