//! Smoke test: train on synthetic data, verify loss decreases and inference works.

use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, optim, Optimizer, VarBuilder, VarMap};

use cm_ml::features::MlFeatures;
use cm_ml::model::MidPredictor;
use cm_ml::normalize::NormStats;
use cm_ml::predictor::MlSignal;

#[test]
fn smoke_train_synthetic() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MidPredictor::new(vb).unwrap();

    let n_features = MlFeatures::NUM_FEATURES;

    // Generate synthetic data: positive imbalance → up, negative → down.
    let n = 200;
    let mut features = Vec::with_capacity(n * n_features);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let sign = if i % 2 == 0 { 1.0_f32 } else { -1.0 };
        // Feature: [imbalance, flow, vpin, vol, spread, return, position]
        features.extend_from_slice(&[
            sign * 0.8,         // book_imbalance
            sign * 0.5,         // trade_flow
            0.3,                // vpin (neutral)
            2.0,                // vol
            3.0,                // spread
            sign * 5.0,         // recent return
            0.0,                // position
        ]);
        labels.push(if sign > 0.0 { 1.0_f32 } else { 0.0 });
    }

    let x = Tensor::from_vec(features, (n, n_features), &device).unwrap();
    let y = Tensor::from_vec(labels, (n, 1), &device).unwrap();

    // Initial loss.
    let pred0 = model.forward_logits(&x).unwrap();
    let loss0 = loss::binary_cross_entropy_with_logit(&pred0, &y)
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();

    // Train for a few epochs.
    let mut optimizer = optim::AdamW::new(
        varmap.all_vars(),
        optim::ParamsAdamW {
            lr: 1e-3,
            ..Default::default()
        },
    )
    .unwrap();

    for _ in 0..200 {
        let logits = model.forward_logits(&x).unwrap();
        let bce = loss::binary_cross_entropy_with_logit(&logits, &y).unwrap();
        optimizer.backward_step(&bce).unwrap();
    }

    let logits_final = model.forward_logits(&x).unwrap();
    let loss_final = loss::binary_cross_entropy_with_logit(&logits_final, &y)
        .unwrap()
        .to_vec0::<f32>()
        .unwrap();

    println!("Loss: {loss0:.4} → {loss_final:.4}");
    assert!(
        loss_final < loss0,
        "loss should decrease: {loss0} → {loss_final}"
    );

    // Check accuracy on synthetic data (logit space: >0 = up).
    let preds = logits_final.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let correct = preds
        .iter()
        .enumerate()
        .filter(|(i, &p)| {
            let expected_up = i % 2 == 0;
            (p > 0.0) == expected_up
        })
        .count();
    let acc = correct as f64 / n as f64 * 100.0;
    println!("Accuracy: {acc:.1}%");
    assert!(acc > 60.0, "accuracy should be > 60% on synthetic data, got {acc}%");
}

#[test]
fn smoke_predictor_integration() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MidPredictor::new(vb).unwrap();
    let mut signal = MlSignal::from_model(model, device);

    // Feed enough samples to warm up normalizer.
    for i in 0..200 {
        let features = MlFeatures {
            book_imbalance: (i as f64 * 0.01).sin(),
            trade_flow_imbalance: (i as f64 * 0.02).cos(),
            vpin: 0.3 + (i as f64 * 0.005).sin() * 0.2,
            volatility_bps: 2.0 + (i as f64 * 0.03).sin(),
            spread_bps: 3.0,
            recent_return_bps: (i as f64 * 0.01).sin() * 5.0,
            normalized_position: 0.0,
        };
        let val = signal.predict(&features);
        assert!(
            val >= -0.5 && val <= 0.5,
            "signal out of range: {val}"
        );
    }
}

#[test]
fn smoke_predictor_with_fixed_stats() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MidPredictor::new(vb).unwrap();
    let stats = NormStats {
        mean: vec![0.0, 0.0, 0.3, 2.0, 3.0, 0.0, 0.0],
        std: vec![0.5, 0.4, 0.1, 1.0, 1.0, 5.0, 0.5],
    };
    let mut signal = MlSignal::from_model_with_stats(model, stats, device);

    // With fixed stats, should produce signal immediately.
    let features = MlFeatures {
        book_imbalance: 0.5,
        trade_flow_imbalance: 0.3,
        vpin: 0.3,
        volatility_bps: 2.0,
        spread_bps: 3.0,
        recent_return_bps: 1.0,
        normalized_position: 0.0,
    };
    let val = signal.predict(&features);
    assert!(
        val >= -0.5 && val <= 0.5,
        "signal should be in [-0.5, 0.5], got {val}"
    );
}

#[test]
fn smoke_save_and_load() {
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MidPredictor::new(vb).unwrap();

    // Predict with original model.
    let features = [0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
    let pred1 = model.predict_one(&features, &device).unwrap();

    // Save and reload.
    let tmp = std::env::temp_dir().join("cm_ml_test_model.safetensors");
    varmap.save(&tmp).unwrap();

    let loaded = MidPredictor::load(&tmp, &device).unwrap();
    let pred2 = loaded.predict_one(&features, &device).unwrap();

    assert!(
        (pred1 - pred2).abs() < 1e-5,
        "loaded model should give same predictions: {pred1} vs {pred2}"
    );

    std::fs::remove_file(&tmp).ok();
}
