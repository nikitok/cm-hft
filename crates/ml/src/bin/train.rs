//! Training binary: trains mid-price predictor on replay data.
//!
//! Pipeline:
//! 1. Load replay events from JSONL.gz files (all symbols by default)
//! 2. Walk events maintaining OrderBook + signal state per symbol
//! 3. Extract MlFeatures at each book update
//! 4. Label: mid_price[t+K] > mid_price[t] → 1.0, else 0.0 (with label smoothing)
//! 5. Normalize features (offline z-score, save stats for inference)
//! 6. Shuffle data
//! 7. Train with BCE loss, AdamW optimizer, cosine LR schedule
//! 8. Save weights + normalization stats

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, optim, VarBuilder, VarMap};
use candle_nn::Optimizer;
use clap::Parser;
use flate2::read::GzDecoder;
use serde::Deserialize;

use cm_core::types::*;
use cm_market_data::orderbook::OrderBook;
use cm_ml::features::MlFeatures;
use cm_ml::model::MidPredictor;
use cm_ml::normalize::NormStats;

// ── Inline signal components ──
// Duplicated from cm-strategy::signals to avoid circular dependency.

struct Ema {
    value: f64,
    alpha: f64,
    initialized: bool,
}

impl Ema {
    fn new(span: usize) -> Self {
        Self {
            value: 0.0,
            alpha: 2.0 / (span as f64 + 1.0),
            initialized: false,
        }
    }

    fn update(&mut self, x: f64) -> f64 {
        if !self.initialized {
            self.value = x;
            self.initialized = true;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
        self.value
    }
}

struct VolatilityTracker {
    ema: Ema,
    last_price: Option<f64>,
}

impl VolatilityTracker {
    fn new(span: usize) -> Self {
        Self {
            ema: Ema::new(span),
            last_price: None,
        }
    }

    fn update(&mut self, mid: f64) {
        if let Some(last) = self.last_price {
            if last > 0.0 {
                let abs_return = ((mid - last) / last).abs();
                self.ema.update(abs_return);
            }
        }
        self.last_price = Some(mid);
    }

    fn volatility_bps(&self) -> f64 {
        if !self.ema.initialized {
            return 0.0;
        }
        self.ema.value * 10_000.0
    }
}

struct TradeFlowSignal {
    buy_pressure: Ema,
    sell_pressure: Ema,
}

impl TradeFlowSignal {
    fn new(span: usize) -> Self {
        Self {
            buy_pressure: Ema::new(span),
            sell_pressure: Ema::new(span),
        }
    }

    fn update(&mut self, is_buy: bool, notional: f64) {
        if is_buy {
            self.buy_pressure.update(notional);
            self.sell_pressure.update(0.0);
        } else {
            self.buy_pressure.update(0.0);
            self.sell_pressure.update(notional);
        }
    }

    fn imbalance(&self) -> f64 {
        if !self.buy_pressure.initialized || !self.sell_pressure.initialized {
            return 0.0;
        }
        let total = self.buy_pressure.value + self.sell_pressure.value;
        if total < 1e-12 {
            return 0.0;
        }
        (self.buy_pressure.value - self.sell_pressure.value) / total
    }
}

struct VpinTracker {
    bucket_size: f64,
    n_buckets: usize,
    current_buy_notional: f64,
    current_sell_notional: f64,
    current_total: f64,
    buckets: VecDeque<f64>,
}

impl VpinTracker {
    fn new(bucket_size: f64, n_buckets: usize) -> Self {
        Self {
            bucket_size,
            n_buckets,
            current_buy_notional: 0.0,
            current_sell_notional: 0.0,
            current_total: 0.0,
            buckets: VecDeque::with_capacity(n_buckets),
        }
    }

    fn update(&mut self, is_buy: bool, notional: f64) {
        let mut remaining = notional;
        while remaining > 0.0 {
            let space = self.bucket_size - self.current_total;
            let fill = remaining.min(space);
            if is_buy {
                self.current_buy_notional += fill;
            } else {
                self.current_sell_notional += fill;
            }
            self.current_total += fill;
            remaining -= fill;
            if self.current_total >= self.bucket_size - 1e-12 {
                let imbalance = (self.current_buy_notional - self.current_sell_notional).abs()
                    / self.bucket_size;
                self.buckets.push_back(imbalance);
                if self.buckets.len() > self.n_buckets {
                    self.buckets.pop_front();
                }
                self.current_buy_notional = 0.0;
                self.current_sell_notional = 0.0;
                self.current_total = 0.0;
            }
        }
    }

    fn vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        self.buckets.iter().sum::<f64>() / self.buckets.len() as f64
    }
}

// ── End inline signals ──

/// Per-symbol state for feature extraction.
struct SymbolState {
    book: OrderBook,
    vol_tracker: VolatilityTracker,
    trade_flow: TradeFlowSignal,
    fair_value_ema: Ema,
    vpin_tracker: VpinTracker,
    mid_history: VecDeque<f64>,
}

impl SymbolState {
    fn new(exchange: Exchange, symbol: Symbol) -> Self {
        Self {
            book: OrderBook::new(exchange, symbol),
            vol_tracker: VolatilityTracker::new(100),
            trade_flow: TradeFlowSignal::new(50),
            fair_value_ema: Ema::new(20),
            vpin_tracker: VpinTracker::new(50_000.0, 20),
            mid_history: VecDeque::with_capacity(100),
        }
    }
}

#[derive(Parser)]
#[command(name = "cm-ml-train", about = "Train mid-price predictor on replay data")]
struct Args {
    /// Directory containing JSONL.gz replay files.
    #[arg(long, default_value = "testdata")]
    data: PathBuf,

    /// Output path for trained weights.
    #[arg(long, default_value = "models/mid_predictor.safetensors")]
    output: PathBuf,

    /// Lookahead window (number of book updates).
    #[arg(long, default_value_t = 50)]
    lookahead: usize,

    /// Number of training epochs.
    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// Learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// Train/validation split ratio.
    #[arg(long, default_value_t = 0.8)]
    train_ratio: f64,

    /// Batch size.
    #[arg(long, default_value_t = 512)]
    batch_size: usize,

    /// Symbol filter (e.g., "btcusdt"). If empty, use all files.
    #[arg(long, default_value = "")]
    symbol: String,

    /// Seed for reproducibility.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Label smoothing factor. 0.0=no smoothing, 0.1=standard.
    #[arg(long, default_value_t = 0.05)]
    label_smoothing: f64,

    /// Early stopping patience (epochs without val improvement).
    #[arg(long, default_value_t = 30)]
    patience: usize,
}

#[derive(Debug, Deserialize)]
struct RecordedEvent {
    #[allow(dead_code)]
    ts_ns: u64,
    kind: String,
    symbol: Option<String>,
    data: serde_json::Value,
}

enum ReplayEvent {
    Book(String, BookUpdate),
    Trade(String, Trade),
}

fn load_events(path: &Path) -> Result<Vec<ReplayEvent>> {
    let file = std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader: Box<dyn BufRead> = if path.extension().map_or(false, |e| e == "gz") {
        Box::new(BufReader::new(GzDecoder::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    // Try to extract symbol from filename (e.g., "bybit_BTCUSDT_20240101.jsonl.gz")
    let filename = path.file_name().unwrap_or_default().to_string_lossy();
    let default_symbol = filename
        .split('_')
        .nth(1)
        .unwrap_or("UNKNOWN")
        .to_string();

    let mut events = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let recorded: RecordedEvent = serde_json::from_str(&line)?;
        let sym = recorded.symbol.unwrap_or_else(|| default_symbol.clone());
        match recorded.kind.as_str() {
            "book" => {
                let book: BookUpdate = serde_json::from_value(recorded.data)?;
                events.push(ReplayEvent::Book(sym, book));
            }
            "trade" => {
                let trade: Trade = serde_json::from_value(recorded.data)?;
                events.push(ReplayEvent::Trade(sym, trade));
            }
            _ => {}
        }
    }
    Ok(events)
}

fn find_data_files(dir: &Path, symbol_filter: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if !name.ends_with(".jsonl.gz") && !name.ends_with(".jsonl") {
                continue;
            }
            if !symbol_filter.is_empty()
                && !name.to_lowercase().contains(&symbol_filter.to_lowercase())
            {
                continue;
            }
            files.push(entry.path());
        }
    }
    files.sort();
    files
}

/// Extract features and mid prices from replay events across all symbols.
fn extract_samples(events: &[ReplayEvent]) -> (Vec<[f64; 7]>, Vec<f64>) {
    let mut states: HashMap<String, SymbolState> = HashMap::new();

    let mut features_vec = Vec::new();
    let mut mids_vec = Vec::new();

    for event in events {
        match event {
            ReplayEvent::Trade(sym, trade) => {
                let state = states
                    .entry(sym.clone())
                    .or_insert_with(|| SymbolState::new(Exchange::Bybit, Symbol::new(sym)));
                let price = trade.price.to_f64();
                let qty = trade.quantity.to_f64();
                let notional = price * qty;
                let is_buy = trade.side == Side::Buy;
                state.trade_flow.update(is_buy, notional);
                state.vpin_tracker.update(is_buy, notional);
            }
            ReplayEvent::Book(sym, update) => {
                let state = states
                    .entry(sym.clone())
                    .or_insert_with(|| SymbolState::new(Exchange::Bybit, Symbol::new(sym)));
                state.book.apply_snapshot(&update.bids, &update.asks, 0);
                let mid = match state.book.mid_price() {
                    Some(p) => p.to_f64(),
                    None => continue,
                };

                state.vol_tracker.update(mid);
                state.fair_value_ema.update(mid);

                let bids = state.book.bid_depth(3);
                let asks = state.book.ask_depth(3);
                let bid_vol: f64 = bids.iter().map(|l| l.quantity.to_f64()).sum();
                let ask_vol: f64 = asks.iter().map(|l| l.quantity.to_f64()).sum();
                let total = bid_vol + ask_vol;
                let book_imbalance = if total > 1e-12 {
                    (bid_vol - ask_vol) / total
                } else {
                    0.0
                };

                let spread_bps = state.book.spread_bps().unwrap_or(0.0);

                let recent_return_bps = if let Some(&prev) = state.mid_history.back() {
                    if prev > 0.0 {
                        (mid - prev) / prev * 10_000.0
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };
                state.mid_history.push_back(mid);
                if state.mid_history.len() > 20 {
                    state.mid_history.pop_front();
                }

                let features = MlFeatures {
                    book_imbalance,
                    trade_flow_imbalance: state.trade_flow.imbalance(),
                    vpin: state.vpin_tracker.vpin(),
                    volatility_bps: state.vol_tracker.volatility_bps(),
                    spread_bps,
                    recent_return_bps,
                    normalized_position: 0.0,
                };

                features_vec.push(features.to_array());
                mids_vec.push(mid);
            }
        }
    }

    (features_vec, mids_vec)
}

/// Assign binary labels with optional label smoothing.
/// mid[t+K] > mid[t] → 1-ε, else ε (where ε = smoothing/2).
fn assign_labels(mids: &[f64], lookahead: usize, smoothing: f64) -> Vec<f32> {
    let n = mids.len();
    let eps = (smoothing / 2.0) as f32;
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        if i + lookahead < n {
            labels.push(if mids[i + lookahead] > mids[i] {
                1.0 - eps
            } else {
                eps
            });
        } else {
            labels.push(0.5_f32); // ambiguous — will be trimmed
        }
    }
    labels
}

/// Offline z-score normalization: compute mean/std per feature on training set,
/// apply to both training and validation.
fn normalize_features(
    train: &[[f64; 7]],
    val: &[[f64; 7]],
) -> (Vec<[f32; 7]>, Vec<[f32; 7]>, NormStats) {
    let n = train.len() as f64;
    let mut mean = [0.0_f64; 7];
    let mut m2 = [0.0_f64; 7];

    // Two-pass: mean, then variance.
    for f in train {
        for i in 0..7 {
            mean[i] += f[i];
        }
    }
    for i in 0..7 {
        mean[i] /= n;
    }
    for f in train {
        for i in 0..7 {
            let d = f[i] - mean[i];
            m2[i] += d * d;
        }
    }
    let mut std = [0.0_f64; 7];
    for i in 0..7 {
        std[i] = (m2[i] / n).sqrt();
        if std[i] < 1e-12 {
            std[i] = 1.0;
        }
    }

    let stats = NormStats {
        mean: mean.to_vec(),
        std: std.to_vec(),
    };

    let normalize = |features: &[[f64; 7]]| -> Vec<[f32; 7]> {
        features
            .iter()
            .map(|f| {
                let mut out = [0.0_f32; 7];
                for i in 0..7 {
                    out[i] = ((f[i] - mean[i]) / std[i]).clamp(-5.0, 5.0) as f32;
                }
                out
            })
            .collect()
    };

    let train_norm = normalize(train);
    let val_norm = normalize(val);
    (train_norm, val_norm, stats)
}

/// Fisher-Yates shuffle with seed.
fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    // Simple LCG PRNG for reproducible shuffle.
    let mut rng = seed;
    for i in (1..n).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }
    indices
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    // 1. Find data files.
    let files = find_data_files(&args.data, &args.symbol);
    if files.is_empty() {
        anyhow::bail!("no data files found in {}", args.data.display());
    }
    println!("Found {} data files", files.len());
    for f in &files {
        println!("  {}", f.display());
    }

    // 2. Load and extract features from ALL files.
    let mut all_features = Vec::new();
    let mut all_mids = Vec::new();
    for file in &files {
        println!("Loading {}...", file.display());
        let events = load_events(file)?;
        let (features, mids) = extract_samples(&events);
        println!("  {} book samples", features.len());
        all_features.extend(features);
        all_mids.extend(mids);
    }
    println!("Total samples: {}", all_features.len());

    // 3. Assign labels.
    let labels = assign_labels(&all_mids, args.lookahead, args.label_smoothing);

    // Trim to valid labels.
    let valid_end = all_features.len().saturating_sub(args.lookahead);
    let features = &all_features[..valid_end];
    let labels = &labels[..valid_end];
    println!("Valid training samples: {}", features.len());

    if features.is_empty() {
        anyhow::bail!("no valid training samples after applying lookahead");
    }

    // Label distribution (before smoothing).
    let up_count = labels.iter().filter(|&&l| l > 0.5).count();
    let down_count = labels.len() - up_count;
    println!(
        "Labels: {} up ({:.1}%), {} down ({:.1}%)",
        up_count,
        up_count as f64 / labels.len() as f64 * 100.0,
        down_count,
        down_count as f64 / labels.len() as f64 * 100.0
    );

    // 4. Train/val split (temporal: first 80% train, last 20% val).
    let split_idx = (features.len() as f64 * args.train_ratio) as usize;
    let (train_features, val_features) = features.split_at(split_idx);
    let (train_labels, val_labels) = labels.split_at(split_idx);
    println!(
        "Train: {} samples, Val: {} samples",
        train_features.len(),
        val_features.len()
    );

    // 5. Normalize features (offline z-score from training set).
    let (train_norm, val_norm, norm_stats) = normalize_features(train_features, val_features);
    println!(
        "Feature means: {:?}",
        norm_stats.mean.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    println!(
        "Feature stds:  {:?}",
        norm_stats.std.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );

    // Save normalization stats alongside model.
    let norm_path = args.output.with_extension("norm.json");
    if let Some(parent) = norm_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    norm_stats.save(&norm_path)?;
    println!("Saved norm stats to {}", norm_path.display());

    // 6. Setup model.
    let device = Device::Cpu;
    println!("Using CPU (optimal for small model)");

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = MidPredictor::new(vb)?;

    let params_count: usize = varmap.all_vars().iter().map(|v| v.elem_count()).sum();
    println!("Model parameters: {}", params_count);

    let mut optimizer = optim::AdamW::new(
        varmap.all_vars(),
        optim::ParamsAdamW {
            lr: args.lr,
            weight_decay: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        },
    )?;

    // Pre-compute validation tensors.
    let val_features_flat: Vec<f32> = val_norm.iter().flat_map(|f| f.iter().copied()).collect();
    let val_labels_flat: Vec<f32> = val_labels.to_vec();
    let n_val = val_norm.len();
    let val_x = Tensor::from_vec(val_features_flat, (n_val, 7), &device)?;
    let val_y = Tensor::from_vec(val_labels_flat, (n_val, 1), &device)?;

    // For accuracy calculation, use hard labels (no smoothing).
    let val_hard_labels: Vec<bool> = val_labels.iter().map(|&l| l > 0.5).collect();

    // 7. Training loop with shuffling + warm-up + cosine LR schedule.
    let n_train = train_norm.len();
    let batch_size = args.batch_size.min(n_train);
    let batches_per_epoch = (n_train + batch_size - 1) / batch_size;
    let total_steps = args.epochs * batches_per_epoch;
    let warmup_steps = batches_per_epoch * 5; // 5 epochs warmup

    let mut best_val_loss = f64::MAX;
    let mut best_val_acc = 0.0_f64;
    let mut best_epoch = 0_usize;
    let mut step = 0_usize;
    let mut patience_counter = 0_usize;

    for epoch in 0..args.epochs {
        // Shuffle training data each epoch.
        let indices = shuffle_indices(n_train, args.seed.wrapping_add(epoch as u64));

        let mut epoch_loss = 0.0;
        let mut n_batches = 0;

        for batch_start in (0..n_train).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(n_train);
            let bs = batch_end - batch_start;

            // Learning rate: linear warmup + cosine decay.
            let lr = if step < warmup_steps {
                args.lr * (step as f64 / warmup_steps as f64)
            } else {
                let progress =
                    (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                args.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
            };
            optimizer.set_learning_rate(lr);
            step += 1;

            // Gather shuffled batch.
            let mut batch_features = Vec::with_capacity(bs * 7);
            let mut batch_labels = Vec::with_capacity(bs);
            for &idx in &indices[batch_start..batch_end] {
                batch_features.extend_from_slice(&train_norm[idx]);
                batch_labels.push(train_labels[idx]);
            }

            let x = Tensor::from_vec(batch_features, (bs, 7), &device)?;
            let y = Tensor::from_vec(batch_labels, (bs, 1), &device)?;

            let logits = model.forward_logits(&x)?;
            let bce = loss::binary_cross_entropy_with_logit(&logits, &y)?;
            optimizer.backward_step(&bce)?;

            epoch_loss += bce.to_vec0::<f32>()? as f64;
            n_batches += 1;
        }

        // Validation.
        let val_logits = model.forward_logits(&val_x)?;
        let val_loss = loss::binary_cross_entropy_with_logit(&val_logits, &val_y)?;
        let val_loss_f = val_loss.to_vec0::<f32>()? as f64;

        let val_pred_vec = val_logits.flatten_all()?.to_vec1::<f32>()?;
        let correct = val_pred_vec
            .iter()
            .zip(val_hard_labels.iter())
            .filter(|(&p, &l)| (p > 0.0) == l)
            .count();
        let val_acc = correct as f64 / n_val as f64 * 100.0;

        let improved = val_loss_f < best_val_loss;
        if improved {
            best_val_loss = val_loss_f;
            best_val_acc = val_acc;
            best_epoch = epoch + 1;
            patience_counter = 0;
            // Save best model.
            if let Some(parent) = args.output.parent() {
                std::fs::create_dir_all(parent)?;
            }
            varmap.save(&args.output)?;
        } else {
            patience_counter += 1;
        }

        let lr_now = if step < warmup_steps {
            args.lr * (step as f64 / warmup_steps as f64)
        } else {
            let progress = (step - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
            args.lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        };

        if (epoch + 1) % 10 == 0 || improved || epoch == 0 {
            println!(
                "Epoch {:>3}/{}: train_loss={:.4}, val_loss={:.4}, val_acc={:.1}%, lr={:.2e}{}",
                epoch + 1,
                args.epochs,
                epoch_loss / n_batches as f64,
                val_loss_f,
                val_acc,
                lr_now,
                if improved { " *best*" } else { "" },
            );
        }

        // Early stopping.
        if patience_counter >= args.patience {
            println!(
                "Early stopping at epoch {} (no improvement for {} epochs)",
                epoch + 1,
                args.patience
            );
            break;
        }
    }

    println!(
        "\nBest: epoch={}, val_loss={:.4}, val_acc={:.1}%",
        best_epoch, best_val_loss, best_val_acc
    );
    println!("Model saved to {}", args.output.display());
    println!("Norm stats saved to {}", norm_path.display());

    Ok(())
}
