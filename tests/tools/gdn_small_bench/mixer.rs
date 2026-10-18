use anyhow::Result;
use open_rknpu2_lm_infer::{
    layers::gated_delta::{LayerState, MixerWeights, Projected, Scratch},
    ops::gated_delta::GdnDimensions,
};
use rayon::ThreadPool;
use serde_json::{Value, json};
use std::{hint::black_box, time::Instant};

pub fn bench(pool: &ThreadPool, rows: usize, layers: usize) -> Result<Value> {
    let dim = GdnDimensions::new(16, 32, 128, 128)?;
    let weights = MixerWeights::new(
        dim,
        4,
        vec![0.1; dim.conv_width() * 4],
        vec![-2.0; 32],
        vec![0.0; 32],
        vec![1.0; 128],
        1e-6,
    )?;
    let series = |n, scale| {
        (0..n)
            .map(|i| (((i * 17) % 251) as f32 - 125.0) * scale / 125.0)
            .collect::<Vec<_>>()
    };
    let qkv = series(rows * dim.conv_width(), 0.7);
    let z = series(rows * dim.value_width(), 1.0);
    let a = series(rows * 32, 0.3);
    let b = series(rows * 32, 0.5);
    let input = Projected::new(rows, &qkv, &z, &a, &b);
    let mut bank = (0..layers)
        .map(|_| LayerState::new(dim, 4, 8))
        .collect::<Result<Vec<_>>>()?;
    let mut scratch = Scratch::default();
    let mut run = |bank: &mut [LayerState]| -> Result<()> {
        for layer in bank {
            black_box(layer.forward(pool, &weights, input, &mut scratch)?);
        }
        Ok(())
    };
    run(&mut bank)?;
    run(&mut bank)?;
    let loops = (128 / rows / layers).clamp(1, 100);
    let mut samples = Vec::new();
    for _ in 0..7 {
        for layer in &mut bank {
            layer.reset();
        }
        run(&mut bank)?; // Nonzero state/history, outside timing.
        let start = Instant::now();
        for _ in 0..loops {
            run(&mut bank)?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / loops as f64);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    eprintln!("mixer rows={rows}, layers={layers}: {mean:.6} ms");
    Ok(json!({"rows":rows,"layers":layers,"loops_per_sample":loops,
        "samples_ms":samples,"mean_ms":mean,"mean_ms_per_layer":mean / layers as f64,
        "scratch_bytes":scratch.allocated_bytes()}))
}
