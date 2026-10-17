//! Replay real weights/projected inputs through the production causal Conv + SiLU.
use anyhow::{Result, ensure};
use open_rknpu2_lm_infer::ops::causal_conv::ConvState;
use serde_json::{Value, json};
use std::{hint::black_box, path::Path, time::Instant};

const CHANNELS: usize = 8192;
const KERNEL: usize = 4;
const LAYERS: usize = 24;
const TRACE_ROWS: usize = 24;

struct Layer {
    weight: Vec<f32>,
    input: Vec<f32>,
    initial: ConvState,
    working: ConvState,
}

fn floats(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len().is_multiple_of(4), "invalid FP32 file");
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn measure(mut operation: impl FnMut() -> Result<()>, rows: usize) -> Result<Value> {
    operation()?;
    operation()?;
    let start = Instant::now();
    operation()?;
    let calibration = start.elapsed().as_secs_f64().max(1e-6);
    let loops = (0.035 / calibration).ceil().clamp(1.0, 100.0) as usize;
    let mut samples = Vec::new();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..loops {
            operation()?;
        }
        samples.push(start.elapsed().as_secs_f64() * 1000.0 / loops as f64);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let stddev = (samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (samples.len() - 1) as f64)
        .sqrt();
    Ok(
        json!({"samples_ms_24_layers":samples,"mean_ms_24_layers":mean,
        "stddev_ms_24_layers":stddev,"mean_us_per_layer":mean*1000.0/LAYERS as f64,
        "mean_us_per_token_24_layers":mean*1000.0/rows as f64,
        "loops_per_sample":loops,"warmup_calls":3}),
    )
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().collect();
    ensure!(
        args.len() == 4,
        "usage: causal_conv_bench WEIGHTS.f32 TRACE.f32 REPORT.json"
    );
    let weights = floats(Path::new(&args[1]))?;
    let trace = floats(Path::new(&args[2]))?;
    ensure!(weights.len() == LAYERS * CHANNELS * KERNEL, "weight shape");
    ensure!(trace.len() == LAYERS * TRACE_ROWS * CHANNELS, "trace shape");
    let mut results = Vec::new();
    for threads in [4, 1] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()?;
        for rows in [1, 8, 32, 128] {
            let mut bank = Vec::new();
            for layer in 0..LAYERS {
                let source =
                    &trace[layer * TRACE_ROWS * CHANNELS..(layer + 1) * TRACE_ROWS * CHANNELS];
                let mut initial = ConvState::new(CHANNELS, KERNEL)?;
                initial.append(&source[..3 * CHANNELS])?;
                let mut input = Vec::with_capacity(rows * CHANNELS);
                for row in 0..rows {
                    let at = 3 + row % (TRACE_ROWS - 3);
                    input.extend_from_slice(&source[at * CHANNELS..(at + 1) * CHANNELS]);
                }
                bank.push(Layer {
                    weight: weights[layer * CHANNELS * KERNEL..(layer + 1) * CHANNELS * KERNEL]
                        .to_vec(),
                    input,
                    working: initial.clone(),
                    initial,
                });
            }
            let mut output = vec![0.0; rows * CHANNELS];
            let mut reference = output.clone();
            let mut max_abs = 0.0f32;
            let mut squared_error = 0.0f64;
            let mut squared_reference = 0.0f64;
            let mut max_reference = 0.0f32;
            for layer in &mut bank {
                let mut scalar = layer.initial.clone();
                scalar.run(&layer.input, &layer.weight, &mut reference)?;
                layer
                    .working
                    .run_parallel(&pool, &layer.input, &layer.weight, &mut output)?;
                ensure!(
                    scalar.history() == layer.working.history(),
                    "history differs"
                );
                for (&a, &b) in output.iter().zip(&reference) {
                    ensure!(a.is_finite(), "nonfinite convolution output");
                    max_abs = max_abs.max((a - b).abs());
                    max_reference = max_reference.max(b.abs());
                    squared_error += (a as f64 - b as f64).powi(2);
                    squared_reference += (b as f64).powi(2);
                }
            }
            let relative_l2 = (squared_error / squared_reference.max(1e-30)).sqrt();
            ensure!(
                max_abs <= 1e-5 + 1e-4 * max_reference && relative_l2 <= 1e-4,
                "parallel convolution differs: abs={max_abs}, rel={relative_l2}"
            );
            let complete = measure(
                || {
                    for layer in &mut bank {
                        layer.working.clone_from(&layer.initial);
                        layer.working.run_parallel(
                            &pool,
                            &layer.input,
                            &layer.weight,
                            &mut output,
                        )?;
                        black_box(output[0]);
                        black_box(layer.working.history());
                    }
                    Ok(())
                },
                rows,
            )?;
            let run_only = measure(
                || {
                    for layer in &mut bank {
                        layer.working.run_parallel(
                            &pool,
                            &layer.input,
                            &layer.weight,
                            &mut output,
                        )?;
                        black_box(output[0]);
                        black_box(layer.working.history());
                    }
                    Ok(())
                },
                rows,
            )?;
            let history = measure(
                || {
                    for layer in &mut bank {
                        layer.working.clone_from(&layer.initial);
                        layer.working.append(&layer.input)?;
                        black_box(layer.working.history());
                    }
                    Ok(())
                },
                rows,
            )?;
            eprintln!(
                "threads={threads}, rows={rows}: 24 layers {:.4} ms",
                complete["mean_ms_24_layers"]
            );
            results.push(json!({"threads":threads,"rows":rows,
                "complete_conv_silu_and_history_copy":complete,
                "conv_silu_and_append":run_only,"history_copy_and_append_only":history,
                "correctness":{"max_abs":max_abs,"relative_l2":relative_l2,"history_exact":true,"passed":true}}));
        }
    }
    let report = json!({"layers":LAYERS,"channels":CHANNELS,"kernel":KERNEL,
        "dtype":"FP32 weights, projected inputs and history; production AArch64 NEON",
        "inputs":"real C7 weights and 24-row runtime QKV traces; first3 rows seed nonzero history, remaining21 rows cycle for longer windows",
        "scope":"CPU component replay; no NPU Conv or end-to-end NPU GDN measurement",
        "results":results});
    std::fs::write(
        Path::new(&args[3]),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    Ok(())
}
