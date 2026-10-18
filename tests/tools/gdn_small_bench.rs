//! Short-window recurrence through the production Rust API and state swap.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::ops::gated_delta::{GdnDimensions, Updates, Workspace};
use serde_json::json;
use std::{hint::black_box, path::PathBuf, time::Instant};

#[path = "gdn_small_bench/mixer.rs"]
mod mixer;

#[derive(Parser)]
struct Args {
    cases: PathBuf,
    output: PathBuf,
    #[arg(long, default_value_t = 24)]
    layers: usize,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,7,8,128")]
    rows: Vec<usize>,
    #[arg(long)]
    mixer: bool,
}

fn floats(path: PathBuf) -> Result<Vec<f32>> {
    let raw = std::fs::read(path)?;
    ensure!(raw.len().is_multiple_of(4), "invalid FP32 file");
    Ok(raw
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.layers > 0 && (1..=4).contains(&args.threads),
        "invalid layers/threads"
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let dim = GdnDimensions::new(16, 32, 128, 128)?;
    let mut report = Vec::new();
    for rows in args.rows {
        if args.mixer {
            report.push(mixer::bench(&pool, rows, args.layers)?);
            continue;
        }
        let raw = floats(args.cases.join(format!("tokens-{rows:04}.input.f32")))?;
        let sizes = [
            rows * dim.key_width(),
            rows * dim.key_width(),
            rows * dim.value_width(),
            rows * dim.value_heads(),
            rows * dim.value_heads(),
            dim.state_elements(),
        ];
        ensure!(
            raw.len() == sizes.iter().sum::<usize>(),
            "input shape mismatch"
        );
        let mut at = 0;
        let parts: Vec<_> = sizes
            .iter()
            .map(|&size| {
                let part = &raw[at..at + size];
                at += size;
                part
            })
            .collect();
        let [q, k, v, g, beta, initial] = parts.as_slice() else {
            unreachable!()
        };
        let updates = Updates {
            rows,
            key: k,
            value: v,
            log_decay: g,
            beta,
        };
        let expected = floats(args.cases.join(format!("tokens-{rows:04}.expected.f32")))?;
        let mut bank = vec![initial.to_vec(); args.layers];
        let mut scratch = initial.to_vec();
        let mut output = vec![0.0; v.len()];
        let mut workspace = Workspace::default();
        workspace.run_from(&pool, dim, initial, &mut scratch, q, updates, &mut output)?;
        ensure!(
            output
                .iter()
                .chain(&scratch)
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
                && expected.len() == output.len() + scratch.len(),
            "differs from pinned ggml"
        );
        let mut run = |bank: &mut [Vec<f32>], scratch: &mut Vec<f32>| -> Result<()> {
            for canonical in bank {
                workspace.run_from(&pool, dim, canonical, scratch, q, updates, &mut output)?;
                std::mem::swap(canonical, scratch);
            }
            black_box(&output);
            Ok(())
        };
        run(&mut bank, &mut scratch)?;
        run(&mut bank, &mut scratch)?;
        let loops = (128 / rows / args.layers).clamp(1, 100);
        let mut samples = Vec::new();
        for _ in 0..7 {
            for state in &mut bank {
                state.copy_from_slice(initial);
            }
            scratch.copy_from_slice(initial);
            let start = Instant::now();
            for _ in 0..loops {
                run(&mut bank, &mut scratch)?;
            }
            samples.push(start.elapsed().as_secs_f64() * 1000.0 / loops as f64);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        report.push(
            json!({"rows":rows,"layers":args.layers,"loops_per_sample":loops,
            "samples_ms":samples,"mean_ms":mean,"mean_ms_per_layer":mean / args.layers as f64,
            "bit_exact":true}),
        );
        eprintln!("rows={rows}, layers={}: {mean:.6} ms", args.layers);
    }
    std::fs::write(
        args.output,
        serde_json::to_string_pretty(&json!({
        "scope": if args.mixer { "complete CPU LayerState.forward, synthetic C7-shape projected inputs; no model/NPU" }
                 else { "production Rust API, input validation, copied initial state, shared scratch and state swap; no model/NPU" },
        "threads":args.threads,"layers":args.layers,"cases":report}))?
            + "\n",
    )?;
    Ok(())
}
