//! Complete-call NPU GELU versus the committed Rust CPU implementation.
//! The optional experimental bridge is loaded only by this explicit tool.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use half::f16;
use open_rknpu2_lm_infer::{ops::vision_half, validation::BoardLock};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[path = "npu_gelu_bench/bridge.rs"]
mod bridge;
use bridge::Bridge;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    bridge: PathBuf,
    #[arg(long, default_value_t = 1_048_576)]
    count: usize,
    #[arg(long, default_value_t = 7)]
    mask: u32,
    #[arg(long, conflicts_with = "finite_domain")]
    input_f32: Option<PathBuf>,
    #[arg(long)]
    finite_domain: bool,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    #[arg(long)]
    output: PathBuf,
}
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "positive repeats required");
    let _lock = BoardLock::acquire()?;
    let input = if let Some(path) = &args.input_f32 {
        let bytes = std::fs::read(path)?;
        ensure!(bytes.len().is_multiple_of(4), "FP32 input length");
        bytes
            .chunks_exact(4)
            .map(|x| f16::from_f32(f32::from_le_bytes(x.try_into().unwrap())).to_bits())
            .collect::<Vec<_>>()
    } else if args.finite_domain {
        (0..=u16::MAX).filter(|x| x & 0x7c00 != 0x7c00).collect()
    } else {
        let mut random = 0x3588abcd_u32;
        (0..args.count)
            .map(|_| {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                f16::from_f32((random as f32 / u32::MAX as f32 - 0.5) * 8.).to_bits()
            })
            .collect()
    };
    ensure!(
        input.iter().all(|x| x & 0x7c00 != 0x7c00),
        "finite FP16 input required"
    );
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build()?;
    let mut cpu = input.clone();
    vision_half::gelu(&pool, &mut cpu, true);
    let prepare = Instant::now();
    let bridge = Bridge::new(&args.bridge, input.len(), args.mask)?;
    let prepare_ms = prepare.elapsed().as_secs_f64() * 1000.;
    let mut device = input.clone();
    let mut stages = [0.; 6];
    bridge.apply(&mut device, &mut stages)?;
    let mut norm = 0.;
    let mut error = 0.;
    let mut max_abs = 0_f64;
    let mut mean = 0.;
    let mut unequal = 0;
    for (&a, &b) in cpu.iter().zip(&device) {
        let x = f16::from_bits(a).to_f64();
        let y = f16::from_bits(b).to_f64();
        ensure!(y.is_finite(), "nonfinite device output");
        norm += x * x;
        error += (x - y) * (x - y);
        max_abs = max_abs.max((x - y).abs());
        mean += y - x;
        unequal += usize::from(a != b);
    }
    let relative_l2 = (error / norm.max(f64::MIN_POSITIVE)).sqrt();
    let mut changed = input.clone();
    changed.reverse();
    bridge.apply(&mut changed, &mut stages)?;
    ensure!(
        changed.iter().eq(device.iter().rev()),
        "input update or per-core LUT mismatch"
    );
    let expected = [cpu, device];
    let mut work = input.clone();
    for _ in 0..2 {
        work.copy_from_slice(&input);
        vision_half::gelu(&pool, &mut work, true);
        work.copy_from_slice(&input);
        bridge.apply(&mut work, &mut stages)?;
    }
    let mut samples = [Vec::new(), Vec::new()];
    let mut details = Vec::new();
    for block in 0..args.repeats {
        for kind in [0, 1, 1, 0] {
            work.copy_from_slice(&input);
            let start = Instant::now();
            if kind == 0 {
                vision_half::gelu(&pool, &mut work, true);
            } else {
                bridge.apply(&mut work, &mut stages)?;
            }
            let total_us = start.elapsed().as_secs_f64() * 1e6;
            ensure!(work == expected[kind], "unstable repeated output");
            ensure!(bridge.guard_ok(), "device guard changed");
            samples[kind].push(total_us);
            details.push(
                json!({"block":block,"kind":if kind==0{"cpu"}else{"npu"},"total_us":total_us,
            "stages_us":if kind==1{Some(stages)}else{None}}),
            );
        }
    }
    let report = json!({"bridge":args.bridge,"count":input.len(),"npu_mask":args.mask,"cpu_threads":4,"input":args.input_f32,
        "finite_domain":args.finite_domain,"prepare_ms":prepare_ms,
        "scope":"complete host FP16 call; clamp/packing, cache sync, LUT reload, submit/wait and output reconstruction included",
        "relative_l2_vs_cpu_fp16":relative_l2,"max_abs_vs_cpu_fp16":max_abs,
        "mean_error":mean/input.len()as f64,"bit_mismatches":unequal,
        "cpu_median_us":median(&samples[0]),"npu_median_us":median(&samples[1]),
        "speedup":median(&samples[0])/median(&samples[1]),"samples":details,"guard_ok":true,
        "changed_input_exact":true});
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)
        .context("write GELU report")?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    ensure!(
        relative_l2 < 0.003 && max_abs < 0.02,
        "GELU error outside experimental gate"
    );
    Ok(())
}
