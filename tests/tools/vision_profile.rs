//! Complete ViT stage timings, bypassing only the image-feature cache.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    cli::ModelArgs,
    model::{checkpoint::TensorStore, qwen35::vision::Encoder},
    npu,
    validation::BoardLock,
    vision,
};
use std::{io::Write, path::PathBuf, sync::atomic::AtomicBool, time::Instant};

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    model: ModelArgs,
    #[arg(long)]
    image: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    repeats: usize,
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(args.repeats > 0, "repeats must be positive");
    let mut options = args.model.options()?;
    ensure!(
        !options.tune_search,
        "profile with precomputed, load-only strategies"
    );
    let _lock = BoardLock::acquire()?;
    npu::configure_cpu(&mut options)?;
    let image = vision::ImageProcessor::load(&options.path)?
        .context("no image processor")?
        .prepare(&std::fs::read(&args.image)?)?;
    let store = TensorStore::open(&options.path)?;
    let mut encoder = Encoder::load(&options, &store, 11)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(options.threads)
        .build()?;
    let cancel = AtomicBool::new(false);
    // The uninstrumented first call also checks that profiling/reuse preserves
    // every FP32 feature bit. This call includes cold workspace preparation.
    let start = Instant::now();
    let reference = encoder.encode(&image, &pool, &cancel)?;
    let cold_ms = start.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..args.warmup {
        encoder.clear_feature_cache();
        let features = encoder.encode(&image, &pool, &cancel)?;
        ensure!(
            reference
                .iter()
                .zip(features.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "warmup features changed"
        );
    }
    encoder.enable_profiling();
    let mut samples = Vec::new();
    for _ in 0..args.repeats {
        encoder.clear_feature_cache();
        let start = Instant::now();
        let features = encoder.encode(&image, &pool, &cancel)?;
        let total_ms = start.elapsed().as_secs_f64() * 1000.0;
        ensure!(
            reference
                .iter()
                .zip(features.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "profiled features changed"
        );
        samples.push(serde_json::json!({"total_ms": total_ms, "stages_ms": encoder.profile_ms()}));
    }
    std::fs::create_dir_all(&args.output)?;
    let mut f = std::fs::File::create(args.output.join("features.f32"))?;
    for x in reference.iter() {
        f.write_all(&x.to_le_bytes())?;
    }
    let report = serde_json::json!({
        "image": args.image, "grid": image.grid, "visual_tokens": image.tokens(),
        "cold_ms": cold_ms, "warmup": args.warmup, "samples": samples,
        "cpu_mask": options.cpu_mask, "threads": options.threads,
        "npu_mask": options.npu_mask, "feature_cache_hits": encoder.hits,
        "vision_dtype": options.vision_dtype.as_str(),
        "encode_calls": encoder.calls, "features_bitwise_equal": true,
    });
    std::fs::write(
        args.output.join("profile.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
