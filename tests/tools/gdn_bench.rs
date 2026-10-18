//! FP32 recurrence A/B against the actual llama.cpp ggml CPU op.
use anyhow::{Result, ensure};
use clap::Parser;
use open_rknpu2_lm_infer::{
    ops::gated_delta::{GdnDimensions, Updates, Workspace},
    ops::norm,
};
use serde_json::json;
use std::{io::Write, path::PathBuf, process::Command, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    reference_binary: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,7,8,32,128,512")]
    rows: Vec<usize>,
}

fn series(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 17 + seed * 29) % 251) as f32 - 125.0) * scale / 125.0)
        .collect()
}

fn error(actual: &[f32], reference: &[f32]) -> serde_json::Value {
    assert_eq!(actual.len(), reference.len());
    let max_abs = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let energy = reference.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
    let squared = actual
        .iter()
        .zip(reference)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>();
    json!({"max_abs":max_abs,"nrmse":(squared / energy.max(1e-30)).sqrt(),
        "passed": max_abs < 2e-5 && actual.iter().all(|x|x.is_finite())})
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        (1..=4).contains(&args.threads),
        "threads must be within 1..4"
    );
    std::fs::create_dir_all(&args.output)?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()?;
    let dim = GdnDimensions::new(16, 32, 128, 128)?;
    let mut cases = Vec::new();
    for rows in args.rows {
        ensure!(rows > 0 && rows <= 512, "rows must be within 1..512");
        let mut q = series(rows * dim.key_width(), 1, 1.0);
        let mut k = series(q.len(), 2, 1.0);
        norm::l2_normalize(&mut q, dim.key_dim())?;
        norm::l2_normalize(&mut k, dim.key_dim())?;
        let value = series(rows * dim.value_width(), 3, 0.3);
        let log_decay = vec![-0.07; rows * dim.value_heads()];
        let beta = vec![0.37; log_decay.len()];
        let initial = series(dim.state_elements(), 4, 0.01);
        let input_path = args.output.join(format!("input-{rows}.f32"));
        let mut input = std::fs::File::create(&input_path)?;
        for data in [&q, &k] {
            for t in 0..rows {
                for h in 0..dim.value_heads() {
                    let offset = t * dim.key_width() + (h / 2) * dim.key_dim();
                    for v in &data[offset..offset + dim.key_dim()] {
                        input.write_all(&v.to_le_bytes())?;
                    }
                }
            }
        }
        for data in [&value, &log_decay, &beta, &initial] {
            for v in data {
                input.write_all(&v.to_le_bytes())?;
            }
        }
        drop(input);
        let loops = (512 / rows).clamp(3, 100);
        let samples = 7;
        let updates = Updates {
            rows,
            key: &k,
            value: &value,
            log_decay: &log_decay,
            beta: &beta,
        };
        let mut state = initial.clone();
        let mut output = vec![0.0; value.len()];
        let mut timings = Vec::new();
        let mut final_results = Vec::new();
        // Measure both the copied ggml loop and the fused version through the
        // real Rust ABI, including validation, state copy and token-major output.
        for mut workspace in [Workspace::ggml_loop_reference(), Workspace::default()] {
            let mut run = || -> Result<()> {
                workspace.run_from(&pool, dim, &initial, &mut state, &q, updates, &mut output)
            };
            run()?;
            run()?;
            let mut milliseconds = Vec::new();
            for _ in 0..samples {
                let start = Instant::now();
                for _ in 0..loops {
                    run()?;
                }
                milliseconds.push(start.elapsed().as_secs_f64() * 1000.0 / loops as f64);
            }
            timings.push(milliseconds);
            final_results.push((output.clone(), state.clone()));
        }
        let out_path = args.output.join(format!("ggml-{rows}.f32"));
        let reference = Command::new(&args.reference_binary)
            .arg(&input_path)
            .arg(&out_path)
            .arg(rows.to_string())
            .arg(args.threads.to_string())
            .arg(loops.to_string())
            .arg(samples.to_string())
            .output()?;
        ensure!(
            reference.status.success(),
            "ggml reference failed: {}",
            String::from_utf8_lossy(&reference.stderr)
        );
        let timing: serde_json::Value = serde_json::from_slice(&reference.stdout)?;
        let reference: Vec<f32> = std::fs::read(&out_path)?
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let mut comparisons = Vec::new();
        for (output, state) in final_results {
            let output_error = error(&output, &reference[..output.len()]);
            let state_error = error(&state, &reference[output.len()..]);
            ensure!(
                output_error["passed"] == true && state_error["passed"] == true,
                "GDN differs from ggml: output={output_error}, state={state_error}"
            );
            comparisons.push(json!({"output":output_error,"state":state_error}));
        }
        cases.push(
            json!({"rows": rows,"loops_per_sample":loops,"cpp_unfused_ms":timings[0],
            "cpp_fused_ms":timings[1],"ggml_ms":timing["milliseconds"],
            "unfused":comparisons[0],"fused":comparisons[1]}),
        );
    }
    let report = json!({"threads":args.threads,"key_heads":16,"value_heads":32,
        "key_dim":128,"value_dim":128,"state_dtype":"FP32", "nonzero_initial_state":true,
        "qk_expanded_for_reference":true,"cases":cases});
    std::fs::write(
        args.output.join("comparison.json"),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
