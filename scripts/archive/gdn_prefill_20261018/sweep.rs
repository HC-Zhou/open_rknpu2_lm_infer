//! Offline complete-call sweep through the public GDN ABI with independent states.
#[allow(
    non_camel_case_types,
    non_upper_case_globals,
    non_snake_case,
    dead_code
)]
mod ffi {
    include!(env!("RKNPU2_GDN_BINDINGS"));
}
use ffi::*;
use std::{ptr, time::Instant};

fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    let mut v = seed as u32 + 1;
    (0..n)
        .map(|_| {
            v = v.wrapping_mul(1664525).wrapping_add(1013904223);
            ((v >> 8) as f32 / 16777216. - 0.5) * scale
        })
        .collect()
}
struct Input {
    qkv: Vec<f32>,
    weights: Vec<f32>,
    history: Vec<f32>,
    decay: Vec<f32>,
    beta: Vec<f32>,
    state: Vec<f32>,
}
fn relative(a: &[f32], b: &[f32]) -> f64 {
    let (mut diff, mut norm) = (0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        assert!(a.is_finite());
        diff += (f64::from(a) - f64::from(b)).powi(2);
        norm += f64::from(b).powi(2);
    }
    (diff / norm.max(1e-30)).sqrt()
}
fn main() {
    let mut windows: Vec<usize> = std::env::args()
        .skip(1)
        .map(|m| m.parse().unwrap())
        .collect();
    if windows.is_empty() {
        windows = vec![13, 128, 256];
    }
    unsafe {
        let context = rk_npu_open(ptr::null());
        assert!(!context.is_null());
        let domain = rk_npu_iommu_domain_create(context, 12);
        assert!(!domain.is_null());
        for rows in windows {
            let inputs: Vec<_> = (0..24)
                .map(|h| Input {
                    qkv: values(rows * 8192, h * 13, 2.),
                    weights: values(8192 * 4, h * 19, 0.6),
                    history: values(8192 * 3, h * 23, 2.),
                    decay: values(rows * 32, h * 29, 0.2)
                        .iter()
                        .map(|v| -0.11 - v)
                        .collect(),
                    beta: values(rows * 32, h * 31, 0.8)
                        .iter()
                        .map(|v| 0.5 + v)
                        .collect(),
                    state: values(32 * 128 * 128, h * 37, 0.02),
                })
                .collect();
            let mut output = vec![0.; rows * 4096];
            let mut state = vec![0.; 32 * 128 * 128];
            let mut history = vec![0.; 8192 * 3];
            let (mut reference_output, mut reference_state) = (Vec::new(), Vec::new());
            // Start with the old default as a common numerical reference.
            let mut candidates = vec![(64, 4, 7)];
            for chunk in [16, 32, 64, 128] {
                for threads in [1, 2, 3, 4] {
                    for mask in [1, 3, 7] {
                        if (chunk, threads, mask) != (64, 4, 7) {
                            candidates.push((chunk, threads, mask));
                        }
                    }
                }
            }
            // Reverse one shape's scan to reduce a persistent order bias.
            if rows == 256 {
                candidates[1..].reverse();
            }
            for (chunk, threads, mask) in candidates {
                let mut cfg = rk_npu_gdn_config::default();
                rk_npu_gdn_config_init(&mut cfg, rows as i32);
                cfg.chunk_size = chunk;
                cfg.cpu_threads = threads;
                cfg.core_mask = mask;
                cfg.precision = RK_NPU_GDN_TF32;
                cfg.head_mapping = RK_NPU_GDN_HEAD_GROUPED;
                let mut work = ptr::null_mut();
                assert_eq!(rk_npu_gdn_prepare(domain, &cfg, &mut work), 0);
                let mut totals = Vec::new();
                let mut walls = Vec::new();
                let mut parts = [0.; 8];
                for (i, layer) in [0, 0, 7, 15, 23, 7, 0].into_iter().enumerate() {
                    let x = &inputs[layer];
                    let mut t = rk_npu_gdn_timings::default();
                    let start = Instant::now();
                    assert_eq!(
                        rk_npu_gdn_run_projected(
                            work,
                            x.qkv.as_ptr(),
                            x.weights.as_ptr(),
                            x.history.as_ptr(),
                            x.decay.as_ptr(),
                            x.beta.as_ptr(),
                            x.state.as_ptr(),
                            output.as_mut_ptr(),
                            state.as_mut_ptr(),
                            history.as_mut_ptr(),
                            &mut t
                        ),
                        0
                    );
                    let wall_us = start.elapsed().as_secs_f64() * 1e6;
                    if i > 0 {
                        totals.push(t.total_us);
                        walls.push(wall_us);
                        for (a, b) in parts.iter_mut().zip([
                            t.prepare_us,
                            t.coefficients_us,
                            t.solve_us,
                            t.collect_us,
                            t.hardware_us,
                            t.gram_hardware_us,
                            t.predict_hardware_us,
                            t.update_hardware_us,
                        ]) {
                            *a += b / 6.;
                        }
                    }
                }
                if reference_output.is_empty() {
                    reference_output.clone_from(&output);
                    reference_state.clone_from(&state);
                }
                println!(
                    "{{\"rows\":{rows},\"chunk\":{chunk},\"threads\":{threads},\"mask\":{mask},\"total_us\":{totals:?},\"wall_us\":{walls:?},\"parts_mean_us\":{parts:?},\"output_relative_l2\":{},\"state_relative_l2\":{}}}",
                    relative(&output, &reference_output),
                    relative(&state, &reference_state)
                );
                rk_npu_gdn_workspace_free(work);
            }
        }
        rk_npu_iommu_domain_free(domain);
        rk_npu_close(context);
    }
}
