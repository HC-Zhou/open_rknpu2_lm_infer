//! FP32 normalization with a fixed scalar reduction order.
use super::activation::{silu, silu_in_place};
use anyhow::{Result, ensure};
use rayon::{ThreadPool, prelude::*};

pub fn l2_normalize(values: &mut [f32], width: usize) -> Result<()> {
    ensure!(
        width > 0 && values.len().is_multiple_of(width),
        "L2 norm shape mismatch"
    );
    for row in values.chunks_mut(width) {
        let scale = (row.iter().map(|v| v * v).sum::<f32>() + 1e-6)
            .sqrt()
            .recip();
        for v in row {
            *v *= scale;
        }
    }
    Ok(())
}

/// `zero_centered` selects effective weights `1 + weight` instead of `weight`.
pub fn rms_norm(
    input: &[f32],
    weight: &[f32],
    epsilon: f32,
    zero_centered: bool,
    output: &mut [f32],
) -> Result<()> {
    let width = weight.len();
    ensure!(
        width > 0 && input.len().is_multiple_of(width) && output.len() == input.len(),
        "RMSNorm shape mismatch"
    );
    ensure!(
        epsilon.is_finite() && epsilon > 0.0,
        "invalid RMSNorm epsilon"
    );
    for (x, y) in input.chunks(width).zip(output.chunks_mut(width)) {
        let scale = (x.iter().map(|v| v * v).sum::<f32>() / width as f32 + epsilon)
            .sqrt()
            .recip();
        for ((x, y), w) in x.iter().zip(y).zip(weight) {
            *y = (x * scale) * (if zero_centered { 1.0 + w } else { *w });
        }
    }
    Ok(())
}

pub fn gated_norm(
    input: &[f32],
    z: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<()> {
    ensure!(z.len() == input.len(), "GDN gate shape mismatch");
    rms_norm(input, weight, epsilon, false, output)?;
    for (out, z) in output.iter_mut().zip(z) {
        *out *= silu(*z);
    }
    Ok(())
}

pub fn gated_norm_parallel(
    pool: &ThreadPool,
    input: &[f32],
    z: &[f32],
    weight: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<()> {
    ensure!(
        !weight.is_empty()
            && input.len().is_multiple_of(weight.len())
            && z.len() == input.len()
            && output.len() == input.len()
            && epsilon.is_finite()
            && epsilon > 0.0,
        "GDN gated norm shape/epsilon mismatch"
    );
    pool.install(|| {
        input
            .par_chunks(weight.len())
            .zip(z.par_chunks(weight.len()))
            .zip(output.par_chunks_mut(weight.len()))
            .for_each(|((x, z), out)| {
                // Preserve the scalar norm's FP32 reduction order. The expensive
                // gate uses ggml's vector SiLU on AArch64.
                let scale = (x.iter().map(|v| v * v).sum::<f32>() / weight.len() as f32 + epsilon)
                    .sqrt()
                    .recip();
                out.copy_from_slice(z);
                silu_in_place(out);
                for ((out, x), w) in out.iter_mut().zip(x).zip(weight) {
                    *out *= (x * scale) * w;
                }
            });
    });
    Ok(())
}
