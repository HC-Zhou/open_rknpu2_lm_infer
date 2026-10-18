//! CPU operations over FP16 storage, with FP32 arithmetic and norm reductions.
//! Rayon uses the model's CPU pool; SIMD temporaries hold at most eight values.
mod convert;
mod norm;
mod rotary;
#[cfg(test)]
mod tests;
pub use rotary::Rotary;

use anyhow::{Result, ensure};
use half::f16;
use rayon::{ThreadPool, prelude::*};

const PARALLEL_ELEMENTS: usize = 1 << 17;
const CHUNK: usize = 4096;

pub fn narrow(input: &[f32]) -> Vec<u16> {
    let mut output = vec![0; input.len()];
    convert::narrow(input, &mut output);
    output
}
pub fn widen(input: &[u16]) -> Vec<f32> {
    let mut output = vec![0.; input.len()];
    convert::widen(input, &mut output);
    output
}
pub fn add(pool: &ThreadPool, output: &mut [u16], input: &[u16]) -> Result<()> {
    ensure!(input.len() == output.len(), "FP16 residual shape mismatch");
    let run = |output: &mut [u16], input: &[u16]| {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            convert::add(output, input);
        }
        #[cfg(not(target_arch = "aarch64"))]
        for (x, y) in output.iter_mut().zip(input) {
            *x = f16::from_f32(f16::from_bits(*x).to_f32() + f16::from_bits(*y).to_f32()).to_bits();
        }
    };
    if output.len() >= PARALLEL_ELEMENTS {
        pool.install(|| {
            output
                .par_chunks_mut(CHUNK)
                .zip(input.par_chunks(CHUNK))
                .for_each(|(a, b)| run(a, b))
        });
    } else {
        run(output, input);
    }
    Ok(())
}
#[inline]
fn gelu_value(x: f32, approximate: bool) -> f32 {
    if approximate {
        // Preserve the fused cubic used by the former GCC kernel.
        let cubic = (0.044715 * x * x).mul_add(x, x);
        0.5 * x * (1. + (0.797_884_6 * cubic).tanh())
    } else {
        0.5 * x * (1. + crate::ops::gelu::erf(x * std::f32::consts::FRAC_1_SQRT_2))
    }
}
pub fn gelu(pool: &ThreadPool, values: &mut [u16], approximate: bool) {
    let run = |values: &mut [u16]| {
        let mut blocks = values.chunks_exact_mut(8);
        for block in &mut blocks {
            let mut lanes = [0.; 8];
            convert::widen(block, &mut lanes);
            for x in &mut lanes {
                *x = gelu_value(*x, approximate);
            }
            convert::narrow(&lanes, block);
        }
        for x in blocks.into_remainder() {
            *x = f16::from_f32(gelu_value(f16::from_bits(*x).to_f32(), approximate)).to_bits();
        }
    };
    if values.len() >= PARALLEL_ELEMENTS {
        pool.install(|| values.par_chunks_mut(CHUNK).for_each(run));
    } else {
        run(values);
    }
}
pub fn layer_norm(
    pool: &ThreadPool,
    input: &[u16],
    output: &mut [u16],
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
) -> Result<()> {
    let width = weight.len();
    ensure!(
        width > 0
            && bias.len() == width
            && input.len() == output.len()
            && input.len().is_multiple_of(width)
            && epsilon.is_finite()
            && epsilon > 0.,
        "FP16 LayerNorm shape/epsilon mismatch"
    );
    if input.len() / width >= 8 {
        pool.install(|| {
            input
                .par_chunks(width)
                .zip(output.par_chunks_mut(width))
                .for_each(|(x, y)| norm::row(x, y, weight, bias, epsilon))
        });
    } else {
        for (x, y) in input.chunks(width).zip(output.chunks_mut(width)) {
            norm::row(x, y, weight, bias, epsilon);
        }
    }
    Ok(())
}
