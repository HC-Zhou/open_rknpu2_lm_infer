//! NEON SiLU adapted from llama.cpp ggml-cpu/vec.h, MIT licensed.
//! Pin and attribution: llm_generated_docs/THIRD_PARTY_NOTICES.md.

use anyhow::{Result, ensure};

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

pub fn sigmoid_gate(output: &mut [f32], gate: &[f32]) -> Result<()> {
    ensure!(output.len() == gate.len(), "sigmoid gate shape mismatch");
    for (out, gate) in output.iter_mut().zip(gate) {
        *out *= sigmoid(*gate);
    }
    Ok(())
}

pub fn silu_in_place(values: &mut [f32]) {
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        while i + 4 <= values.len() {
            let x = vld1q_f32(values.as_ptr().add(i));
            // Retain the stable scalar behavior for extreme values and NaNs;
            // the common range follows ggml's vector exp polynomial exactly.
            if vminvq_u32(vcleq_f32(vabsq_f32(x), vdupq_n_f32(80.0))) == u32::MAX {
                let y = vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp(vnegq_f32(x))));
                vst1q_f32(values.as_mut_ptr().add(i), y);
            } else {
                for value in &mut values[i..i + 4] {
                    *value = silu(*value);
                }
            }
            i += 4;
        }
    }
    for value in &mut values[i..] {
        *value = silu(*value);
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn exp(x: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // Rust has no hex float literals: these are the exact f32 bit patterns of
    // the pinned ggml_v_expf coefficients. The caller bounds x to [-80, 80].
    unsafe {
        let r = vdupq_n_f32(f32::from_bits(0x4b400000));
        let z = vfmaq_n_f32(r, x, f32::from_bits(0x3fb8aa3b));
        let n = vsubq_f32(z, r);
        let b = vfmsq_n_f32(
            vfmsq_n_f32(x, n, f32::from_bits(0x3f317200)),
            n,
            f32::from_bits(0x35bfbe8e),
        );
        let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
        let k = vreinterpretq_f32_u32(vaddq_u32(e, vdupq_n_u32(1.0f32.to_bits())));
        let u = vmulq_f32(b, b);
        let j = vfmaq_f32(
            vmulq_n_f32(b, f32::from_bits(0x3f7ffff6)),
            vfmaq_f32(
                vfmaq_f32(
                    vdupq_n_f32(f32::from_bits(0x3efffedb)),
                    vdupq_n_f32(f32::from_bits(0x3e2aaf33)),
                    b,
                ),
                vfmaq_f32(
                    vdupq_n_f32(f32::from_bits(0x3d2b9f17)),
                    vdupq_n_f32(f32::from_bits(0x3c072010)),
                    b,
                ),
                u,
            ),
            u,
        );
        vfmaq_f32(k, j, k)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn vector_silu_matches_stable_scalar_including_tails_and_extremes() {
        let mut values: Vec<f32> = (-4000..4001).map(|i| i as f32 / 50.0).collect();
        values.extend([-1000.0, 1000.0, -0.0, 0.0, -100.0]);
        let expected: Vec<f32> = values.iter().map(|x| super::silu(*x)).collect();
        super::silu_in_place(&mut values);
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() <= 4e-7 * expected.abs().max(1.0));
        }
    }
}
