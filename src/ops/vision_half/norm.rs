#[cfg(target_arch = "aarch64")]
use super::convert::{load8, store8};
use half::f16;

pub(super) fn row(input: &[u16], output: &mut [u16], weight: &[f32], bias: &[f32], epsilon: f32) {
    let width = input.len();
    #[cfg(target_arch = "aarch64")]
    let mut total;
    #[cfg(not(target_arch = "aarch64"))]
    let mut total = 0.;
    let mut first = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut sums = vdupq_n_f32(0.);
        while first + 8 <= width {
            let (low, high) = load8(input.as_ptr().add(first));
            sums = vaddq_f32(vaddq_f32(sums, low), high);
            first += 8;
        }
        total = vaddvq_f32(sums);
    }
    for x in &input[first..] {
        total += f16::from_bits(*x).to_f32();
    }
    let mean = total / width as f32;
    #[cfg(target_arch = "aarch64")]
    let mut variance;
    #[cfg(not(target_arch = "aarch64"))]
    let mut variance = 0.;
    first = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let mut sums = vdupq_n_f32(0.);
        let means = vdupq_n_f32(mean);
        while first + 8 <= width {
            let (low, high) = load8(input.as_ptr().add(first));
            let a = vsubq_f32(low, means);
            let b = vsubq_f32(high, means);
            sums = vfmaq_f32(vfmaq_f32(sums, a, a), b, b);
            first += 8;
        }
        variance = vaddvq_f32(sums);
    }
    for x in &input[first..] {
        let x = f16::from_bits(*x).to_f32() - mean;
        variance = x.mul_add(x, variance);
    }
    let scale = 1. / (variance / width as f32 + epsilon).sqrt();
    first = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let means = vdupq_n_f32(mean);
        while first + 8 <= width {
            let (low, high) = load8(input.as_ptr().add(first));
            let a = vmulq_n_f32(vsubq_f32(low, means), scale);
            let b = vmulq_n_f32(vsubq_f32(high, means), scale);
            let low = vfmaq_f32(
                vld1q_f32(bias.as_ptr().add(first)),
                a,
                vld1q_f32(weight.as_ptr().add(first)),
            );
            let high = vfmaq_f32(
                vld1q_f32(bias.as_ptr().add(first + 4)),
                b,
                vld1q_f32(weight.as_ptr().add(first + 4)),
            );
            store8(output.as_mut_ptr().add(first), low, high);
            first += 8;
        }
    }
    for j in first..width {
        let value = (f16::from_bits(input[j]).to_f32() - mean) * scale;
        output[j] = f16::from_f32(value.mul_add(weight[j], bias[j])).to_bits();
    }
}
