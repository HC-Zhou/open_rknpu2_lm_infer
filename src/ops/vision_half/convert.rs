use half::f16;

#[cfg(target_arch = "aarch64")]
use std::arch::{aarch64::*, asm};

// Callers provide eight in-bounds elements. FP16 storage conversion is available
// on baseline AArch64; no unstable Rust FP16 arithmetic types are needed.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn load8(input: *const u16) -> (float32x4_t, float32x4_t) {
    unsafe {
        let bits = vld1q_u16(input);
        let low;
        let high;
        asm!(
            "fcvtl {low:v}.4s, {bits:v}.4h",
            "fcvtl2 {high:v}.4s, {bits:v}.8h",
            bits = in(vreg) bits, low = out(vreg) low, high = out(vreg) high,
            options(nomem, nostack, pure),
        );
        (low, high)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
pub(super) unsafe fn store8(output: *mut u16, low: float32x4_t, high: float32x4_t) {
    unsafe {
        let bits: uint16x8_t;
        asm!(
            "fcvtn {bits:v}.4h, {low:v}.4s",
            "fcvtn2 {bits:v}.8h, {high:v}.4s",
            bits = out(vreg) bits, low = in(vreg) low, high = in(vreg) high,
            options(nomem, nostack, pure),
        );
        vst1q_u16(output, bits);
    }
}
// RK3588 provides FP16 vector arithmetic, as used by the former CPU kernel.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
pub(super) unsafe fn add(output: &mut [u16], input: &[u16]) {
    unsafe {
        let mut first = 0;
        while first + 8 <= output.len() {
            let a = vld1q_u16(output.as_ptr().add(first));
            let b = vld1q_u16(input.as_ptr().add(first));
            let sum: uint16x8_t;
            asm!(
                "fadd {sum:v}.8h, {a:v}.8h, {b:v}.8h",
                a = in(vreg) a, b = in(vreg) b, sum = out(vreg) sum,
                options(nomem, nostack, pure),
            );
            vst1q_u16(output.as_mut_ptr().add(first), sum);
            first += 8;
        }
        for (x, y) in output[first..].iter_mut().zip(&input[first..]) {
            *x = f16::from_f32(f16::from_bits(*x).to_f32() + f16::from_bits(*y).to_f32()).to_bits();
        }
    }
}
#[inline]
pub(super) fn narrow(input: &[f32], output: &mut [u16]) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    let mut first = 0;
    #[cfg(not(target_arch = "aarch64"))]
    let first = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        while first + 8 <= input.len() {
            store8(
                output.as_mut_ptr().add(first),
                vld1q_f32(input.as_ptr().add(first)),
                vld1q_f32(input.as_ptr().add(first + 4)),
            );
            first += 8;
        }
    }
    for (x, y) in input[first..].iter().zip(&mut output[first..]) {
        *y = f16::from_f32(*x).to_bits();
    }
}
#[inline]
pub(super) fn widen(input: &[u16], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    #[cfg(target_arch = "aarch64")]
    let mut first = 0;
    #[cfg(not(target_arch = "aarch64"))]
    let first = 0;
    #[cfg(target_arch = "aarch64")]
    unsafe {
        while first + 8 <= input.len() {
            let (low, high) = load8(input.as_ptr().add(first));
            vst1q_f32(output.as_mut_ptr().add(first), low);
            vst1q_f32(output.as_mut_ptr().add(first + 4), high);
            first += 8;
        }
    }
    for (x, y) in input[first..].iter().zip(&mut output[first..]) {
        *y = f16::from_bits(*x).to_f32();
    }
}
