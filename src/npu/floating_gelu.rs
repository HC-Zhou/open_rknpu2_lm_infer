//! Collect the mapped FP32 panel8/N4 GELU carrier directly into activations.
use super::*;
use rayon::{ThreadPool, prelude::*};

trait Output: Copy + Send {
    fn scalar(x: f32) -> Self;
    unsafe fn block(raw: *const f32, dst: *mut Self, scale: f32);
}
impl Output for u16 {
    fn scalar(x: f32) -> Self {
        f16::from_f32(x).to_bits()
    }
    unsafe fn block(raw: *const f32, dst: *mut Self, scale: f32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let lo = vmulq_n_f32(vld1q_f32(raw), scale);
            let hi = vmulq_n_f32(vld1q_f32(raw.add(32)), scale);
            vst1q_u16(
                dst,
                vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi))),
            );
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            for j in 0..8 {
                *dst.add(j) = Self::scalar(*raw.add(j / 4 * 32 + j % 4) * scale);
            }
        }
    }
}
impl Output for f32 {
    fn scalar(x: f32) -> Self {
        x
    }
    unsafe fn block(raw: *const f32, dst: *mut Self, scale: f32) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            vst1q_f32(dst, vmulq_n_f32(vld1q_f32(raw), scale));
            vst1q_f32(dst.add(4), vmulq_n_f32(vld1q_f32(raw.add(32)), scale));
        }
        #[cfg(not(target_arch = "aarch64"))]
        unsafe {
            for j in 0..8 {
                *dst.add(j) = *raw.add(j / 4 * 32 + j % 4) * scale;
            }
        }
    }
}
fn collect<T: Output>(
    pool: &ThreadPool,
    view: &rk_npu_mm_gelu_view,
    output: &mut [T],
) -> Result<()> {
    let (m, n, sm, sn) = (
        view.rows as usize,
        view.columns as usize,
        view.storage_rows as usize,
        view.storage_columns as usize,
    );
    ensure!(
        m > 0
            && n > 0
            && sm >= m
            && sn >= n
            && sm.is_multiple_of(8)
            && sn.is_multiple_of(32)
            && sm.checked_mul(sn) == Some(view.elements as usize)
            && m.checked_mul(n) == Some(output.len())
            && !view.data.is_null(),
        "invalid fused GELU view/output shape"
    );
    // Only called while the native workspace is borrowed; sync FROM_DEVICE
    // completed in run. The view covers padded rows/channels for vector loads.
    let raw = unsafe { std::slice::from_raw_parts(view.data, view.elements as usize) };
    let scale = view.scale;
    pool.install(|| {
        output
            .par_chunks_mut(n * 8)
            .enumerate()
            .for_each(|(panel, dst)| {
                let height = dst.len() / n;
                for col in (0..n / 8 * 8).step_by(8) {
                    for row in 0..height {
                        let offset = panel * 8 * sn + (col / 4 * 8 + row) * 4;
                        unsafe {
                            T::block(
                                raw.as_ptr().add(offset),
                                dst.as_mut_ptr().add(row * n + col),
                                scale,
                            );
                        }
                    }
                }
                for col in n / 8 * 8..n {
                    for row in 0..height {
                        let offset = panel * 8 * sn + (col / 4 * 8 + row) * 4 + col % 4;
                        dst[row * n + col] = T::scalar(raw[offset] * scale);
                    }
                }
            })
    });
    Ok(())
}
pub(super) fn collect_half(
    pool: &ThreadPool,
    view: &rk_npu_mm_gelu_view,
    output: &mut [u16],
) -> Result<()> {
    collect(pool, view, output)
}
pub(super) fn collect_float(
    pool: &ThreadPool,
    view: &rk_npu_mm_gelu_view,
    output: &mut [f32],
) -> Result<()> {
    collect(pool, view, output)
}
