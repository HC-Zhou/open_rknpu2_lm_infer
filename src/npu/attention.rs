//! Native FP16 Attention cache, staging and execution.
use super::*;

pub struct Attention {
    pub(super) cache: Handle<rk_npu_attention_f16_cache>,
    pub(super) query_heads: usize,
    pub(super) kv_heads: usize,
    pub(super) dimension: usize,
    pub(super) owner: Backend,
}

fn pack_half(input: &[f32], output: &mut [u16]) -> Result<()> {
    let mut first = 0;
    #[cfg(target_arch = "aarch64")]
    while first + 8 <= input.len() {
        unsafe {
            core::arch::asm!(
                "ld1 {{v0.4s, v1.4s}}, [{source}]",
                "fcvtn v2.4h, v0.4s",
                "fcvtn2 v2.8h, v1.4s",
                "st1 {{v2.8h}}, [{target}]",
                source = in(reg) input.as_ptr().add(first),
                target = in(reg) output.as_mut_ptr().add(first),
                out("v0") _, out("v1") _, out("v2") _,
                options(nostack),
            );
        }
        first += 8;
    }
    for (&value, encoded) in input[first..].iter().zip(output[first..].iter_mut()) {
        *encoded = f16::from_f32(value).to_bits();
    }
    ensure!(
        output[..input.len()]
            .iter()
            .all(|&value| value & 0x7c00 != 0x7c00),
        "nonfinite/overflowed FP16 GQA input"
    );
    Ok(())
}
impl Attention {
    pub fn truncate(&self, length: usize) -> Result<()> {
        ensure!(
            length <= i32::MAX as usize,
            "invalid native KV prefix length"
        );
        check(
            unsafe { rk_npu_attention_f16_cache_truncate(self.cache.raw, length as i32) },
            "truncate native KV cache",
        )
    }
    pub fn reset(&self) -> Result<()> {
        let b = self.owner.0.borrow();
        check(
            unsafe {
                rk_npu_attention_f16_cache_load(
                    b.context.raw,
                    self.cache.raw,
                    ptr::null(),
                    ptr::null(),
                    0,
                )
            },
            "reset native KV cache",
        )
    }
    pub fn bytes(&self) -> Result<u64> {
        let mut info = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(self.cache.raw, &mut info) },
            "native KV memory",
        )?;
        Ok(info.key_bytes + info.value_bytes + info.tail_bytes)
    }
    pub fn reserve(&self, capacity: usize) -> Result<()> {
        let b = self.owner.0.borrow();
        ensure!(
            capacity <= b.options.context,
            "native GQA context capacity exceeded"
        );
        check(
            unsafe {
                rk_npu_attention_f16_cache_reserve(
                    b.context.raw,
                    self.cache.raw,
                    (capacity.div_ceil(32) * 32) as i32,
                )
            },
            "reserve native KV cache",
        )
    }
    pub fn run(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        output: &mut [f32],
        rows: usize,
        position: usize,
    ) -> Result<()> {
        let mut b = self.owner.0.borrow_mut();
        let (qh, kh, d) = (self.query_heads, self.kv_heads, self.dimension);
        ensure!(
            rows > 0 && rows <= b.options.chunk && position + rows <= b.options.context,
            "native GQA context capacity exceeded"
        );
        ensure!(
            q.len() >= rows * qh * d
                && output.len() >= rows * qh * d
                && k.len() >= rows * kh * d
                && v.len() >= rows * kh * d,
            "attention input/output shape mismatch"
        );
        let mut info = rk_npu_attention_f16_cache_info::default();
        check(
            unsafe { rk_npu_attention_f16_cache_get_info(self.cache.raw, &mut info) },
            "native KV position",
        )?;
        ensure!(
            info.length as usize == position,
            "native KV cache position mismatch"
        );
        let context = b.context.raw;
        let work = b.attention.as_mut().unwrap();
        for h in 0..kh {
            for r in 0..rows {
                let source = (r * kh + h) * d;
                let target = (h * rows + r) * d;
                pack_half(&k[source..source + d], &mut work.k[target..target + d])?;
                pack_half(&v[source..source + d], &mut work.v[target..target + d])?;
            }
        }
        check(
            unsafe {
                rk_npu_attention_f16_cache_append(
                    context,
                    self.cache.raw,
                    work.k.as_ptr(),
                    work.v.as_ptr(),
                    rows as i32,
                )
            },
            "append native KV",
        )?;
        let step = work.config.max_query_rows as usize;
        for offset in (0..rows).step_by(step) {
            let m = step.min(rows - offset);
            for h in 0..qh {
                for r in 0..m {
                    let source = ((offset + r) * qh + h) * d;
                    let target = (h * m + r) * d;
                    pack_half(&q[source..source + d], &mut work.q[target..target + d])?;
                }
            }
            check(
                unsafe {
                    rk_npu_attention_f16_run(
                        context,
                        work.workspace.raw,
                        self.cache.raw,
                        work.q.as_ptr(),
                        m as i32,
                        (position + offset) as i32,
                        work.output.as_mut_ptr(),
                        ptr::null_mut(),
                        ptr::null(),
                    )
                },
                "native GQA run",
            )?;
            for h in 0..qh {
                for r in 0..m {
                    let target = ((offset + r) * qh + h) * d;
                    let source = (h * m + r) * d;
                    output[target..target + d].copy_from_slice(&work.output[source..source + d]);
                }
            }
        }
        Ok(())
    }
}
