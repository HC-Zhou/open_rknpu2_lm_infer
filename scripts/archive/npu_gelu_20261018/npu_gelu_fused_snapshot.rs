//! Experimental fused UP. Only the copied research crate includes this module.
use anyhow::{Result, ensure};
use rayon::prelude::*;
use std::ffi::{CStr, c_char, c_void};

type Create = unsafe extern "C" fn(i32, i32, *const u16, *const f32) -> *mut c_void;
type Free = unsafe extern "C" fn(*mut c_void);
type Run = unsafe extern "C" fn(*mut c_void, i32, *const u16, *mut f64) -> *const f32;
type Guard = unsafe extern "C" fn(*mut c_void) -> i32;
type Error = unsafe extern "C" fn() -> *const c_char;
pub(super) struct Fused {
    library: *mut c_void,
    handle: *mut c_void,
    n: usize,
    free: Free,
    run: Run,
    guard: Guard,
    error: Error,
}
unsafe fn symbol<T: Copy>(library: *mut c_void, name: &CStr) -> Result<T> {
    let p = unsafe { libc::dlsym(library, name.as_ptr()) };
    ensure!(!p.is_null(), "missing fused symbol {name:?}");
    Ok(unsafe { std::mem::transmute_copy(&p) })
}
impl Fused {
    pub(super) fn new(n: usize, k: usize, weight: &[u16], bias: &[f32]) -> Result<Self> {
        let library = unsafe {
            libc::dlopen(
                c"logs/npu-gelu-bridge/libnpu_gelu_fused.so".as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL,
            )
        };
        ensure!(!library.is_null(), "load fused GELU bridge");
        unsafe {
            let create: Create = symbol(library, c"npu_gelu_fused_create")?;
            let free = symbol(library, c"npu_gelu_fused_free")?;
            let run = symbol(library, c"npu_gelu_fused_run")?;
            let guard = symbol(library, c"npu_gelu_fused_guard")?;
            let error: Error = symbol(library, c"npu_gelu_fused_error")?;
            let handle = create(n as i32, k as i32, weight.as_ptr(), bias.as_ptr());
            ensure!(
                !handle.is_null(),
                "{}",
                CStr::from_ptr(error()).to_string_lossy()
            );
            Ok(Self {
                library,
                handle,
                n,
                free,
                run,
                guard,
                error,
            })
        }
    }
    pub(super) fn apply(
        &self,
        pool: &rayon::ThreadPool,
        input: &[u16],
        output: &mut [u16],
        rows: usize,
    ) -> Result<()> {
        let p = unsafe {
            (self.run)(
                self.handle,
                rows as i32,
                input.as_ptr(),
                [0.; 5].as_mut_ptr(),
            )
        };
        ensure!(
            !p.is_null(),
            "{}",
            unsafe { CStr::from_ptr((self.error)()) }.to_string_lossy()
        );
        let raw = unsafe { std::slice::from_raw_parts(p, rows * self.n) };
        let n = self.n;
        // Device FP32 carrier is panel8/N4. Collect and narrow directly into
        // the normal FP16 activation; there is no full FP32 host temporary.
        pool.install(|| {
            output
                .par_chunks_mut(n * 8)
                .enumerate()
                .for_each(|(panel, dst)| {
                    for col in (0..n).step_by(8) {
                        for row in 0..8 {
                            let offset = panel * 8 * n + (col / 4 * 8 + row) * 4;
                            #[cfg(target_arch = "aarch64")]
                            unsafe {
                                use std::arch::aarch64::*;
                                let lo =
                                    vmulq_n_f32(vld1q_f32(raw.as_ptr().add(offset)), 1. / 8192.);
                                let hi = vmulq_n_f32(
                                    vld1q_f32(raw.as_ptr().add(offset + 32)),
                                    1. / 8192.,
                                );
                                vst1q_u16(
                                    dst.as_mut_ptr().add(row * n + col),
                                    vreinterpretq_u16_f16(vcombine_f16(
                                        vcvt_f16_f32(lo),
                                        vcvt_f16_f32(hi),
                                    )),
                                );
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            for j in 0..8 {
                                let at = offset + j / 4 * 32 + j % 4;
                                dst[row * n + col + j] =
                                    half::f16::from_f32(raw[at] / 8192.).to_bits();
                            }
                        }
                    }
                })
        });
        ensure!(
            unsafe { (self.guard)(self.handle) } == 1,
            "fused GELU output guard"
        );
        Ok(())
    }
}
impl Drop for Fused {
    fn drop(&mut self) {
        unsafe {
            (self.free)(self.handle);
            libc::dlclose(self.library);
        }
    }
}
