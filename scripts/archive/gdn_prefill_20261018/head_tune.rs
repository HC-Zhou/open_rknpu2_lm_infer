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
use std::{ffi::CString, ptr};
fn main() {
    unsafe {
        let ctx = rk_npu_open(ptr::null());
        assert!(!ctx.is_null());
        for kt in [2048, 2560] {
            let mut cfg = rk_npu_matmul_autotune_config::default();
            rk_npu_matmul_autotune_config_init(&mut cfg, 256, 248320, 2560);
            cfg.required_k_tile = kt;
            cfg.allowed_npu_core_mask = 7;
            cfg.allowed_cpu_core_mask = 0xf0;
            cfg.warmup = 1;
            cfg.loops = 3;
            cfg.repeats = 3;
            cfg.verbose = 1;
            let path = CString::new(format!(
                ".cache/qwen35-gdn-prefill/w8-m256-k2560-n248320-kt{kt}.tune"
            ))
            .unwrap();
            let mut fastest = rk_npu_matmul_strategy::default();
            let mut stable = rk_npu_matmul_strategy::default();
            let mut hit = 0;
            assert_eq!(
                rk_npu_matmul_autotune_cached(
                    ctx,
                    &cfg,
                    RK_NPU_MATMUL_F32I8F32_DYNAMIC,
                    path.as_ptr(),
                    0,
                    &mut fastest,
                    &mut stable,
                    &mut hit
                ),
                0
            );
            println!(
                "head M256 kt={kt} total_us={} cache_hit={hit}",
                fastest.total_us
            );
        }
        rk_npu_close(ctx);
    }
}
