//! Diagnostic reporting, included only by the copied research crate.
use serde_json::json;
use std::{ffi::c_void,sync::OnceLock};
#[repr(C)]
#[derive(Default)]
struct State {
    lut:i32, cache_lut:i32, can_lut:i32, mask_mode:i32, flags:i32,
    query_rows:i32, physical_rows:i32, span:i32, blocks:i32, matmul_tasks:i32,
    tasks:i32, heads:i32, head_dim:i32, kv_tile:i32, core_mask:i32, lut_split:i32,
    stage_tasks:[i32;3], value_max:f32, lut_value_limit:f32,
}
type Read=unsafe extern "C" fn(*const c_void,*const c_void,*mut State)->i32;
static READ:OnceLock<Read>=OnceLock::new();
pub(super) fn record(cache:*const c_void,workspace:*const c_void,
    t:&super::ffi::rk_npu_attention_f16_timings,rows:usize,offset:usize) {
    let read=READ.get_or_init(|| unsafe {
        // Keep the diagnostic library loaded for the function pointer lifetime.
        let lib=libc::dlopen(c"logs/npu-gelu-bridge/libvision_attention_state.so".as_ptr(),libc::RTLD_NOW|libc::RTLD_LOCAL);
        assert!(!lib.is_null());
        let p=libc::dlsym(lib,c"vision_attention_state".as_ptr());assert!(!p.is_null());
        std::mem::transmute::<*mut c_void,Read>(p)
    });
    let mut s=State::default();assert_eq!(unsafe{read(cache,workspace,&mut s)},0);
    eprintln!("ATTN_STATE {}",json!({"image_rows":rows,"offset":offset,
        "lut":s.lut,"cache_lut":s.cache_lut,"can_lut":s.can_lut,
        "mask_mode":s.mask_mode,"flags":s.flags,"query_rows":s.query_rows,
        "physical_rows":s.physical_rows,"span":s.span,"blocks":s.blocks,
        "matmul_tasks":s.matmul_tasks,"tasks":s.tasks,"stage_tasks":s.stage_tasks,
        "heads":s.heads,"head_dim":s.head_dim,"kv_tile":s.kv_tile,
        "core_mask":s.core_mask,"lut_split":s.lut_split,
        "value_max":s.value_max,"lut_value_limit":s.lut_value_limit,
        "timings":{"prepare_us":t.prepare_us,"tail_stage_us":t.tail_stage_us,
        "query_pack_us":t.query_pack_us,"submit_us":t.submit_us,"hardware_us":t.hardware_us,
        "output_sync_us":t.output_sync_us,"finish_us":t.finish_us,"total_us":t.total_us,
        "tasks":t.tasks,"mask_update_us":t.mask_update_us}}));
}
