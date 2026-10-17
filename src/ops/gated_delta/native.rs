use super::{GdnDimensions, Updates};
use rayon::ThreadPool;

#[repr(C)]
struct Parameters {
    rows: usize,
    key_heads: usize,
    value_heads: usize,
    key_dim: usize,
    value_dim: usize,
    threads: usize,
    initial_state: *const f32,
    state: *mut f32,
    query: *const f32,
    key: *const f32,
    value: *const f32,
    log_decay: *const f32,
    beta: *const f32,
    output: *mut f32,
}

unsafe extern "C" {
    fn rknpu2_gdn_run(params: *const Parameters);
    fn rknpu2_gdn_run_unfused(params: *const Parameters);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    pool: &ThreadPool,
    dim: GdnDimensions,
    initial: Option<&[f32]>,
    state: &mut [f32],
    query: Option<&[f32]>,
    updates: Updates<'_>,
    output: Option<&mut [f32]>,
    unfused_reference: bool,
) {
    if updates.rows == 0 {
        return;
    }
    let params = Parameters {
        rows: updates.rows,
        key_heads: dim.key_heads,
        value_heads: dim.value_heads,
        key_dim: dim.key_dim,
        value_dim: dim.value_dim,
        threads: pool.current_num_threads(),
        initial_state: initial.map_or(state.as_ptr(), |s| s.as_ptr()),
        state: state.as_mut_ptr(),
        query: query.map_or(std::ptr::null(), |q| q.as_ptr()),
        key: updates.key.as_ptr(),
        value: updates.value.as_ptr(),
        log_decay: updates.log_decay.as_ptr(),
        beta: updates.beta.as_ptr(),
        output: output.map_or(std::ptr::null_mut(), |o| o.as_mut_ptr()),
    };
    // The safe caller checked all products/lengths. The C++ call is synchronous;
    // each worker exclusively owns one head's state and output channel range.
    unsafe {
        if unfused_reference {
            rknpu2_gdn_run_unfused(&params);
        } else {
            rknpu2_gdn_run(&params);
        }
    }
}
