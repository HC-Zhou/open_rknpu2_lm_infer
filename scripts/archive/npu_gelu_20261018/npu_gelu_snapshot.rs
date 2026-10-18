//! Diagnostic override used only in the copied ViT experiment crate.
#[path = "../../../tests/tools/npu_gelu_bench/bridge.rs"]
mod bridge;
use bridge::Bridge;
use std::{cell::RefCell, path::Path};

thread_local! {
    static KERNEL: RefCell<Option<(usize, Bridge)>> = const { RefCell::new(None) };
}

pub(super) fn apply(values: &mut [u16]) {
    KERNEL.with(|kernel| {
        let mut current = kernel.borrow_mut();
        if current
            .as_ref()
            .is_none_or(|(count, _)| *count != values.len())
        {
            *current = None;
            let bridge = Bridge::new(
                Path::new("logs/npu-gelu-bridge/libnpu_gelu_parallel.so"),
                values.len(),
                7,
            )
            .expect("prepare experimental NPU GELU");
            *current = Some((values.len(), bridge));
        }
        let bridge = &current.as_ref().unwrap().1;
        bridge
            .apply(values, &mut [0.; 6])
            .expect("experimental NPU GELU");
        assert!(bridge.guard_ok(), "experimental GELU guard changed");
    });
}
