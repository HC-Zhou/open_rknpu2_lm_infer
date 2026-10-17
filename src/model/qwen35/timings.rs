//! Synchronous stage totals; fused Linear finish remains part of its call.
use serde::Serialize;
use std::time::Instant;

#[derive(Default, Serialize)]
pub(super) struct Timings {
    linear_ms: f64,
    gdn_ms: f64,
    attention_ms: f64,
    cpu_ms: f64,
}

pub(super) enum Phase {
    Linear,
    Gdn,
    Attention,
    Cpu,
}

impl Timings {
    pub fn measure<T>(&mut self, phase: Phase, operation: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = operation();
        let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
        *match phase {
            Phase::Linear => &mut self.linear_ms,
            Phase::Gdn => &mut self.gdn_ms,
            Phase::Attention => &mut self.attention_ms,
            Phase::Cpu => &mut self.cpu_ms,
        } += milliseconds;
        result
    }
}
