//! Synchronous stage totals; fused Linear finish remains part of its call.
use serde::Serialize;
use std::time::Instant;

#[derive(Default, Serialize)]
pub(super) struct Timings {
    linear_ms: f64,
    gdn_ms: f64,
    attention_ms: f64,
    cpu_ms: f64,
    gdn_split_ms: f64,
    gdn_output_copy_ms: f64,
    attention_verify_calls: usize,
    attention_verify_rows: usize,
}

pub(super) enum Phase {
    Linear,
    Gdn,
    Attention,
    Cpu,
}

impl Timings {
    pub fn record_attention_verification(&mut self, rows: usize) {
        self.attention_verify_calls += 1;
        self.attention_verify_rows += rows;
    }
    pub fn record_gdn_io(&mut self, split_ms: f64, output_copy_ms: f64) {
        self.gdn_split_ms += split_ms;
        self.gdn_output_copy_ms += output_copy_ms;
    }
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
