//! Per-layer committed state and compact verification log. A shared Scratch can
//! serve successive layers; each layer retains its own log until commit/abort.
use anyhow::{Result, ensure};
use rayon::ThreadPool;
use serde::Serialize;

use super::mixer::{MixerWeights, MixerWorkspace, Projected, ValidatedProjected};
use crate::ops::{
    causal_conv::ConvState,
    gated_delta::{self, GdnDimensions, Updates},
};

#[derive(Default)]
struct UpdateLog {
    key: Vec<f32>,
    value: Vec<f32>,
    log_decay: Vec<f32>,
    beta: Vec<f32>,
    conv_input: Vec<f32>,
}

impl UpdateLog {
    fn reserve(&mut self, dim: GdnDimensions, rows: usize) {
        self.key.reserve(rows * dim.key_width());
        self.value.reserve(rows * dim.value_width());
        self.log_decay.reserve(rows * dim.value_heads);
        self.beta.reserve(rows * dim.value_heads);
        self.conv_input.reserve(rows * dim.conv_width());
    }
    fn record(&mut self, mixer: &MixerWorkspace, input: Projected<'_>) {
        self.key.clone_from(&mixer.key);
        self.value.clone_from(&mixer.value);
        self.log_decay.clone_from(&mixer.log_decay);
        self.beta.clone_from(&mixer.beta);
        self.conv_input.clear();
        self.conv_input.extend_from_slice(input.qkv);
    }
    fn updates(&self, rows: usize) -> Updates<'_> {
        Updates {
            rows,
            key: &self.key,
            value: &self.value,
            log_decay: &self.log_decay,
            beta: &self.beta,
        }
    }
    fn allocated_bytes(&self) -> usize {
        [
            &self.key,
            &self.value,
            &self.log_decay,
            &self.beta,
            &self.conv_input,
        ]
        .iter()
        .map(|v| v.capacity() * 4)
        .sum()
    }
}

#[derive(Default)]
pub struct Scratch {
    state: Vec<f32>,
    conv: Option<ConvState>,
    mixer: MixerWorkspace,
}

impl Scratch {
    pub fn allocated_bytes(&self) -> usize {
        self.state.capacity() * 4
            + self.conv.as_ref().map_or(0, ConvState::allocated_bytes)
            + self.mixer.allocated_bytes()
    }
}

pub struct LayerState {
    dim: GdnDimensions,
    kernel: usize,
    state: Vec<f32>,
    conv: ConvState,
    position: usize,
    max_verify_rows: usize,
    pending: Option<usize>,
    log: UpdateLog,
}

#[derive(Debug, Serialize)]
pub struct MemoryUsage {
    pub canonical_state_bytes: usize,
    pub convolution_history_bytes: usize,
    pub update_log_bytes: usize,
}

impl LayerState {
    pub fn new(dim: GdnDimensions, kernel: usize, max_verify_rows: usize) -> Result<Self> {
        ensure!(
            max_verify_rows > 0
                && max_verify_rows
                    .checked_mul(dim.conv_width())
                    .and_then(|v| v.checked_mul(4))
                    .is_some(),
            "invalid GDN verification capacity"
        );
        let mut log = UpdateLog::default();
        log.reserve(dim, max_verify_rows);
        Ok(Self {
            dim,
            kernel,
            state: vec![0.0; dim.state_elements()],
            conv: ConvState::new(dim.conv_width(), kernel)?,
            position: 0,
            max_verify_rows,
            pending: None,
            log,
        })
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn canonical(&self) -> &[f32] {
        &self.state
    }
    pub fn convolution(&self) -> &[f32] {
        self.conv.history()
    }
    pub fn pending_rows(&self) -> Option<usize> {
        self.pending
    }
    pub fn memory_usage(&self) -> MemoryUsage {
        MemoryUsage {
            canonical_state_bytes: self.state.capacity() * 4,
            convolution_history_bytes: self.conv.allocated_bytes(),
            update_log_bytes: self.log.allocated_bytes(),
        }
    }

    fn prepare<'a>(
        &self,
        weights: &MixerWeights,
        input: Projected<'a>,
        scratch: &mut Scratch,
    ) -> Result<ValidatedProjected<'a>> {
        ensure!(
            weights.dim == self.dim && weights.kernel == self.kernel,
            "GDN state/weight mismatch"
        );
        let checked = input.validate(self.dim)?;
        ensure!(
            self.position.checked_add(input.rows).is_some(),
            "GDN position overflow"
        );
        match scratch.conv.as_mut() {
            Some(conv) => conv.clone_from(&self.conv),
            None => scratch.conv = Some(self.conv.clone()),
        }
        Ok(checked)
    }

    fn compute<'a>(
        &self,
        pool: &ThreadPool,
        weights: &MixerWeights,
        input: Projected<'_>,
        scratch: &'a mut Scratch,
    ) -> Result<&'a [f32]> {
        let checked = self.prepare(weights, input, scratch)?;
        scratch.state.resize(self.state.len(), 0.0);
        scratch.mixer.run(
            pool,
            weights,
            checked,
            Some(&self.state),
            &mut scratch.state,
            scratch.conv.as_mut().unwrap(),
        )?;
        Ok(scratch.mixer.output())
    }

    /// Normal prefill/decode updates canonical state in place. Verification
    /// keeps the separate working state; ordinary forward retains no log.
    pub fn forward<'a>(
        &mut self,
        pool: &ThreadPool,
        weights: &MixerWeights,
        input: Projected<'_>,
        scratch: &'a mut Scratch,
    ) -> Result<&'a [f32]> {
        ensure!(
            self.pending.is_none(),
            "GDN verification transaction is active"
        );
        let checked = self.prepare(weights, input, scratch)?;
        scratch.mixer.run(
            pool,
            weights,
            checked,
            None,
            &mut self.state,
            scratch.conv.as_mut().unwrap(),
        )?;
        std::mem::swap(&mut self.conv, scratch.conv.as_mut().unwrap());
        self.position += input.rows;
        Ok(scratch.mixer.output())
    }

    /// Speculative validation leaves canonical state/history byte-for-byte intact.
    pub fn verify<'a>(
        &mut self,
        pool: &ThreadPool,
        weights: &MixerWeights,
        input: Projected<'_>,
        scratch: &'a mut Scratch,
    ) -> Result<&'a [f32]> {
        ensure!(
            self.pending.is_none(),
            "GDN verification transaction is active"
        );
        ensure!(
            input.rows <= self.max_verify_rows,
            "GDN verification exceeds reserved window"
        );
        self.compute(pool, weights, input, scratch)?;
        self.log.record(&scratch.mixer, input);
        self.pending = Some(input.rows);
        Ok(scratch.mixer.output())
    }

    /// Count consumed target inputs: anchor + accepted proposals. Zero acceptance
    /// still consumes one anchor; abort_round handles a cancelled transaction.
    pub fn finish_round(&mut self, pool: &ThreadPool, consumed_rows: usize) -> Result<()> {
        let rows = self
            .pending
            .context("no active GDN verification transaction")?;
        ensure!(
            (1..=rows).contains(&consumed_rows),
            "invalid GDN commit boundary"
        );
        let updates = self.log.updates(rows).prefix(self.dim, consumed_rows)?;
        gated_delta::replay(pool, self.dim, &mut self.state, updates)?;
        self.conv
            .append(&self.log.conv_input[..consumed_rows * self.dim.conv_width()])?;
        self.position += consumed_rows;
        self.pending = None;
        Ok(())
    }

    pub fn abort_round(&mut self) {
        self.pending = None;
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.conv.reset();
        self.position = 0;
        self.pending = None;
    }
}

use anyhow::Context;
