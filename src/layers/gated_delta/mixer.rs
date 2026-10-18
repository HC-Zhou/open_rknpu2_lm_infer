//! Linear-attention mixer between input/output projections; shared CPU/NPU state.
use anyhow::{Result, ensure};
use rayon::ThreadPool;
use std::time::Instant;

use crate::ops::{
    activation,
    causal_conv::ConvState,
    gated_delta::{GdnDimensions, Updates, Workspace},
    norm,
};

pub struct MixerWeights {
    pub(super) dim: GdnDimensions,
    pub(super) kernel: usize,
    conv: Vec<f32>,
    decay_scale: Vec<f32>,
    dt_bias: Vec<f32>,
    norm: Vec<f32>,
    epsilon: f32,
}

impl MixerWeights {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: GdnDimensions,
        kernel: usize,
        conv: Vec<f32>,
        a_log: Vec<f32>,
        dt_bias: Vec<f32>,
        norm: Vec<f32>,
        epsilon: f32,
    ) -> Result<Self> {
        ensure!(
            kernel > 0
                && dim.conv_width().checked_mul(kernel) == Some(conv.len())
                && a_log.len() == dim.value_heads
                && dt_bias.len() == dim.value_heads
                && norm.len() == dim.value_dim,
            "GDN mixer weight shape mismatch"
        );
        ensure!(
            conv.iter()
                .chain(&a_log)
                .chain(&dt_bias)
                .chain(&norm)
                .all(|v| v.is_finite())
                && epsilon.is_finite()
                && epsilon > 0.0,
            "invalid GDN mixer weights"
        );
        Ok(Self {
            dim,
            kernel,
            conv,
            decay_scale: a_log.into_iter().map(f32::exp).collect(),
            dt_bias,
            norm,
            epsilon,
        })
    }
}

/// Token-major projected QKV/Z/a/b, before any activation. Constructors accept
/// separate contiguous tensors or views into a merged projection.
#[derive(Clone, Copy)]
pub struct Projected<'a> {
    pub rows: usize,
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
    stride: usize,
}

/// The layer checked projection shapes before touching canonical state.
pub(super) struct ValidatedProjected<'a>(Projected<'a>);

impl<'a> Projected<'a> {
    pub fn new(rows: usize, qkv: &'a [f32], z: &'a [f32], a: &'a [f32], b: &'a [f32]) -> Self {
        Self {
            rows,
            qkv,
            z,
            a,
            b,
            stride: 0,
        }
    }

    /// Token-major concatenated QKV/Z/a/b, read directly from the projection.
    pub fn from_merged(dim: GdnDimensions, rows: usize, values: &'a [f32]) -> Result<Self> {
        let (cw, vw, hv) = (dim.conv_width(), dim.value_width(), dim.value_heads());
        let stride = cw + vw + 2 * hv;
        ensure!(
            rows > 0 && rows.checked_mul(stride) == Some(values.len()),
            "GDN merged projection shape mismatch"
        );
        let span = |offset, width| &values[offset..(rows - 1) * stride + offset + width];
        Ok(Self {
            rows,
            qkv: span(0, cw),
            z: span(cw, vw),
            a: span(cw + vw, hv),
            b: span(cw + vw + hv, hv),
            stride,
        })
    }

    pub(super) fn row_stride(self, width: usize) -> usize {
        if self.stride == 0 { width } else { self.stride }
    }

    pub(super) fn validate(self, dim: GdnDimensions) -> Result<ValidatedProjected<'a>> {
        let span = |width| {
            self.rows
                .checked_sub(1)
                .and_then(|n| n.checked_mul(self.row_stride(width)))
                .and_then(|n| n.checked_add(width))
        };
        ensure!(
            self.rows > 0
                && span(dim.conv_width()) == Some(self.qkv.len())
                && span(dim.value_width()) == Some(self.z.len())
                && span(dim.value_heads) == Some(self.a.len())
                && self.a.len() == self.b.len(),
            "GDN projected input shape mismatch"
        );
        Ok(ValidatedProjected(self))
    }
}

#[derive(Default)]
pub struct MixerWorkspace {
    convolved: Vec<f32>,
    query: Vec<f32>,
    pub(super) key: Vec<f32>,
    pub(super) value: Vec<f32>,
    pub(super) log_decay: Vec<f32>,
    pub(super) beta: Vec<f32>,
    core: Vec<f32>,
    output: Vec<f32>,
    recurrence: Workspace,
    pub(super) buffers_ms: f64,
    pub(super) gates_ms: f64,
    pub(super) output_norm_ms: f64,
}

impl MixerWorkspace {
    pub fn output(&self) -> &[f32] {
        &self.output
    }
    pub fn allocated_bytes(&self) -> usize {
        [
            &self.convolved,
            &self.query,
            &self.key,
            &self.value,
            &self.log_decay,
            &self.beta,
            &self.core,
            &self.output,
        ]
        .iter()
        .map(|v| v.capacity() * 4)
        .sum::<usize>()
            + self.recurrence.allocated_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &mut self,
        pool: &ThreadPool,
        weights: &MixerWeights,
        input: ValidatedProjected<'_>,
        initial_state: Option<&[f32]>,
        state: &mut [f32],
        conv: &mut ConvState,
        prefill: Option<&mut crate::npu::GatedDelta>,
        output: Option<&mut [f32]>,
    ) -> Result<()> {
        let dim = weights.dim;
        let input = input.0;
        let start = Instant::now();
        let elements = input.rows * dim.value_width();
        ensure!(
            output.as_ref().is_none_or(|out| out.len() == elements),
            "GDN mixer output shape mismatch"
        );
        self.log_decay.resize(input.rows * dim.value_heads, 0.0);
        self.beta.resize(self.log_decay.len(), 0.0);
        self.core.resize(input.rows * dim.value_width(), 0.0);
        if output.is_none() {
            self.output.resize(self.core.len(), 0.0);
        }
        self.buffers_ms += start.elapsed().as_secs_f64() * 1000.;
        let start = Instant::now();
        for i in 0..self.log_decay.len() {
            let h = i % dim.value_heads;
            let source = (i / dim.value_heads) * input.row_stride(dim.value_heads) + h;
            self.log_decay[i] = -weights.decay_scale[h]
                * activation::softplus(input.a[source] + weights.dt_bias[h]);
            self.beta[i] = activation::sigmoid(input.b[source]);
        }
        self.gates_ms += start.elapsed().as_secs_f64() * 1000.;
        if let Some(prefill) = prefill {
            prefill.run_projected(
                input.rows,
                input.qkv,
                input.row_stride(dim.conv_width()),
                &weights.conv,
                conv.history_mut(),
                &self.log_decay,
                &self.beta,
                state,
                &mut self.core,
            )?;
        } else {
            self.convolved.resize(input.rows * dim.conv_width(), 0.0);
            conv.run_parallel_strided(
                pool,
                input.qkv,
                input.row_stride(dim.conv_width()),
                &weights.conv,
                &mut self.convolved,
            )?;
            self.query.resize(input.rows * dim.key_width(), 0.0);
            self.key.resize(self.query.len(), 0.0);
            self.value.resize(input.rows * dim.value_width(), 0.0);
            for (t, row) in self.convolved.chunks(dim.conv_width()).enumerate() {
                self.query[t * dim.key_width()..(t + 1) * dim.key_width()]
                    .copy_from_slice(&row[..dim.key_width()]);
                self.key[t * dim.key_width()..(t + 1) * dim.key_width()]
                    .copy_from_slice(&row[dim.key_width()..2 * dim.key_width()]);
                self.value[t * dim.value_width()..(t + 1) * dim.value_width()]
                    .copy_from_slice(&row[2 * dim.key_width()..]);
            }
            norm::l2_normalize(&mut self.query, dim.key_dim)?;
            norm::l2_normalize(&mut self.key, dim.key_dim)?;
            let updates = Updates {
                rows: input.rows,
                key: &self.key,
                value: &self.value,
                log_decay: &self.log_decay,
                beta: &self.beta,
            };
            // Shape and gate checks finish before the native call mutates state.
            // Output norm's shapes/epsilon follow from MixerWeights and these buffers.
            match initial_state {
                Some(initial) => self.recurrence.run_from(
                    pool,
                    dim,
                    initial,
                    state,
                    &self.query,
                    updates,
                    &mut self.core,
                )?,
                None => {
                    self.recurrence
                        .run(pool, dim, state, &self.query, updates, &mut self.core)?
                }
            }
        }
        let start = Instant::now();
        norm::gated_norm_parallel_strided(
            pool,
            &self.core,
            input.z,
            input.row_stride(dim.value_width()),
            dim.value_width(),
            &weights.norm,
            weights.epsilon,
            output.unwrap_or(&mut self.output),
        )?;
        self.output_norm_ms += start.elapsed().as_secs_f64() * 1000.;
        Ok(())
    }
}
