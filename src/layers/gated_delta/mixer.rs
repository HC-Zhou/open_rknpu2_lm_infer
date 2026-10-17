//! CPU portion of a linear-attention layer, between input/output projections.
use anyhow::{Result, ensure};
use rayon::ThreadPool;

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
    a_log: Vec<f32>,
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
            a_log,
            dt_bias,
            norm,
            epsilon,
        })
    }
}

/// All slices are token-major; qkv is [Q heads, K heads, V heads].
/// z/a/b come from their separate projections, before any activation.
#[derive(Clone, Copy)]
pub struct Projected<'a> {
    pub rows: usize,
    pub qkv: &'a [f32],
    pub z: &'a [f32],
    pub a: &'a [f32],
    pub b: &'a [f32],
}

impl Projected<'_> {
    pub(super) fn validate(self, dim: GdnDimensions) -> Result<()> {
        ensure!(
            self.rows > 0
                && self.rows.checked_mul(dim.conv_width()) == Some(self.qkv.len())
                && self.rows.checked_mul(dim.value_width()) == Some(self.z.len())
                && self.rows.checked_mul(dim.value_heads) == Some(self.a.len())
                && self.a.len() == self.b.len(),
            "GDN projected input shape mismatch"
        );
        ensure!(
            self.qkv
                .iter()
                .chain(self.z)
                .chain(self.a)
                .chain(self.b)
                .all(|v| v.is_finite()),
            "nonfinite GDN projected input"
        );
        Ok(())
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
        input: Projected<'_>,
        initial_state: &[f32],
        state: &mut [f32],
        conv: &mut ConvState,
    ) -> Result<()> {
        let dim = weights.dim;
        input.validate(dim)?;
        self.convolved.resize(input.qkv.len(), 0.0);
        conv.run_parallel(pool, input.qkv, &weights.conv, &mut self.convolved)?;
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
        self.log_decay.resize(input.a.len(), 0.0);
        self.beta.resize(input.b.len(), 0.0);
        for i in 0..input.a.len() {
            let h = i % dim.value_heads;
            self.log_decay[i] =
                -weights.a_log[h].exp() * activation::softplus(input.a[i] + weights.dt_bias[h]);
            self.beta[i] = activation::sigmoid(input.b[i]);
        }
        self.core.resize(input.rows * dim.value_width(), 0.0);
        self.recurrence.run_from(
            pool,
            dim,
            initial_state,
            state,
            &self.query,
            Updates {
                rows: input.rows,
                key: &self.key,
                value: &self.value,
                log_decay: &self.log_decay,
                beta: &self.beta,
            },
            &mut self.core,
        )?;
        self.output.resize(self.core.len(), 0.0);
        norm::gated_norm_parallel(
            pool,
            &self.core,
            input.z,
            &weights.norm,
            weights.epsilon,
            &mut self.output,
        )
    }
}
