use super::config::Config;
use crate::{
    model::{
        cpu::Cpu,
        projection::{Linear, Projection},
    },
    npu::Attention,
    ops::rope::MRope,
};
use anyhow::Result;

pub(super) struct Layer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub query: Linear,
    pub kv: Projection,
    pub output: Linear,
    pub gate_up: Projection,
    pub down: Linear,
    pub cache: Attention,
}

impl Layer {
    fn project_kv(
        &self,
        c: &Config,
        cpu: &Cpu,
        rope: &MRope,
        input: &[f32],
        rows: usize,
        start: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let kv = c.num_key_value_heads * c.head_dim;
        let mut joined = vec![0.; rows * 2 * kv];
        let mut scratch = vec![0.; rows * self.kv.scratch_width()];
        self.kv.run(input, &mut joined, rows, &mut scratch)?;
        let mut keys = Vec::with_capacity(rows * kv);
        let mut values = Vec::with_capacity(rows * kv);
        for row in joined.chunks_exact(2 * kv) {
            keys.extend_from_slice(&row[..kv]);
            values.extend_from_slice(&row[kv..]);
        }
        let mut normalized = vec![0.; keys.len()];
        cpu.norm(&keys, &mut normalized, &self.k_norm, c.rms_norm_eps);
        let positions: Vec<_> = (start..start + rows).map(|p| [p; 3]).collect();
        rope.apply(&mut normalized, c.num_key_value_heads, &positions)?;
        Ok((normalized, values))
    }
    pub fn append_context(
        &self,
        c: &Config,
        cpu: &Cpu,
        rope: &MRope,
        input: &[f32],
        rows: usize,
        start: usize,
    ) -> Result<()> {
        let (keys, values) = self.project_kv(c, cpu, rope, input, rows, start)?;
        self.cache.append(&keys, &values, rows, start)
    }
    /// Context and noise share one K/V projection. Only the context survives
    /// this draft call; each round recomputes its noisy proposal block.
    pub fn forward(
        &self,
        c: &Config,
        cpu: &Cpu,
        rope: &MRope,
        hidden: &mut [f32],
        context: &[f32],
        position: usize,
    ) -> Result<()> {
        let h = c.hidden_size;
        let rows = hidden.len() / h;
        let context_rows = context.len() / h;
        let mut normalized = vec![0.; hidden.len()];
        cpu.norm(hidden, &mut normalized, &self.input_norm, c.rms_norm_eps);
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let mut query = vec![0.; rows * q];
        self.query.run(&normalized, &mut query, rows)?;
        let mut normalized_q = vec![0.; query.len()];
        cpu.norm(&query, &mut normalized_q, &self.q_norm, c.rms_norm_eps);
        let positions: Vec<_> = (position..position + rows).map(|p| [p; 3]).collect();
        rope.apply(&mut normalized_q, c.num_attention_heads, &positions)?;
        let mut inputs = Vec::with_capacity(context.len() + normalized.len());
        inputs.extend_from_slice(context);
        inputs.extend_from_slice(&normalized);
        let (keys, values) = self.project_kv(
            c,
            cpu,
            rope,
            &inputs,
            rows + context_rows,
            position - context_rows,
        )?;
        let offset = context_rows * kv;
        if context_rows > 0 {
            self.cache.append(
                &keys[..offset],
                &values[..offset],
                context_rows,
                position - context_rows,
            )?;
        }
        let mut attended = vec![0.; rows * q];
        self.cache.run(
            &normalized_q,
            &keys[offset..],
            &values[offset..],
            &mut attended,
            rows,
            position,
        )?;
        self.cache.truncate(position)?;
        let mut projected = vec![0.; rows * h];
        self.output.run(&attended, &mut projected, rows)?;
        cpu.add(hidden, &projected);
        cpu.norm(hidden, &mut normalized, &self.post_norm, c.rms_norm_eps);
        let f = c.intermediate_size;
        let mut gate_up = vec![0.; rows * 2 * f];
        let mut scratch = vec![0.; rows * self.gate_up.scratch_width()];
        self.gate_up
            .run(&normalized, &mut gate_up, rows, &mut scratch)?;
        let mut middle = vec![0.; rows * f];
        cpu.swiglu(&gate_up, &mut middle, f);
        self.down.run(&middle, &mut projected, rows)?;
        cpu.add(hidden, &projected);
        Ok(())
    }
}
