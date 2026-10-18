use super::{
    config::Config,
    timings::{Phase, Timings},
};
use crate::model::trace::Trace;
use crate::{
    layers::gated_delta::{LayerState, MixerWeights, Scratch},
    ops::norm,
};
use crate::{
    model::{cpu::Cpu, projection::Projection},
    npu::{Attention, Linear},
};
use anyhow::Result;

mod mixers;

#[derive(Clone, Copy)]
pub(super) struct Execution<'a> {
    pub config: &'a Config,
    pub cpu: &'a Cpu,
    pub rows: usize,
    pub position: usize,
    pub index: i32,
    pub trace: &'a Trace,
    pub rope_positions: &'a [[usize; 3]],
}

pub(super) enum Mixer {
    Recurrent {
        weights: MixerWeights,
        state: Box<LayerState>,
    },
    Attention {
        cache: Attention,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    },
}

pub(super) struct Layer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub inputs: Projection,
    pub output: Linear,
    pub gate_up: Projection,
    pub down: Linear,
    pub mixer: Mixer,
}

pub(super) struct Buffers {
    pub hidden: Vec<f32>,
    pub normalized: Vec<f32>,
    projected: Vec<f32>,
    mixed: Vec<f32>,
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    gate: Vec<f32>,
    norm_temp: Vec<f32>,
    attended: Vec<f32>,
    gate_up: Vec<f32>,
    middle: Vec<f32>,
    scatter: Vec<f32>,
    pub recurrent: Scratch,
}

impl Buffers {
    pub fn new(c: &Config, rows: usize) -> Self {
        let q = c.num_attention_heads * c.head_dim;
        let kv = c.num_key_value_heads * c.head_dim;
        let mixed = (2 * q + 2 * kv)
            .max(c.gdn.conv_width() + c.gdn.value_width() + 2 * c.gdn.value_heads());
        let zero = |width| vec![0.0; rows * width];
        Self {
            hidden: zero(c.hidden_size),
            normalized: zero(c.hidden_size),
            projected: zero(c.hidden_size),
            mixed: zero(mixed),
            query: zero(q),
            key: zero(kv),
            value: zero(kv),
            gate: zero(q),
            norm_temp: zero(q.max(kv)),
            attended: zero(q.max(c.gdn.value_width())),
            gate_up: zero(2 * c.intermediate_size),
            middle: zero(c.intermediate_size),
            scatter: zero(mixed.max(2 * c.intermediate_size)),
            recurrent: Scratch::default(),
        }
    }
    pub fn allocated_bytes(&self) -> usize {
        [
            &self.hidden,
            &self.normalized,
            &self.projected,
            &self.mixed,
            &self.query,
            &self.key,
            &self.value,
            &self.gate,
            &self.norm_temp,
            &self.attended,
            &self.gate_up,
            &self.middle,
            &self.scatter,
        ]
        .iter()
        .map(|v| v.capacity() * 4)
        .sum::<usize>()
            + self.recurrent.allocated_bytes()
    }
}

impl Layer {
    pub fn forward(
        &mut self,
        w: &mut Buffers,
        e: &Execution<'_>,
        timings: &mut Timings,
    ) -> Result<()> {
        let Execution {
            config: c,
            cpu,
            rows,
            position,
            index,
            trace,
            ..
        } = *e;
        let h = c.hidden_size;
        timings.measure(Phase::Cpu, || {
            norm::rms_norm(
                &w.hidden[..rows * h],
                &self.input_norm,
                c.epsilon,
                true,
                &mut w.normalized[..rows * h],
            )
        })?;
        trace.save(position, index, "input", &w.normalized[..rows * h])?;
        let attended_width = self.mixer.forward(&self.inputs, w, e, timings)?;
        trace.save(
            position,
            index,
            "attn",
            &w.attended[..rows * attended_width],
        )?;
        timings.measure(Phase::Linear, || {
            self.output.run(
                &w.attended[..rows * attended_width],
                &mut w.projected[..rows * h],
                rows,
            )
        })?;
        trace.save(position, index, "o", &w.projected[..rows * h])?;
        timings.measure(Phase::Cpu, || {
            cpu.add(&mut w.hidden[..rows * h], &w.projected[..rows * h]);
            norm::rms_norm(
                &w.hidden[..rows * h],
                &self.post_norm,
                c.epsilon,
                true,
                &mut w.normalized[..rows * h],
            )
        })?;
        trace.save(position, index, "post", &w.normalized[..rows * h])?;
        let f = c.intermediate_size;
        let fused = !trace.enabled()
            && timings.measure(Phase::Linear, || {
                self.gate_up
                    .run_swiglu(&w.normalized[..rows * h], &mut w.middle[..rows * f], rows)
            })?;
        if !fused {
            timings.measure(Phase::Linear, || {
                self.gate_up.run(
                    &w.normalized[..rows * h],
                    &mut w.gate_up[..rows * 2 * f],
                    rows,
                    &mut w.scatter,
                )
            })?;
            timings.measure(Phase::Cpu, || {
                cpu.swiglu(&w.gate_up[..rows * 2 * f], &mut w.middle[..rows * f], f)
            });
        }
        trace.save(position, index, "middle", &w.middle[..rows * f])?;
        timings.measure(Phase::Linear, || {
            self.down
                .run(&w.middle[..rows * f], &mut w.projected[..rows * h], rows)
        })?;
        timings.measure(Phase::Cpu, || {
            cpu.add(&mut w.hidden[..rows * h], &w.projected[..rows * h])
        });
        trace.save(position, index, "hidden", &w.hidden[..rows * h])
    }
}
