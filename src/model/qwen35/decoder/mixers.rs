use super::super::attention;
use super::super::timings::{Phase, Timings};
use super::{Buffers, Execution, Mixer};
use crate::model::projection::Projection;
use crate::{
    layers::gated_delta::Projected,
    ops::{activation, norm},
};
use anyhow::Result;
use std::time::Instant;

impl Mixer {
    pub fn forward(
        &mut self,
        inputs: &Projection,
        w: &mut Buffers,
        e: &Execution<'_>,
        timings: &mut Timings,
    ) -> Result<usize> {
        let Execution {
            config: c,
            cpu,
            rows,
            position,
            index,
            trace,
            rope_positions,
        } = *e;
        let h = c.hidden_size;
        let attended_width = match self {
            Mixer::Recurrent { weights, state } => {
                let (cw, vw, hv) = (c.gdn.conv_width(), c.gdn.value_width(), c.gdn.value_heads());
                let width = cw + vw + 2 * hv;
                timings.measure(Phase::Linear, || {
                    inputs.run(
                        &w.normalized[..rows * h],
                        &mut w.mixed[..rows * width],
                        rows,
                        &mut w.scatter,
                    )
                })?;
                let (split_ms, copy_ms) =
                    timings.measure(Phase::Gdn, || -> Result<(f64, f64)> {
                        let split_ms = if trace.enabled() {
                            let start = Instant::now();
                            let mut qkv = Vec::with_capacity(rows * cw);
                            for row in w.mixed[..rows * width].chunks(width) {
                                qkv.extend_from_slice(&row[..cw]);
                            }
                            let split_ms = start.elapsed().as_secs_f64() * 1000.;
                            trace.save(position, index, "gdn_qkv", &qkv)?;
                            split_ms
                        } else {
                            0.
                        };
                        state.forward_into(
                            &cpu.pool,
                            weights,
                            Projected::from_merged(c.gdn, rows, &w.mixed[..rows * width])?,
                            &mut w.recurrent,
                            &mut w.attended[..rows * vw],
                        )?;
                        Ok((split_ms, 0.))
                    })?;
                timings.record_gdn_io(split_ms, copy_ms);
                vw
            }
            Mixer::Attention {
                cache,
                q_norm,
                k_norm,
            } => {
                let (q, kv, d) = (
                    c.num_attention_heads * c.head_dim,
                    c.num_key_value_heads * c.head_dim,
                    c.head_dim,
                );
                let width = 2 * q + 2 * kv;
                timings.measure(Phase::Linear, || {
                    inputs.run(
                        &w.normalized[..rows * h],
                        &mut w.mixed[..rows * width],
                        rows,
                        &mut w.scatter,
                    )
                })?;
                timings.measure(Phase::Cpu, || -> Result<()> {
                    for (t, row) in w.mixed[..rows * width].chunks(width).enumerate() {
                        attention::split_query_gate(
                            &row[..2 * q],
                            d,
                            &mut w.query[t * q..(t + 1) * q],
                            &mut w.gate[t * q..(t + 1) * q],
                        )?;
                        w.key[t * kv..(t + 1) * kv].copy_from_slice(&row[2 * q..2 * q + kv]);
                        w.value[t * kv..(t + 1) * kv].copy_from_slice(&row[2 * q + kv..]);
                    }
                    norm::rms_norm(
                        &w.query[..rows * q],
                        q_norm,
                        c.epsilon,
                        true,
                        &mut w.norm_temp[..rows * q],
                    )?;
                    w.query[..rows * q].copy_from_slice(&w.norm_temp[..rows * q]);
                    norm::rms_norm(
                        &w.key[..rows * kv],
                        k_norm,
                        c.epsilon,
                        true,
                        &mut w.norm_temp[..rows * kv],
                    )?;
                    w.key[..rows * kv].copy_from_slice(&w.norm_temp[..rows * kv]);
                    c.rope.apply(
                        &mut w.query[..rows * q],
                        c.num_attention_heads,
                        rope_positions,
                    )?;
                    c.rope.apply(
                        &mut w.key[..rows * kv],
                        c.num_key_value_heads,
                        rope_positions,
                    )?;
                    trace.save(position, index, "query", &w.query[..rows * q])?;
                    trace.save(position, index, "key", &w.key[..rows * kv])?;
                    trace.save(position, index, "value", &w.value[..rows * kv])?;
                    Ok(())
                })?;
                timings.measure(Phase::Attention, || {
                    cache.run(
                        &w.query[..rows * q],
                        &w.key[..rows * kv],
                        &w.value[..rows * kv],
                        &mut w.attended[..rows * q],
                        rows,
                        position,
                    )
                })?;
                timings.measure(Phase::Cpu, || {
                    activation::sigmoid_gate(&mut w.attended[..rows * q], &w.gate[..rows * q])
                })?;
                q
            }
        };
        Ok(attended_width)
    }
}
