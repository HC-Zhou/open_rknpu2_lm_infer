use super::super::attention;
use super::super::timings::{Phase, Timings};
use super::{Buffers, Execution, Mixer};
use crate::model::projection::Projection;
use crate::{
    layers::gated_delta::Projected,
    ops::{activation, norm},
};
use anyhow::Result;

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
                timings.measure(Phase::Gdn, || -> Result<()> {
                    for (t, row) in w.mixed[..rows * width].chunks(width).enumerate() {
                        w.qkv[t * cw..(t + 1) * cw].copy_from_slice(&row[..cw]);
                        w.z[t * vw..(t + 1) * vw].copy_from_slice(&row[cw..cw + vw]);
                        w.a[t * hv..(t + 1) * hv].copy_from_slice(&row[cw + vw..cw + vw + hv]);
                        w.b[t * hv..(t + 1) * hv].copy_from_slice(&row[cw + vw + hv..]);
                    }
                    trace.save(position, index, "gdn_qkv", &w.qkv[..rows * cw])?;
                    let output = state.forward(
                        &cpu.pool,
                        weights,
                        Projected {
                            rows,
                            qkv: &w.qkv[..rows * cw],
                            z: &w.z[..rows * vw],
                            a: &w.a[..rows * hv],
                            b: &w.b[..rows * hv],
                        },
                        &mut w.recurrent,
                    )?;
                    w.attended[..rows * vw].copy_from_slice(output);
                    Ok(())
                })?;
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
                    let positions: Vec<_> = (position..position + rows).map(|p| [p; 3]).collect();
                    c.rope
                        .apply(&mut w.query[..rows * q], c.num_attention_heads, &positions)?;
                    c.rope
                        .apply(&mut w.key[..rows * kv], c.num_key_value_heads, &positions)?;
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
