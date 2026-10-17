//! Qwen3.5 RoPE config defaults and supported scaling policy.
use crate::ops::rope::MRope;
use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub(super) fn from_config(text: &Value, head_dim: usize) -> Result<MRope> {
    let rope = &text["rope_parameters"];
    ensure!(
        rope["rope_type"] == "default" && text["rope_scaling"].is_null(),
        "unsupported Qwen3.5 RoPE scaling"
    );
    let theta = rope["rope_theta"].as_f64().context("missing rope_theta")? as f32;
    let partial = rope
        .get("partial_rotary_factor")
        .or_else(|| text.get("partial_rotary_factor"))
        .map_or(Some(0.25), Value::as_f64)
        .context("invalid partial_rotary_factor")?;
    ensure!(
        partial.is_finite() && partial > 0.0 && partial <= 1.0,
        "invalid partial_rotary_factor"
    );
    let rotary_dim = (head_dim as f64 * partial) as usize;
    let sections: [usize; 3] = serde_json::from_value(
        rope.get("mrope_section")
            .cloned()
            .unwrap_or_else(|| serde_json::json!([11, 11, 10])),
    )?;
    let interleaved = rope
        .get("mrope_interleaved")
        .map_or(Some(true), Value::as_bool)
        .context("invalid mrope_interleaved")?;
    MRope::new(head_dim, rotary_dim, theta, sections, interleaved)
}
