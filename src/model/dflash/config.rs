use anyhow::{Result, ensure};
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Deserialize)]
pub struct DraftConfig {
    pub block_size: usize,
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}
#[derive(Clone, Deserialize)]
pub struct RopeConfig {
    pub rope_theta: f32,
}
#[derive(Clone, Deserialize)]
pub struct Config {
    pub architectures: Vec<String>,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeConfig,
    pub dflash_config: DraftConfig,
}
impl Config {
    pub fn load(path: &Path, context: usize) -> Result<Self> {
        let c: Self = serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
        ensure!(
            c.architectures.iter().any(|s| s == "DFlashDraftModel"),
            "expected a DFlashDraftModel checkpoint"
        );
        ensure!(
            c.layer_types.len() == c.num_hidden_layers,
            "invalid DFlash layer list"
        );
        ensure!(
            context <= c.sliding_window,
            "DFlash ctx-size exceeds the sliding window supported by this backend"
        );
        ensure!(
            c.layer_types
                .iter()
                .all(|s| matches!(s.as_str(), "sliding_attention" | "full_attention")),
            "unsupported DFlash attention type"
        );
        Ok(c)
    }
    pub fn context_width(&self) -> usize {
        self.hidden_size * self.dflash_config.target_layer_ids.len()
    }
}
