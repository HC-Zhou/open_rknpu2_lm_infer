use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod checkpoint;
pub mod dense;

#[derive(Debug, Clone)]
pub struct ModelOptions {
    pub path: PathBuf,
    pub context: usize,
    pub chunk: usize,
    pub threads: usize,
    pub npu_mask: u32,
    pub cpu_mask: u64,
    pub tune: bool,
    /// Permit device search only in the explicit tuning command.
    pub tune_search: bool,
    pub tune_cache: PathBuf,
    pub attention_tile: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub architecture: String,
    pub context: usize,
    pub vocab_size: usize,
    pub eos_tokens: Vec<u32>,
}

/// An instance owns one reusable KV cache and is used by a single worker.
/// New architectures register through `load` and implement this contract.
pub trait LanguageModel {
    fn info(&self) -> &ModelInfo;
    fn reset(&mut self) -> Result<()>;
    /// Reserve KV for a complete request before tuning and timed forwards.
    fn reserve(&mut self, _tokens: usize) -> Result<()> {
        Ok(())
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()>;
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]>;
    fn forward_cancellable(
        &mut self,
        tokens: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        anyhow::ensure!(
            !cancelled.load(std::sync::atomic::Ordering::Relaxed),
            "generation cancelled"
        );
        self.forward(tokens)
    }
    fn position(&self) -> usize;
    /// Score the next-token targets for every input row, retaining the KV prefix.
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        anyhow::ensure!(
            tokens.len() == targets.len(),
            "scoring input/target length mismatch"
        );
        tokens
            .iter()
            .zip(targets)
            .map(|(&token, &target)| {
                crate::ppl::negative_log_likelihood(self.forward(&[token])?, target)
            })
            .collect()
    }
    /// Keep an already computed prefix. Return false when the backend cannot
    /// truncate its cache and the caller should reset and recompute instead.
    fn truncate(&mut self, _position: usize) -> Result<bool> {
        Ok(false)
    }
    fn stats(&self) -> serde_json::Value;
}

pub fn load(mut options: ModelOptions) -> Result<Box<dyn LanguageModel>> {
    crate::npu::configure_cpu(&mut options)?;
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
    match config["model_type"].as_str() {
        Some("qwen3" | "llama" | "granite") => Ok(Box::new(dense::Dense::load(options)?)),
        other => anyhow::bail!(
            "unsupported model architecture: {other:?}; supported: qwen3, llama, granite"
        ),
    }
}
