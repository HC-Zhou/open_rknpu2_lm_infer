use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub mod checkpoint;
pub(crate) mod cpu;
pub mod dense;
pub(crate) mod metadata;
pub(crate) mod metrics;
pub(crate) mod output_head;
pub(crate) mod projection;
pub mod qwen35;
pub(crate) mod trace;

/// ViT activation storage and CPU arithmetic. NPU GEMMs remain FP16 in both modes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum VisionDtype {
    #[default]
    Fp16,
    Fp32,
}
impl VisionDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        }
    }
}

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
    pub vision_dtype: VisionDtype,
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
    /// Consume prompt/history tokens without evaluating final output norm or
    /// lm_head. Generation uses `forward` when next-token logits are required.
    fn prefill(&mut self, _tokens: &[u32]) -> Result<()> {
        anyhow::bail!("this model has no head-free prefill implementation")
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]>;
    /// Consume a suffix of a prepared prompt. The worker owns all model/device
    /// state; image decoding and token expansion have already completed.
    fn forward_prepared(
        &mut self,
        input: &crate::input::PreparedInput,
        start: usize,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<&[f32]> {
        anyhow::ensure!(
            input.images.is_empty(),
            "this model does not support images"
        );
        anyhow::ensure!(start < input.tokens.len(), "empty prepared suffix");
        self.forward_cancellable(&input.tokens[start..], cancelled)
    }
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
        Some("qwen3_5" | "qwen3_5_text") => Ok(Box::new(qwen35::Qwen35::load(options)?)),
        other => anyhow::bail!(
            "unsupported model architecture: {other:?}; supported: qwen3, llama, granite, qwen3_5, qwen3_5_text"
        ),
    }
}
