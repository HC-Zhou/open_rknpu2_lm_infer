use super::{
    config::Config,
    decoder::{Buffers, Execution, Layer, Mixer},
    multimodal::EmbeddedChunk,
    timings::{Phase, Timings},
};
use crate::model::trace::Trace;
use crate::ops::norm;
use crate::{
    model::{
        LanguageModel, ModelInfo, ModelOptions,
        checkpoint::{QuantLoader, Tensor},
        cpu::Cpu,
    },
    npu::Backend,
};
use anyhow::{Result, ensure};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub struct Qwen35 {
    pub(super) options: ModelOptions,
    pub(super) info: ModelInfo,
    pub(super) config: Config,
    pub(super) loader: Arc<QuantLoader>,
    pub(super) cpu: Cpu,
    pub(super) backends: Vec<Backend>,
    pub(super) layers: Vec<Layer>,
    pub(super) embedding: Tensor,
    pub(super) final_norm: Vec<f32>,
    pub(super) head: crate::model::output_head::OutputHead,
    pub(super) work: Buffers,
    pub(super) position: usize,
    pub(super) reserved: usize,
    pub(super) logits: Vec<f32>,
    pub(super) trace: Trace,
    pub(super) load_ms: f64,
    pub(super) forward_ms: f64,
    pub(super) head_ms: f64,
    pub(super) phases: Timings,
    pub(super) mtp_available: bool,
    pub(super) vision: Option<super::vision::Encoder>,
    pub(super) next_rope_position: usize,
}

impl Qwen35 {
    pub fn load(options: ModelOptions) -> Result<Self> {
        super::load::load(options)
    }

    pub(super) fn check_input(&self, tokens: &[u32]) -> Result<()> {
        ensure!(
            !tokens.is_empty() && tokens.len() <= self.options.context - self.position,
            "empty Qwen3.5 input or context capacity exceeded"
        );
        ensure!(
            tokens.iter().all(|v| (*v as usize) < self.info.vocab_size),
            "token outside Qwen3.5 vocabulary"
        );
        Ok(())
    }

    pub(super) fn chunk(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        prepared: Option<EmbeddedChunk<'_>>,
    ) -> Result<()> {
        let start = Instant::now();
        let h = self.config.hidden_size;
        self.phases.measure(Phase::Cpu, || {
            for (row, token) in tokens.iter().enumerate() {
                self.embedding.row(
                    *token as usize,
                    &mut self.work.hidden[row * h..(row + 1) * h],
                );
            }
        });
        let positions: Vec<_>;
        let rope_positions = if let Some(EmbeddedChunk {
            positions: p,
            features,
        }) = prepared
        {
            ensure!(p.len() == tokens.len(), "prepared position shape mismatch");
            for (start, embedding) in features {
                let end = start + embedding.len() / h;
                let first = self.position.max(*start);
                let last = (self.position + tokens.len()).min(end);
                if first < last {
                    self.work.hidden[(first - self.position) * h..(last - self.position) * h]
                        .copy_from_slice(&embedding[(first - start) * h..(last - start) * h]);
                }
            }
            p
        } else {
            positions = (self.next_rope_position..self.next_rope_position + tokens.len())
                .map(|p| [p; 3])
                .collect();
            &positions
        };
        for (i, layer) in self.layers.iter_mut().enumerate() {
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            layer.forward(
                &mut self.work,
                &Execution {
                    config: &self.config,
                    cpu: &self.cpu,
                    rows: tokens.len(),
                    position: self.position,
                    index: i as i32,
                    trace: &self.trace,
                    rope_positions,
                },
                &mut self.phases,
            )?;
        }
        self.position += tokens.len();
        self.next_rope_position = rope_positions
            .iter()
            .flatten()
            .copied()
            .max()
            .unwrap_or(self.next_rope_position)
            + 1;
        self.forward_ms += start.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    fn forward_impl(
        &mut self,
        tokens: &[u32],
        cancelled: Option<&AtomicBool>,
        logits: bool,
    ) -> Result<()> {
        self.check_input(tokens)?;
        self.reserve(self.position + tokens.len())?;
        let result = (|| {
            let mut rows = 0;
            for chunk in tokens.chunks(self.options.chunk) {
                self.chunk(chunk, cancelled, None)?;
                rows = chunk.len();
            }
            ensure!(
                !cancelled.is_some_and(|c| c.load(Ordering::Relaxed)),
                "generation cancelled"
            );
            if logits {
                self.output_head(rows)?;
            }
            Ok(())
        })();
        // A failed/cancelled hybrid chunk may have modified earlier layers. No
        // arbitrary GDN rollback is exposed until prefix checkpoints exist.
        if result.is_err() {
            self.reset()?;
        }
        result
    }

    fn score(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        self.check_input(tokens)?;
        ensure!(
            tokens.len() == targets.len()
                && targets.iter().all(|v| (*v as usize) < self.info.vocab_size),
            "invalid Qwen3.5 scoring targets"
        );
        self.reserve(self.position + tokens.len())?;
        let mut values = Vec::with_capacity(tokens.len());
        for (tokens, targets) in tokens
            .chunks(self.options.chunk)
            .zip(targets.chunks(self.options.chunk))
        {
            self.chunk(tokens, None, None)?;
            let start = Instant::now();
            let h = self.config.hidden_size;
            let rows = tokens.len();
            norm::rms_norm(
                &self.work.hidden[..rows * h],
                &self.final_norm,
                self.config.epsilon,
                true,
                &mut self.work.normalized[..rows * h],
            )?;
            let vocab = self.info.vocab_size;
            self.logits.resize(rows * vocab, 0.0);
            self.head.prepare(rows)?;
            self.head.run(
                &self.cpu,
                &self.work.normalized[..rows * h],
                &mut self.logits,
                rows,
            )?;
            for (logits, target) in self.logits.chunks(vocab).zip(targets) {
                values.push(crate::ppl::negative_log_likelihood(logits, *target)?);
            }
            self.head_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(values)
    }
}

impl LanguageModel for Qwen35 {
    fn info(&self) -> &ModelInfo {
        &self.info
    }
    fn position(&self) -> usize {
        self.position
    }
    fn reset(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Recurrent { state, .. } => state.reset(),
                Mixer::Attention { cache, .. } => cache.reset()?,
            }
        }
        self.position = 0;
        self.next_rope_position = 0;
        self.forward_ms = 0.0;
        self.head_ms = 0.0;
        self.phases = Timings::default();
        Ok(())
    }
    fn reserve(&mut self, tokens: usize) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= self.options.context,
            "Qwen3.5 reservation exceeds context"
        );
        if tokens > self.reserved {
            for layer in &self.layers {
                if let Mixer::Attention { cache, .. } = &layer.mixer {
                    cache.reserve(tokens)?;
                }
            }
            self.reserved = tokens.div_ceil(32) * 32;
        }
        Ok(())
    }
    fn truncate(&mut self, position: usize) -> Result<bool> {
        ensure!(
            position <= self.position,
            "cannot extend Qwen3.5 state by truncation"
        );
        if position == self.position {
            return Ok(true);
        }
        if position == 0 {
            self.reset()?;
            return Ok(true);
        }
        Ok(false)
    }
    fn prepare(&mut self, prompt_tokens: usize) -> Result<()> {
        ensure!(
            prompt_tokens > 0 && prompt_tokens <= self.options.context,
            "Qwen3.5 prompt exceeds context"
        );
        let mut rows = vec![1, self.options.chunk, prompt_tokens.min(self.options.chunk)];
        if !prompt_tokens.is_multiple_of(self.options.chunk) {
            rows.push(prompt_tokens % self.options.chunk);
        }
        rows.sort_unstable();
        rows.dedup();
        for backend in self.backends.iter().chain(self.head.backend()) {
            backend.retain_workspaces(&rows)?;
            for rows in &rows {
                backend.prepare(*rows)?;
            }
        }
        self.loader.store.release_pages()?;
        Ok(())
    }
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        self.forward_impl(tokens, None, false)
    }
    fn forward(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.forward_impl(tokens, None, true)?;
        Ok(&self.logits)
    }
    fn forward_prepared(
        &mut self,
        input: &crate::input::PreparedInput,
        start: usize,
        cancelled: &AtomicBool,
    ) -> Result<&[f32]> {
        self.consume_prepared(input, start, cancelled)?;
        Ok(&self.logits)
    }
    fn forward_cancellable(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<&[f32]> {
        self.forward_impl(tokens, Some(cancelled), true)?;
        Ok(&self.logits)
    }
    fn token_nll(&mut self, tokens: &[u32], targets: &[u32]) -> Result<Vec<f64>> {
        let result = self.score(tokens, targets);
        if result.is_err() {
            self.reset()?;
        }
        result
    }
    fn stats(&self) -> serde_json::Value {
        self.statistics()
    }
}
