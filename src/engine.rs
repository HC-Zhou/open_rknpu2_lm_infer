use crate::{
    model::{self, LanguageModel, ModelInfo, ModelOptions},
    sampling::{Sampler, SamplingParams},
    text::{StopFilter, TextCodec},
};
use anyhow::{Context, Result, anyhow, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    sync::{
        Arc, Mutex, RwLock,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
mod context;
use context::{PrefixCache, validate_request_tokens};

#[derive(Debug, Clone)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<u32>),
    Chat {
        messages: Vec<Value>,
        tools: Option<Value>,
        enable_thinking: bool,
    },
}

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub prompt: Prompt,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub finish_reason: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub cached_tokens: usize,
    pub timings: Value,
    pub stats: Value,
}

#[derive(Debug, Clone)]
pub enum GenerationEvent {
    Token(String),
    Finished(GenerationResult),
    Error(String),
}

pub struct GenerationHandle {
    pub events: mpsc::Receiver<GenerationEvent>,
    pub cancel: Arc<AtomicBool>,
}
impl Drop for GenerationHandle {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct EngineBusy;
impl std::fmt::Display for EngineBusy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("inference request queue is full")
    }
}
impl std::error::Error for EngineBusy {}

struct Job {
    input: crate::input::PreparedInput,
    request: GenerationRequest,
    events: mpsc::Sender<GenerationEvent>,
    cancel: Arc<AtomicBool>,
    queued: Instant,
}

struct EngineInner {
    sender: Option<mpsc::Sender<Job>>,
    info: ModelInfo,
    codec: Arc<TextCodec>,
    stats: Arc<RwLock<Value>>,
    active: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    shutdown: Arc<AtomicBool>,
    worker: Option<thread::JoinHandle<()>>,
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Ok(active) = self.active.lock()
            && let Some(cancel) = active.as_ref()
        {
            cancel.store(true, Ordering::Relaxed);
        }
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

impl Engine {
    pub fn start(options: ModelOptions) -> Result<Self> {
        let codec = Arc::new(TextCodec::load(&options.path)?);
        let worker_codec = codec.clone();
        let (sender, mut receiver) = mpsc::channel::<Job>(4);
        let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
        let stats = Arc::new(RwLock::new(Value::Null));
        let worker_stats = stats.clone();
        let active = Arc::new(Mutex::new(None));
        let worker_active = active.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = shutdown.clone();
        let worker = thread::Builder::new()
            .name("rknpu-inference".into())
            .spawn(move || {
                let loaded = (|| -> Result<_> {
                    let chunk = options.chunk;
                    let mut options = options;
                    options.tune_search = false;
                    let mut model = model::load(options)?;
                    model.prepare(chunk)?;
                    Ok(model)
                })();
                let mut model = match loaded {
                    Ok(loaded) => loaded,
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                        return;
                    }
                };
                if let Ok(mut latest) = worker_stats.write() {
                    *latest = model.stats();
                }
                if ready_tx.send(Ok(model.info().clone())).is_err() {
                    return;
                }
                let mut cache = PrefixCache::default();
                while let Some(job) = receiver.blocking_recv() {
                    if worker_shutdown.load(Ordering::Relaxed) || job.cancel.load(Ordering::Relaxed)
                    {
                        continue;
                    }
                    if let Ok(mut running) = worker_active.lock() {
                        *running = Some(job.cancel.clone());
                    }
                    let outcome = run_generation(model.as_mut(), &worker_codec, &job, &mut cache);
                    if let Err(error) = outcome {
                        let _ = model.reset();
                        cache.clear();
                        if !job.cancel.load(Ordering::Relaxed) {
                            let _ = send_event(&job, GenerationEvent::Error(format!("{error:#}")));
                        }
                    }
                    if let Ok(mut latest) = worker_stats.write() {
                        *latest = model.stats();
                    }
                    if let Ok(mut running) = worker_active.lock() {
                        *running = None;
                    }
                }
            })?;
        let info = ready_rx
            .recv()
            .context("inference worker stopped during loading")??;
        Ok(Self {
            inner: Arc::new(EngineInner {
                sender: Some(sender),
                info,
                codec,
                stats,
                active,
                shutdown,
                worker: Some(worker),
            }),
        })
    }

    pub fn info(&self) -> &ModelInfo {
        &self.inner.info
    }
    pub fn stats(&self) -> Value {
        self.inner
            .stats
            .read()
            .map(|v| v.clone())
            .unwrap_or(Value::Null)
    }
    pub fn tokenize(&self, prompt: &Prompt) -> Result<Vec<u32>> {
        Ok(self.inner.codec.prepare(prompt)?.tokens)
    }
    pub fn prompt_opens_thinking(&self, prompt: &Prompt) -> Result<bool> {
        Ok(matches!(prompt, Prompt::Chat { .. })
            && self
                .inner
                .codec
                .format(prompt)?
                .trim_end()
                .ends_with("<think>"))
    }
    pub fn shutdown(&self) {
        self.inner.shutdown.store(true, Ordering::Relaxed);
        if let Ok(active) = self.inner.active.lock()
            && let Some(cancel) = active.as_ref()
        {
            cancel.store(true, Ordering::Relaxed);
        }
    }

    pub fn generate(&self, request: GenerationRequest) -> Result<GenerationHandle> {
        ensure!(
            !self.inner.shutdown.load(Ordering::Relaxed),
            "inference worker is shutting down"
        );
        request.sampling.validate()?;
        ensure!(
            request.stop.len() <= 16,
            "at most 16 stop sequences are supported"
        );
        ensure!(
            request.stop.iter().all(|stop| !stop.is_empty()),
            "stop sequences must not be empty"
        );
        let input = self.inner.codec.prepare(&request.prompt)?;
        validate_request_tokens(&input.tokens, request.max_tokens, self.info())?;
        let (events_tx, events) = mpsc::channel(32);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            input,
            request,
            events: events_tx,
            cancel: cancel.clone(),
            queued: Instant::now(),
        };
        match self.inner.sender.as_ref().unwrap().try_send(job) {
            Ok(()) => Ok(GenerationHandle { events, cancel }),
            Err(mpsc::error::TrySendError::Full(_)) => Err(EngineBusy.into()),
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow!("inference worker is unavailable"))
            }
        }
    }
}

fn send_event(job: &Job, mut event: GenerationEvent) -> Result<()> {
    loop {
        ensure!(!job.cancel.load(Ordering::Relaxed), "generation cancelled");
        match job.events.try_send(event) {
            Ok(()) => return Ok(()),
            Err(mpsc::error::TrySendError::Closed(_)) => anyhow::bail!("generation cancelled"),
            Err(mpsc::error::TrySendError::Full(returned)) => {
                event = returned;
                thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

fn run_generation(
    model: &mut dyn LanguageModel,
    codec: &TextCodec,
    job: &Job,
    cache: &mut PrefixCache,
) -> Result<()> {
    let start = Instant::now();
    let queue_ms = job.queued.elapsed().as_secs_f64() * 1000.0;
    let mut cached_tokens = cache.reusable(&job.input, model.position());
    if cached_tokens > 0 && cached_tokens < model.position() {
        if model.truncate(cached_tokens)? {
            cache.truncate(cached_tokens);
        } else {
            cached_tokens = 0;
        }
    }
    if cached_tokens == 0 {
        model.reset()?;
        cache.clear();
    }
    let remaining = &job.input.tokens[cached_tokens..];
    let prepare_start = Instant::now();
    model.reserve(job.input.tokens.len() + job.request.max_tokens)?;
    if !remaining.is_empty() {
        model.prepare(remaining.len())?;
    }
    let prepare_ms = prepare_start.elapsed().as_secs_f64() * 1000.0;
    let prefill_start = Instant::now();
    if !remaining.is_empty() {
        cache.logits.clear();
        cache.logits.extend_from_slice(model.forward_prepared(
            &job.input,
            cached_tokens,
            &job.cancel,
        )?);
        cache.extend_input(&job.input, cached_tokens);
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let mut sampler = Sampler::new(
        job.request.sampling.clone(),
        model.info().vocab_size,
        &job.input.tokens,
    )?;
    let mut filter = StopFilter::new(job.request.stop.clone())?;
    let mut decoder = codec.tokenizer.decode_stream(true);
    let mut text = String::new();
    let mut generated = 0;
    let mut reason = "length";
    let mut ttft_ms = None;
    let mut decode_calls = 0;
    let mut decode_model_ms = 0.0;
    let decode_start = Instant::now();
    for index in 0..job.request.max_tokens {
        ensure!(!job.cancel.load(Ordering::Relaxed), "generation cancelled");
        let token = sampler.sample(&cache.logits)?;
        generated += 1;
        if model.info().eos_tokens.contains(&token) {
            reason = "stop";
            break;
        }
        sampler.observe(token);
        if let Some(piece) = decoder
            .step(token)
            .map_err(|e| anyhow!("stream decode: {e}"))?
        {
            let (output, stopped) = filter.push(&piece);
            if !output.is_empty() {
                text.push_str(&output);
                if ttft_ms.is_none() {
                    ttft_ms = Some(start.elapsed().as_secs_f64() * 1000.0);
                }
                send_event(job, GenerationEvent::Token(output))?;
            }
            if stopped {
                reason = "stop";
                break;
            }
        }
        if index + 1 < job.request.max_tokens {
            let call = Instant::now();
            cache.logits.clear();
            cache
                .logits
                .extend_from_slice(model.forward_cancellable(&[token], &job.cancel)?);
            decode_model_ms += call.elapsed().as_secs_f64() * 1000.0;
            decode_calls += 1;
            cache.append_token(token);
        }
    }
    let tail = filter.finish();
    if !tail.is_empty() {
        text.push_str(&tail);
        send_event(job, GenerationEvent::Token(tail))?;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let result = GenerationResult {
        text,
        finish_reason: reason.into(),
        prompt_tokens: job.input.tokens.len(),
        completion_tokens: generated,
        cached_tokens,
        timings: json!({"queue_ms":queue_ms,"prepare_ms":prepare_ms,"prefill_ms":prefill_ms,
            "ttft_ms":ttft_ms,"decode_ms":decode_ms,"decode_model_ms":decode_model_ms,"decode_calls":decode_calls,
            "decode_tokens_per_second": if decode_model_ms > 0.0 { Some(decode_calls as f64 * 1000.0 / decode_model_ms) } else { None },
            "generation_ms":start.elapsed().as_secs_f64()*1000.0}),
        stats: model.stats(),
    };
    tracing::info!(prompt_tokens = result.prompt_tokens, cached_tokens = result.cached_tokens,
        completion_tokens = result.completion_tokens, timings = %result.timings,
        tune_cache_hits = %result.stats["tune_cache_hits"],
        tune_cache_misses = %result.stats["tune_cache_misses"],
        tune_calls = %result.stats["tune_calls"],
        weight_reload_calls = %result.stats["weight_reload_calls"],
        weight_profile_creations = %result.stats["weight_profile_creations"],
        strategy_fallback_calls = %result.stats["strategy_fallback_calls"],
        strategy_fallbacks = %result.stats["strategy_fallbacks"], "generation finished");
    send_event(job, GenerationEvent::Finished(result))
}
