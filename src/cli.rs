use crate::{
    engine::{Engine, GenerationEvent, GenerationRequest, Prompt},
    model::ModelOptions,
    sampling::SamplingParams,
    text::TextCodec,
};
use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use std::{
    io::{self, Read, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    name = "open_rknpu2_lm_infer",
    version,
    about = "Local LLM inference on the RK3588 NPU",
    subcommand_required = true
)]
pub struct Cli {
    #[command(flatten)]
    pub model: ModelArgs,
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Args)]
pub struct ModelArgs {
    #[arg(
        short = 'm',
        long,
        global = true,
        env = "RKNPU_MODEL",
        help = "Native safetensors model directory"
    )]
    pub model: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 4096)]
    pub context: usize,
    #[arg(long, global = true, default_value_t = 64)]
    pub chunk: usize,
    #[arg(long, global = true, default_value_t = 4)]
    pub threads: usize,
    #[arg(long, global = true, default_value = "7", value_parser = parse_u32)]
    pub npu_mask: u32,
    #[arg(long, global = true, default_value = "0xf0", value_parser = parse_u64)]
    pub cpu_mask: u64,
    #[arg(long, global = true)]
    pub tune_cache: Option<PathBuf>,
    #[arg(long, global = true)]
    pub no_tune: bool,
    #[arg(long, global = true, default_value_t = 512)]
    pub attention_tile: usize,
}
fn parse_u64(s: &str) -> std::result::Result<u64, String> {
    if let Some(s) = s.strip_prefix("0x") {
        u64::from_str_radix(s, 16)
    } else {
        s.parse()
    }
    .map_err(|e| e.to_string())
}
fn parse_u32(s: &str) -> std::result::Result<u32, String> {
    u32::try_from(parse_u64(s)?).map_err(|e| e.to_string())
}
impl ModelArgs {
    pub fn options(&self) -> Result<ModelOptions> {
        let path = self
            .model
            .clone()
            .context("provide --model or set RKNPU_MODEL")?;
        if self.context == 0 || self.chunk == 0 || self.threads == 0 || self.attention_tile == 0 {
            bail!("context, chunk, threads and attention-tile must be positive");
        }
        if self.npu_mask == 0 || self.npu_mask > 7 {
            bail!("npu-mask must select one or more of the three NPU cores (1..7)");
        }
        Ok(ModelOptions {
            path,
            context: self.context,
            chunk: self.chunk,
            threads: self.threads,
            npu_mask: self.npu_mask,
            cpu_mask: self.cpu_mask,
            tune: !self.no_tune,
            tune_cache: self.tune_cache.clone().unwrap_or_else(default_tune_cache),
            attention_tile: self.attention_tile,
        })
    }
}
fn default_tune_cache() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("open_rknpu2_lm_infer")
}
#[derive(Subcommand)]
pub enum Command {
    /// Generate text; read the prompt from stdin when omitted.
    Run {
        prompt: Option<String>,
        #[arg(long, conflicts_with = "prompt")]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        chat_template: bool,
        #[arg(long)]
        thinking: bool,
        #[arg(long)]
        stats: bool,
        #[command(flatten)]
        generation: GenerationArgs,
    },
    /// Open an interactive terminal conversation.
    Chat {
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        thinking: bool,
        #[command(flatten)]
        generation: GenerationArgs,
    },
    /// Serve the OpenAI chat and text completion endpoints.
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, env = "OPEN_RKNPU_API_KEY")]
        api_key: Option<String>,
    },
    /// Inspect model metadata without opening the NPU.
    Inspect,
    /// Encode text or decode token IDs without opening the NPU.
    Tokenize {
        prompt: Option<String>,
        #[arg(long)]
        chat_template: bool,
        #[arg(long)]
        thinking: bool,
        #[arg(long, value_delimiter = ',', num_args = 1.., conflicts_with = "prompt")]
        decode: Vec<u32>,
        #[arg(long, conflicts_with_all = ["prompt", "decode"])]
        tokens_file: Option<PathBuf>,
    },
    /// Evaluate supplied token sequences and export logits for reference comparison.
    Eval {
        #[arg(long)]
        tokens_file: PathBuf,
        #[arg(long)]
        continuation_file: Option<PathBuf>,
        #[arg(long)]
        dump_logits: PathBuf,
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Evaluate next-token perplexity on independent complete token windows.
    Ppl {
        #[arg(long)]
        tokens_file: PathBuf,
        #[arg(long, default_value_t = 257)]
        score_start: usize,
        #[arg(long)]
        report: PathBuf,
    },
    /// Measure prefill and cached decode on synthetic token sequences.
    #[command(alias = "perf")]
    Bench {
        #[arg(long, default_value_t = 128)]
        prompt: usize,
        #[arg(long, default_value_t = 32)]
        generate: usize,
        #[arg(long, default_value_t = 3)]
        repetitions: usize,
        #[arg(long, default_value_t = 0)]
        depth: usize,
        #[arg(long)]
        suite: Option<String>,
        #[arg(long)]
        report: Option<PathBuf>,
    },
}
#[derive(Args, Clone)]
pub struct GenerationArgs {
    #[arg(short = 'n', long, default_value_t = 256)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 0.6)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 20)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    pub min_p: f32,
    #[arg(long)]
    pub seed: Option<u64>,
    #[arg(long, default_value_t = 1.0)]
    pub repeat_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,
    #[arg(long)]
    pub stop: Vec<String>,
}
impl GenerationArgs {
    pub fn sampling(&self) -> Result<SamplingParams> {
        if self.max_tokens == 0 {
            bail!("max-tokens must be positive");
        }
        if self.stop.iter().any(String::is_empty) {
            bail!("stop strings must be nonempty");
        }
        let params = SamplingParams {
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            seed: self.seed,
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
        };
        params.validate()?;
        Ok(params)
    }
}
fn read_prompt(prompt: Option<String>, file: Option<PathBuf>) -> Result<String> {
    if let Some(prompt) = prompt {
        return Ok(prompt);
    }
    if let Some(path) = file {
        return std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()));
    }
    let mut text = String::new();
    io::stdin().read_to_string(&mut text)?;
    Ok(text)
}
fn prompt(text: String, chat: bool, thinking: bool) -> Prompt {
    if chat {
        Prompt::Chat {
            messages: vec![serde_json::json!({"role":"user","content":text})],
            tools: None,
            enable_thinking: thinking,
        }
    } else {
        Prompt::Text(text)
    }
}
pub async fn execute(cli: Cli) -> Result<()> {
    let options = cli.model.options()?;
    match cli.command {
        Command::Inspect => {
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(options.path.join("config.json"))?)?;
            let tokenizer_config = options.path.join("tokenizer_config.json");
            let tokenizer: serde_json::Value = if tokenizer_config.exists() {
                serde_json::from_slice(&std::fs::read(tokenizer_config)?)?
            } else {
                serde_json::Value::Null
            };
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"path":options.path,"model_type":config["model_type"],"architectures":config["architectures"],"layers":config["num_hidden_layers"],"hidden_size":config["hidden_size"],"attention_heads":config["num_attention_heads"],"kv_heads":config["num_key_value_heads"],"vocab_size":config["vocab_size"],"max_position_embeddings":config["max_position_embeddings"],"quantization_config":config["quantization_config"],"chat_template":options.path.join("chat_template.jinja").exists() || tokenizer.get("chat_template").is_some(),"requested_context":options.context})
                )?
            );
        }
        Command::Tokenize {
            prompt: text,
            chat_template,
            thinking,
            decode,
            tokens_file,
        } => {
            let codec = TextCodec::load(&options.path)?;
            if let Some(path) = tokens_file {
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                let tokens: Vec<u32> =
                    serde_json::from_value(value.get("tokens").cloned().unwrap_or(value))?;
                println!("{}", codec.decode(&tokens, false)?);
            } else if !decode.is_empty() {
                println!("{}", codec.decode(&decode, false)?);
            } else {
                let tokens =
                    codec.encode(&prompt(read_prompt(text, None)?, chat_template, thinking))?;
                println!(
                    "{}",
                    serde_json::to_string(
                        &serde_json::json!({"count":tokens.len(),"tokens":tokens})
                    )?
                );
            }
        }
        Command::Run {
            prompt: text,
            prompt_file,
            chat_template,
            thinking,
            stats,
            generation,
        } => {
            let sampling = generation.sampling()?;
            let text = read_prompt(text, prompt_file)?;
            let engine = Engine::start(options)?;
            let mut handle = engine.generate(GenerationRequest {
                prompt: prompt(text, chat_template, thinking),
                sampling,
                max_tokens: generation.max_tokens,
                stop: generation.stop,
            })?;
            let interrupted = tokio::signal::ctrl_c();
            tokio::pin!(interrupted);
            loop {
                tokio::select! {
                    _=&mut interrupted => { handle.cancel.store(true,std::sync::atomic::Ordering::Relaxed); eprintln!("\nCancelled"); break; }
                    event=handle.events.recv() => match event {
                        Some(GenerationEvent::Token(text)) => { print!("{text}"); io::stdout().flush()?; }
                        Some(GenerationEvent::Finished(result)) => { println!(); if stats { eprintln!("{}",serde_json::to_string_pretty(&serde_json::json!({"prompt_tokens":result.prompt_tokens,"completion_tokens":result.completion_tokens,"cached_tokens":result.cached_tokens,"finish_reason":result.finish_reason,"timings":result.timings,"stats":result.stats}))?); } break; }
                        Some(GenerationEvent::Error(message)) => bail!("{message}"),
                        None => bail!("generation ended without a result"),
                    }
                }
            }
        }
        Command::Chat {
            system,
            thinking,
            generation,
        } => {
            let sampling = generation.sampling()?;
            let engine = Engine::start(options)?;
            crate::tui::chat(
                engine,
                system,
                thinking,
                sampling,
                generation.max_tokens,
                generation.stop,
            )
            .await?;
        }
        Command::Serve {
            host,
            port,
            api_key,
        } => {
            let engine = Engine::start(options)?;
            crate::api::serve(engine, &host, port, api_key).await?;
        }
        Command::Bench {
            prompt,
            generate,
            repetitions,
            depth,
            suite,
            report,
        } => match suite.as_deref() {
            Some("representative") => crate::bench::suite(options, repetitions, report)?,
            Some("contexts") => crate::bench::contexts(options, repetitions, report)?,
            None => crate::bench::run(options, prompt, generate, repetitions, report, depth)?,
            Some(other) => bail!("unknown benchmark suite: {other}"),
        },
        Command::Eval {
            tokens_file,
            continuation_file,
            dump_logits,
            report,
        } => crate::validation::run(options, tokens_file, continuation_file, dump_logits, report)?,
        Command::Ppl {
            tokens_file,
            score_start,
            report,
        } => crate::ppl::run(options, tokens_file, score_start, report)?,
    }
    Ok(())
}
