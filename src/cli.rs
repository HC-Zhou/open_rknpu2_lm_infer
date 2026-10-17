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
    #[arg(
        short = 'c',
        long = "ctx-size",
        global = true,
        default_value_t = 4096,
        help = "KV capacity in tokens; 0 selects the required bench capacity or model limit"
    )]
    pub context: usize,
    #[arg(
        long = "ubatch-size",
        global = true,
        default_value_t = 128,
        help = "Physical prefill batch size; also accepts -ub"
    )]
    pub chunk: usize,
    #[arg(short = 't', long, global = true, default_value_t = 4)]
    pub threads: usize,
    #[arg(long, global = true, default_value = "7", value_parser = parse_u32)]
    pub npu_mask: u32,
    #[arg(short = 'C', long, global = true, default_value = "0xf0", value_parser = parse_u64)]
    pub cpu_mask: u64,
    #[arg(long, global = true)]
    pub tune_cache: Option<PathBuf>,
    #[arg(
        long,
        global = true,
        help = "Use built-in strategies instead of reading the tuning cache"
    )]
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
        if self.chunk == 0 || self.threads == 0 || self.attention_tile == 0 {
            bail!("ubatch-size, threads and attention-tile must be positive");
        }
        let context = if self.context == 0 {
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            crate::model::metadata::context_limit(&config)?
        } else {
            self.context
        };
        if self.npu_mask == 0 || self.npu_mask > 7 {
            bail!("npu-mask must select one or more of the three NPU cores (1..7)");
        }
        Ok(ModelOptions {
            path,
            context,
            chunk: self.chunk,
            threads: self.threads,
            npu_mask: self.npu_mask,
            cpu_mask: self.cpu_mask,
            tune: !self.no_tune,
            tune_search: false,
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
        input: Option<String>,
        #[arg(short = 'p', long, conflicts_with = "input")]
        prompt: Option<String>,
        #[arg(short = 'f', long = "file", conflicts_with_all = ["input", "prompt"])]
        prompt_file: Option<PathBuf>,
        #[arg(
            long = "conversation",
            help = "Apply the model chat template; also accepts -cnv"
        )]
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
        #[arg(long = "system-prompt", help = "System message; also accepts -sys")]
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
        input: Option<String>,
        #[arg(short = 'p', long, conflicts_with = "input")]
        prompt: Option<String>,
        #[arg(
            long = "conversation",
            help = "Apply the model chat template; also accepts -cnv"
        )]
        chat_template: bool,
        #[arg(long)]
        thinking: bool,
        #[arg(long, value_delimiter = ',', num_args = 1.., conflicts_with_all = ["input", "prompt"])]
        decode: Vec<u32>,
        #[arg(long, conflicts_with_all = ["input", "prompt", "decode"])]
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
    /// Benchmark independent pp/tg tests or a consecutive prefill+decode test (-pg).
    #[command(alias = "perf")]
    Bench(BenchArgs),
    /// Explicitly search missing strategies and save them for later inference.
    Tune {
        #[arg(long, value_delimiter = ',', num_args = 1..,
            help = "Actual prefill lengths to prepare; default: 1 and --ubatch-size")]
        batches: Vec<usize>,
        #[arg(long, help = "Write final statistics as JSON")]
        report: Option<PathBuf>,
    },
}
#[derive(Args)]
pub struct BenchArgs {
    #[arg(
        short = 'p',
        long = "n-prompt",
        value_delimiter = ',',
        default_value = "512",
        help = "Independent prefill lengths; 0 skips prefill tests"
    )]
    pub n_prompt: Vec<usize>,
    #[arg(
        short = 'n',
        long = "n-gen",
        value_delimiter = ',',
        default_value = "128",
        help = "Independent decode lengths; 0 skips decode tests"
    )]
    pub n_gen: Vec<usize>,
    #[arg(
        short = 'd',
        long = "n-depth",
        value_delimiter = ',',
        default_value = "0",
        help = "Untimed KV prefix lengths before each test"
    )]
    pub n_depth: Vec<usize>,
    #[arg(long = "n-prompt-gen", value_name = "PP,TG", value_parser = parse_prompt_gen,
        help = "Consecutive prefill+decode pair; also accepts -pg PP,TG; repeat for multiple pairs")]
    pub n_prompt_gen: Vec<PromptGen>,
    #[arg(short = 'r', long, default_value_t = 3)]
    pub repetitions: usize,
    #[arg(long, conflicts_with_all = ["n_prompt", "n_gen", "n_depth", "n_prompt_gen"])]
    pub suite: Option<String>,
    #[arg(long)]
    pub report: Option<PathBuf>,
}
#[derive(Clone, Debug)]
pub struct PromptGen(pub usize, pub usize);
fn parse_prompt_gen(value: &str) -> std::result::Result<PromptGen, String> {
    let (prompt, generate) = value
        .split_once(',')
        .ok_or("-pg requires two token counts: PP,TG")?;
    Ok(PromptGen(
        prompt.parse().map_err(|_| "invalid prefill token count")?,
        generate.parse().map_err(|_| "invalid decode token count")?,
    ))
}
impl BenchArgs {
    fn cases(&self) -> Result<Vec<crate::bench::Case>> {
        if let Some(suite) = &self.suite {
            return crate::bench::suite_cases(suite);
        }
        let mut cases = Vec::new();
        for &depth in &self.n_depth {
            for &prompt in &self.n_prompt {
                if prompt > 0 {
                    cases.push(crate::bench::Case::new(prompt, 0, depth));
                }
            }
            for &generate in &self.n_gen {
                if generate > 0 {
                    cases.push(crate::bench::Case::new(0, generate, depth));
                }
            }
            for &PromptGen(prompt, generate) in &self.n_prompt_gen {
                if prompt > 0 || generate > 0 {
                    cases.push(crate::bench::Case::new(prompt, generate, depth));
                }
            }
        }
        anyhow::ensure!(
            !cases.is_empty(),
            "benchmark needs at least one pp, tg or pg test"
        );
        Ok(cases)
    }
}
impl Cli {
    pub fn parse_args() -> Self {
        Self::parse_from(llama_args(std::env::args_os()))
    }
}
fn llama_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> Vec<std::ffi::OsString> {
    let mut positional = false;
    args.into_iter()
        .map(|arg| {
            if arg == "--" {
                positional = true;
            }
            if !positional && arg == "-pg" {
                "--n-prompt-gen".into()
            } else if !positional && arg == "-ub" {
                "--ubatch-size".into()
            } else if !positional && arg == "-cnv" {
                "--conversation".into()
            } else if !positional && arg == "-sys" {
                "--system-prompt".into()
            } else {
                arg
            }
        })
        .collect()
}
#[derive(Args, Clone)]
pub struct GenerationArgs {
    #[arg(
        short = 'n',
        long = "n-predict",
        visible_alias = "predict",
        default_value_t = 256
    )]
    pub max_tokens: usize,
    #[arg(long = "temp", visible_alias = "temperature", default_value_t = 0.6)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 20)]
    pub top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    pub min_p: f32,
    #[arg(short = 's', long)]
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
            bail!("n-predict must be positive");
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
    let mut model_args = cli.model;
    let bench_cases = if let Command::Bench(args) = &cli.command {
        let cases = args.cases()?;
        if model_args.context == 0 {
            model_args.context = cases.iter().try_fold(model_args.chunk, |capacity, case| {
                Ok::<_, anyhow::Error>(capacity.max(case.capacity()?))
            })?;
        }
        Some(cases)
    } else {
        None
    };
    let options = model_args.options()?;
    match cli.command {
        Command::Inspect => {
            println!(
                "{}",
                serde_json::to_string_pretty(&crate::model::metadata::inspect(
                    &options.path,
                    options.context
                )?)?
            );
        }
        Command::Tokenize {
            input,
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
                let tokens = codec.encode(&prompt(
                    read_prompt(text.or(input), None)?,
                    chat_template,
                    thinking,
                ))?;
                println!(
                    "{}",
                    serde_json::to_string(
                        &serde_json::json!({"count":tokens.len(),"tokens":tokens})
                    )?
                );
            }
        }
        Command::Run {
            input,
            prompt: text,
            prompt_file,
            chat_template,
            thinking,
            stats,
            generation,
        } => {
            let sampling = generation.sampling()?;
            let text = read_prompt(text.or(input), prompt_file)?;
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
        Command::Bench(args) => crate::bench::run(
            options,
            &bench_cases.expect("bench cases were validated before model options"),
            args.repetitions,
            args.report,
        )?,
        Command::Tune { batches, report } => {
            anyhow::ensure!(options.tune, "tune cannot be combined with --no-tune");
            let mut batches = if batches.is_empty() {
                vec![1, options.chunk]
            } else {
                batches
            };
            anyhow::ensure!(
                batches.iter().all(|&m| m > 0 && m <= options.chunk),
                "tuning batches must be within 1..ubatch-size"
            );
            batches.sort_unstable();
            batches.dedup();
            let _lock = crate::engine::BoardLock::acquire()?;
            let start = std::time::Instant::now();
            eprintln!(
                "Tuning: loading model; cache={} (existing entries are reused)",
                options.tune_cache.display()
            );
            let mut options = options;
            options.tune_search = true;
            let mut model = crate::model::load(options)?;
            for (index, &rows) in batches.iter().enumerate() {
                eprintln!(
                    "Tuning batch {}/{}: M={rows}, elapsed={:.1}s",
                    index + 1,
                    batches.len(),
                    start.elapsed().as_secs_f64()
                );
                model.prepare(rows)?;
            }
            let stats = serde_json::json!({"batches":batches,"elapsed_seconds":start.elapsed().as_secs_f64(),"stats":model.stats()});
            eprintln!(
                "Tuning complete: {:.1}s; hits={}, searched={}, fallback={}",
                start.elapsed().as_secs_f64(),
                stats["stats"]["tune_cache_hits"],
                stats["stats"]["tune_calls"],
                stats["stats"]["strategy_fallbacks"]
            );
            if let Some(path) = report {
                std::fs::write(path, serde_json::to_vec_pretty(&stats)?)?;
            }
            println!("{}", serde_json::to_string_pretty(&stats)?);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Cli {
        Cli::try_parse_from(llama_args(args.iter().map(std::ffi::OsString::from))).unwrap()
    }

    #[test]
    fn llama_resource_options_work_for_every_command() {
        for command in [
            vec!["inspect"],
            vec!["bench"],
            vec![
                "run",
                "-p",
                "hello",
                "--n-predict",
                "32",
                "--temp",
                "0",
                "-cnv",
            ],
            vec!["run", "-f", "prompt.txt"],
            vec!["chat", "-n", "32", "-sys", "Be concise.", "-s", "42"],
            vec!["serve"],
            vec![
                "eval",
                "--tokens-file",
                "ids.json",
                "--dump-logits",
                "logits.f32",
            ],
            vec!["ppl", "--tokens-file", "ids.json", "--report", "ppl.json"],
            vec!["tokenize", "-p", "hello"],
        ] {
            let mut args = vec![
                "infer", "-m", "model", "-c", "1024", "-ub", "32", "-t", "2", "-C", "0x30",
            ];
            args.extend(command);
            let cli = parse(&args);
            assert_eq!(
                (
                    cli.model.context,
                    cli.model.chunk,
                    cli.model.threads,
                    cli.model.cpu_mask
                ),
                (1024, 32, 2, 0x30)
            );
        }
    }

    #[test]
    fn llama_bench_cases_distinguish_pp_tg_and_pg() {
        let cli = parse(&[
            "infer", "bench", "-p", "512,1024", "-n", "128", "-d", "0,512",
        ]);
        let Command::Bench(args) = cli.command else {
            panic!("expected bench")
        };
        assert_eq!(
            args.cases().unwrap(),
            vec![
                crate::bench::Case::new(512, 0, 0),
                crate::bench::Case::new(1024, 0, 0),
                crate::bench::Case::new(0, 128, 0),
                crate::bench::Case::new(512, 0, 512),
                crate::bench::Case::new(1024, 0, 512),
                crate::bench::Case::new(0, 128, 512),
            ]
        );
        let cli = parse(&[
            "infer", "bench", "-c", "0", "-p", "0", "-n", "0", "-pg", "512,128", "-pg", "1024,32",
        ]);
        let Command::Bench(args) = cli.command else {
            panic!("expected bench")
        };
        let cases = args.cases().unwrap();
        assert_eq!(
            cases,
            vec![
                crate::bench::Case::new(512, 128, 0),
                crate::bench::Case::new(1024, 32, 0)
            ]
        );
        assert_eq!(cases[0].capacity().unwrap(), 640);
        assert_eq!(cases[1].capacity().unwrap(), 1056);
        let cli = parse(&["infer", "bench", "--suite", "contexts"]);
        let Command::Bench(args) = cli.command else {
            panic!("expected bench")
        };
        assert_eq!(args.cases().unwrap().len(), 6);
        assert!(
            Cli::try_parse_from(llama_args(["infer", "bench", "-pg", "512"].map(Into::into)))
                .is_err()
        );
        let cli = parse(&["infer", "tokenize", "--", "-pg"]);
        let Command::Tokenize { input, .. } = cli.command else {
            panic!("expected tokenize")
        };
        assert_eq!(input.as_deref(), Some("-pg"));
    }
}
