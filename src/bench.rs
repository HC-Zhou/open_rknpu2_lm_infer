use crate::{
    engine::BoardLock,
    model::{self, LanguageModel, ModelOptions},
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{path::PathBuf, time::Instant};

fn summary(samples: &[f64], tokens: usize) -> serde_json::Value {
    let speeds: Vec<_> = samples
        .iter()
        .map(|ms| tokens as f64 * 1000.0 / ms)
        .collect();
    let mean = speeds.iter().sum::<f64>() / speeds.len() as f64;
    let stddev = if speeds.len() > 1 {
        (speeds.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (speeds.len() - 1) as f64).sqrt()
    } else {
        0.0
    };
    json!({"tokens":tokens,"samples_ms":samples,"samples_tokens_per_second":speeds,"mean_tokens_per_second":mean,"stddev_tokens_per_second":stddev})
}

pub fn run(
    options: ModelOptions,
    prompt: usize,
    generate: usize,
    repetitions: usize,
    report: Option<PathBuf>,
    depth: usize,
) -> Result<()> {
    run_cases(options, &[(prompt, generate, depth)], repetitions, report)
}

pub fn suite(options: ModelOptions, repetitions: usize, report: Option<PathBuf>) -> Result<()> {
    run_cases(
        options,
        &[(512, 0, 0), (4096, 0, 0), (0, 128, 0), (0, 128, 2048)],
        repetitions,
        report,
    )
}

pub fn contexts(options: ModelOptions, repetitions: usize, report: Option<PathBuf>) -> Result<()> {
    run_cases(
        options,
        &[
            (512, 0, 0),
            (1024, 0, 0),
            (4096, 0, 0),
            (0, 128, 512),
            (0, 128, 1024),
            (0, 128, 4096),
        ],
        repetitions,
        report,
    )
}

fn run_cases(
    options: ModelOptions,
    cases: &[(usize, usize, usize)],
    repetitions: usize,
    report: Option<PathBuf>,
) -> Result<()> {
    ensure!(
        repetitions > 0
            && cases
                .iter()
                .all(|&(prompt, generate, _)| prompt > 0 || generate > 0),
        "benchmark needs repetitions and at least one phase"
    );
    let capacity = cases
        .iter()
        .map(|&(prompt, generate, depth)| prompt.max(generate).saturating_add(depth))
        .max()
        .unwrap_or(0);
    ensure!(
        capacity <= options.context,
        "benchmark exceeds context capacity"
    );
    let _lock = BoardLock::acquire()?;
    let mut model = model::load(options)?;
    let mut rng = Mt19937::new(42);
    let tokens: Vec<_> = (0..capacity)
        .map(|_| rng.next() % model.info().vocab_size as u32)
        .collect();
    let mut results = Vec::new();
    for &(prompt, generate, depth) in cases {
        model.reserve(prompt.max(generate) + depth)?;
        eprintln!("bench: pp{prompt} tg{generate} depth{depth}, warmup");
        measure_phases(model.as_mut(), &tokens, prompt, generate, depth)?;
        let mut pp = Vec::new();
        let mut tg = Vec::new();
        for repetition in 0..repetitions {
            let (prefill, decode) =
                measure_phases(model.as_mut(), &tokens, prompt, generate, depth)?;
            eprintln!(
                "bench: sample {}/{repetitions}, prefill {prefill:.3} ms, decode {decode:.3} ms",
                repetition + 1
            );
            pp.push(prefill);
            tg.push(decode);
        }
        results.push(json!({"prompt":prompt,"generate":generate,"depth":depth,
            "prefill":if prompt>0 {summary(&pp,prompt)} else {serde_json::Value::Null},
            "decode":if generate>0 {summary(&tg,generate)} else {serde_json::Value::Null},"stats":model.stats()}));
    }
    let mut result = json!({"model":model.info(),"warmup":1,"repetitions":repetitions,
        "timing":"independent pp and tg; complete synchronous model calls including output head; excludes load, preparation, untimed depth prefill, tokenization, sampling",
        "seed":42,"token_source":"mt19937(seed) % vocab_size","synthetic_token_ids":tokens,
        "cases":results,"stats":model.stats()});
    if cases.len() == 1 {
        for key in ["prompt", "generate", "depth", "prefill", "decode"] {
            result[key] = results[0][key].clone();
        }
    }
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(path) = report {
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&result)?),
        )?;
    }
    Ok(())
}

fn measure_phases(
    model: &mut dyn LanguageModel,
    tokens: &[u32],
    prompt: usize,
    generate: usize,
    depth: usize,
) -> Result<(f64, f64)> {
    let mut pp = 0.0;
    if prompt > 0 {
        model.reset()?;
        if depth > 0 {
            model.prepare(depth)?;
            model.forward(&tokens[..depth])?;
        }
        model.prepare(prompt)?;
        let start = Instant::now();
        model.forward(&tokens[depth..depth + prompt])?;
        pp = start.elapsed().as_secs_f64() * 1000.0;
    }
    let mut tg = 0.0;
    if generate > 0 {
        model.reset()?;
        if depth > 0 {
            model.prepare(depth)?;
            model.forward(&tokens[..depth])?;
        }
        model.prepare(1)?;
        let start = Instant::now();
        for &token in &tokens[depth..depth + generate] {
            model.forward(&[token])?;
        }
        tg = start.elapsed().as_secs_f64() * 1000.0;
    }
    Ok((pp, tg))
}

struct Mt19937 {
    state: [u32; 624],
    index: usize,
}
impl Mt19937 {
    fn new(seed: u32) -> Self {
        let mut state = [0; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }
    fn next(&mut self) -> u32 {
        if self.index == 624 {
            for i in 0..624 {
                let bits = (self.state[i] & 0x80000000) | (self.state[(i + 1) % 624] & 0x7fffffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (bits >> 1)
                    ^ if bits & 1 != 0 { 0x9908b0df } else { 0 };
            }
            self.index = 0;
        }
        let mut v = self.state[self.index];
        self.index += 1;
        v ^= v >> 11;
        v ^= (v << 7) & 0x9d2c5680;
        v ^= (v << 15) & 0xefc60000;
        v ^= v >> 18;
        v
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn synthetic_ids_follow_std_mt19937() {
        let mut rng = super::Mt19937::new(5489);
        let values: Vec<_> = (0..5).map(|_| rng.next()).collect();
        assert_eq!(
            values,
            vec![3499211612, 581869302, 3890346734, 3586334585, 545404204]
        );
    }
}
