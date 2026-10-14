use anyhow::{Result, ensure};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            seed: None,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.temperature.is_finite() && (0.0..=2.0).contains(&self.temperature),
            "temperature must be in [0, 2]"
        );
        ensure!(
            self.top_p.is_finite() && (0.0..=1.0).contains(&self.top_p),
            "top_p must be in [0, 1]"
        );
        ensure!(
            self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p),
            "min_p must be in [0, 1]"
        );
        ensure!(
            self.repeat_penalty.is_finite() && self.repeat_penalty > 0.0,
            "repeat_penalty must be positive"
        );
        ensure!(
            self.presence_penalty.is_finite() && (-2.0..=2.0).contains(&self.presence_penalty),
            "presence_penalty must be in [-2, 2]"
        );
        ensure!(
            self.frequency_penalty.is_finite() && (-2.0..=2.0).contains(&self.frequency_penalty),
            "frequency_penalty must be in [-2, 2]"
        );
        Ok(())
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: ChaCha8Rng,
    counts: Vec<u32>,
    candidates: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams, vocab_size: usize, history: &[u32]) -> Result<Self> {
        params.validate()?;
        let rng = match params.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };
        let mut sampler = Self {
            params,
            rng,
            counts: vec![0; vocab_size],
            candidates: Vec::with_capacity(vocab_size),
        };
        for &token in history {
            sampler.observe(token);
        }
        Ok(sampler)
    }

    pub fn observe(&mut self, token: u32) {
        if let Some(count) = self.counts.get_mut(token as usize) {
            *count += 1;
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> Result<u32> {
        ensure!(
            !logits.is_empty() && logits.len() == self.counts.len(),
            "invalid logits vocabulary"
        );
        self.candidates.clear();
        for (token, &original) in logits.iter().enumerate() {
            ensure!(original.is_finite(), "nonfinite logit at token {token}");
            let count = self.counts[token];
            let mut value = original;
            if count > 0 {
                value = if value >= 0.0 {
                    value / self.params.repeat_penalty
                } else {
                    value * self.params.repeat_penalty
                };
                value -=
                    self.params.presence_penalty + self.params.frequency_penalty * count as f32;
            }
            self.candidates.push((token as u32, value));
        }
        if self.params.temperature == 0.0 {
            // Stable first-token tie break matches the greedy reference path.
            let mut best = self.candidates[0];
            for &candidate in &self.candidates[1..] {
                if candidate.1 > best.1 {
                    best = candidate;
                }
            }
            return Ok(best.0);
        }
        self.candidates
            .sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        if self.params.top_k > 0 {
            self.candidates.truncate(self.params.top_k);
        }
        let maximum = self.candidates[0].1;
        let mut sum = 0.0f64;
        for candidate in &mut self.candidates {
            candidate.1 = ((candidate.1 - maximum) / self.params.temperature).exp();
            sum += candidate.1 as f64;
        }
        let minimum = self.candidates[0].1 * self.params.min_p;
        let mut cumulative = 0.0f64;
        let mut keep = 0;
        for candidate in &self.candidates {
            if candidate.1 < minimum {
                break;
            }
            cumulative += candidate.1 as f64;
            keep += 1;
            if cumulative >= sum * self.params.top_p as f64 {
                break;
            }
        }
        self.candidates.truncate(keep.max(1));
        let total: f64 = self.candidates.iter().map(|c| c.1 as f64).sum();
        let target = self.rng.r#gen::<f64>() * total;
        let mut accum = 0.0;
        for &(token, probability) in &self.candidates {
            accum += probability as f64;
            if target < accum {
                return Ok(token);
            }
        }
        Ok(self.candidates.last().unwrap().0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn zero_top_p_keeps_the_most_likely_token() {
        let params = SamplingParams {
            temperature: 1.0,
            top_p: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params, 3, &[]).unwrap();
        assert_eq!(sampler.sample(&[1.0, 3.0, 2.0]).unwrap(), 1);
    }
    #[test]
    fn penalties_change_greedy_choice() {
        let params = SamplingParams {
            frequency_penalty: 1.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params, 3, &[0, 0]).unwrap();
        assert_eq!(sampler.sample(&[3.0, 2.0, 1.0]).unwrap(), 1);
    }
    #[test]
    fn seeded_sampling_reproduces_and_top_k_limits() {
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 2,
            seed: Some(17),
            ..Default::default()
        };
        let mut a = Sampler::new(params.clone(), 3, &[]).unwrap();
        let mut b = Sampler::new(params, 3, &[]).unwrap();
        let x: Vec<_> = (0..32)
            .map(|_| a.sample(&[2.0, 1.9, -10.0]).unwrap())
            .collect();
        let y: Vec<_> = (0..32)
            .map(|_| b.sample(&[2.0, 1.9, -10.0]).unwrap())
            .collect();
        assert_eq!(x, y);
        assert!(x.iter().all(|&token| token < 2));
        assert!(x.contains(&0) && x.contains(&1));
    }
}
