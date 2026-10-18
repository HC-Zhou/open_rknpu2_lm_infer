//! Reusable affine LayerNorm, with population variance in FP32.
use anyhow::{Result, ensure};
use rayon::{ThreadPool, prelude::*};
pub struct LayerNorm {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub epsilon: f32,
}
impl LayerNorm {
    pub fn run_half(&self, pool: &ThreadPool, input: &[u16], output: &mut [u16]) -> Result<()> {
        crate::ops::vision_half::layer_norm(
            pool,
            input,
            output,
            &self.weight,
            &self.bias,
            self.epsilon,
        )
    }
    pub fn run(&self, pool: &ThreadPool, input: &[f32], output: &mut [f32]) -> Result<()> {
        let width = self.weight.len();
        ensure!(
            width > 0
                && self.bias.len() == width
                && input.len() == output.len()
                && input.len().is_multiple_of(width)
                && self.epsilon.is_finite()
                && self.epsilon > 0.0,
            "LayerNorm shape/epsilon mismatch"
        );
        pool.install(|| {
            input
                .par_chunks(width)
                .zip(output.par_chunks_mut(width))
                .for_each(|(x, y)| {
                    let mean = x.iter().sum::<f32>() / width as f32;
                    let variance =
                        x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / width as f32;
                    let scale = (variance + self.epsilon).sqrt().recip();
                    for j in 0..width {
                        y[j] = (x[j] - mean) * scale * self.weight[j] + self.bias[j];
                    }
                })
        });
        Ok(())
    }
}
