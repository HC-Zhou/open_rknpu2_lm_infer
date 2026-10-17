use open_rknpu2_lm_infer::{
    layers::gated_delta::{MixerWeights, Projected},
    ops::gated_delta::GdnDimensions,
};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct Fixtures {
    pub gdn: GdnFixture,
    pub mixer: MixerFixture,
    pub ops: OpsFixture,
}

#[derive(Deserialize)]
pub struct GdnFixture {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub log_decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub initial_state: Vec<f32>,
    pub output: Vec<f32>,
    pub state: Vec<f32>,
}

#[derive(Deserialize)]
pub struct MixerFixture {
    pub rows: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub conv_weight: Vec<f32>,
    pub a_log: Vec<f32>,
    pub dt_bias: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub output: Vec<f32>,
    pub state: Vec<f32>,
    pub history: Vec<f32>,
}

impl MixerFixture {
    pub fn dim(&self) -> GdnDimensions {
        GdnDimensions::new(
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
        )
        .unwrap()
    }
    pub fn weights(&self) -> MixerWeights {
        MixerWeights::new(
            self.dim(),
            self.kernel,
            self.conv_weight.clone(),
            self.a_log.clone(),
            self.dt_bias.clone(),
            self.norm_weight.clone(),
            1e-6,
        )
        .unwrap()
    }
    pub fn input(&self) -> Projected<'_> {
        Projected {
            rows: self.rows,
            qkv: &self.qkv,
            z: &self.z,
            a: &self.a,
            b: &self.b,
        }
    }
}

#[derive(Deserialize)]
pub struct OpsFixture {
    pub norm_input: Vec<f32>,
    pub norm_weight: Vec<f32>,
    pub norm_output: Vec<f32>,
    pub rope_input: Vec<f32>,
    pub rope_output: Vec<f32>,
    pub positions: Vec<[usize; 3]>,
    pub heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
}

pub fn fixtures() -> Fixtures {
    serde_json::from_str(include_str!(
        "../../llm_generated_docs/evidence/qwen35_mtp/cpu_fixtures.json"
    ))
    .unwrap()
}

pub fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    let max_ref = expected.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let ref_l2 = expected
        .iter()
        .map(|v| (*v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let diff_l2 = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(actual.iter().all(|v| v.is_finite()));
    assert!(
        max_error <= 1e-5 + 1e-4 * max_ref,
        "max_abs={max_error}, max_ref={max_ref}"
    );
    assert!(
        diff_l2 <= 1e-4 * ref_l2.max(1e-12),
        "relative L2={}",
        diff_l2 / ref_l2
    );
}

pub fn series(n: usize, offset: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i + offset) as f32 * 0.173).sin() * scale)
        .collect()
}

pub struct Inputs {
    dim: GdnDimensions,
    pub qkv: Vec<f32>,
    pub z: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl Inputs {
    pub fn new(dim: GdnDimensions, rows: usize, offset: usize) -> Self {
        Self {
            dim,
            qkv: series(rows * dim.conv_width(), offset, 0.7),
            z: series(rows * dim.value_width(), offset + 8, 1.0),
            a: series(rows * dim.value_heads(), offset + 19, 0.3),
            b: series(rows * dim.value_heads(), offset + 41, 1.0),
        }
    }
    pub fn view(&self, start: usize, rows: usize) -> Projected<'_> {
        let range = |width| start * width..(start + rows) * width;
        Projected {
            rows,
            qkv: &self.qkv[range(self.dim.conv_width())],
            z: &self.z[range(self.dim.value_width())],
            a: &self.a[range(self.dim.value_heads())],
            b: &self.b[range(self.dim.value_heads())],
        }
    }
}
