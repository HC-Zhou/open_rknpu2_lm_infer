//! Explicit C7 residual-basis adapter supplied alongside the checkpoint.
use anyhow::{Result, ensure};
use serde::Deserialize;
use std::path::Path;

// torch.randint(0,2,[2560], Generator().manual_seed(42)), four low-to-high bits
// per hex digit. The dense oracle is in generate_mtp_adapter_fixture.py.
const SIGNS: &str = "22247df371c7ba106faba4ff7ef5b656700d128ac56e0442832d105cbcfb524162ecba712e98d798d33dca85639ccb5720b73fb1e79ae3a9e3355aaaba42a84b6f59b89aa9af1f4aed59ece8b72ba8cc7194c9c87f96ee6bcbee50269626f1b8166c9d20271646b2a42ca04d9a53a91c3662ef0894d58925485cf7ff7614395c78a12140d01815897d76b1181ff165b07dda402f7261ea8732a9c22ebb509e881e1d8adac790dee20de5bb23dbacb487ee0031f56c7170e9ee400636c3c410f9647261e8e76b36f555954e121898a594661c34e982a3c5977a077c90160adfeac284745cc5d77d2d0b72e63ec0d12df341d86ea521d80686f401e50fa4a3a507435a6139a4d378369a3321f721c585b2b98596281d551694831e6ed263d3ce80cc01e4e961a765b5efadee99242db00cd1e1918b4e678c1bb997b6e2d76855ad";

#[derive(Deserialize)]
struct Metadata {
    rotation: String,
    seed: u64,
    original_final_norm_gain: Vec<f32>,
}
pub(super) struct Adapter {
    gamma: Vec<f32>,
    signs: Vec<f32>,
    paley: [[f32; 20]; 20],
}
impl Adapter {
    pub fn load(path: &Path, width: usize) -> Result<Option<Self>> {
        let path = path.join("mtp_coordinate_adapter.json");
        if !path.exists() {
            return Ok(None);
        }
        let m: Metadata = serde_json::from_slice(&std::fs::read(path)?)?;
        ensure!(
            width == 2560 && m.rotation == "signed Hadamard 20x128" && m.seed == 42,
            "unsupported MTP coordinate adapter"
        );
        ensure!(
            m.original_final_norm_gain.len() == width
                && m.original_final_norm_gain
                    .iter()
                    .all(|x| x.is_finite() && *x != 0.),
            "invalid MTP original norm gain"
        );
        let signs = SIGNS
            .bytes()
            .flat_map(|c| {
                let n = (c as char).to_digit(16).unwrap();
                (0..4).map(move |i| if n & (1 << i) != 0 { 1. } else { -1. })
            })
            .collect();
        let mut paley = [[1.; 20]; 20];
        for (i, row) in paley.iter_mut().enumerate().skip(1) {
            for (j, value) in row.iter_mut().enumerate().skip(1) {
                let residue = (i + 19 - j) % 19;
                *value = if (1..19).any(|k| k * k % 19 == residue) {
                    1.
                } else {
                    -1.
                };
            }
        }
        Ok(Some(Self {
            gamma: m.original_final_norm_gain,
            signs,
            paley,
        }))
    }
    fn rotate(&self, values: &mut [f32], inverse: bool) {
        let scale = (2560f32).sqrt().recip();
        for row in values.chunks_exact_mut(2560) {
            if !inverse {
                for (x, s) in row.iter_mut().zip(&self.signs) {
                    *x *= s;
                }
            }
            for block in row.chunks_exact_mut(128) {
                let mut step = 1;
                while step < 128 {
                    for group in block.chunks_exact_mut(2 * step) {
                        for j in 0..step {
                            let (a, b) = (group[j], group[j + step]);
                            group[j] = a + b;
                            group[j + step] = a - b;
                        }
                    }
                    step *= 2;
                }
            }
            for k in 0..128 {
                let column: [f32; 20] = std::array::from_fn(|i| row[i * 128 + k]);
                for j in 0..20 {
                    let sum: f32 = (0..20)
                        .map(|i| {
                            column[i]
                                * if inverse {
                                    self.paley[j][i]
                                } else {
                                    self.paley[i][j]
                                }
                        })
                        .sum();
                    row[j * 128 + k] =
                        sum * scale * if inverse { self.signs[j * 128 + k] } else { 1. };
                }
            }
        }
    }
    pub fn embedding(&self, values: &mut [f32]) {
        self.rotate(values, true);
    }
    pub fn target_hidden(&self, values: &mut [f32]) {
        self.rotate(values, true);
        for row in values.chunks_exact_mut(2560) {
            for (x, g) in row.iter_mut().zip(&self.gamma) {
                *x *= g;
            }
        }
    }
    pub fn head_input(&self, values: &mut [f32]) {
        for row in values.chunks_exact_mut(2560) {
            for (x, g) in row.iter_mut().zip(&self.gamma) {
                *x /= g;
            }
        }
        self.rotate(values, false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fast_transform_matches_independent_dense_matrix_and_round_trip() -> Result<()> {
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../../../../tests/fixtures/mtp/adapter.json"))?;
        let dir = tempfile::tempdir()?;
        let metadata = serde_json::json!({"rotation":"signed Hadamard 20x128","seed":42,
            "original_final_norm_gain":vec![1.;2560]});
        std::fs::write(
            dir.path().join("mtp_coordinate_adapter.json"),
            serde_json::to_vec(&metadata)?,
        )?;
        let a = Adapter::load(dir.path(), 2560)?.unwrap();
        let input: Vec<_> = (0..2560)
            .map(|i| ((i * 17 % 107) as f32 - 53.) / 32.)
            .collect();
        for (name, inverse) in [("forward", false), ("inverse", true)] {
            let mut actual = input.clone();
            a.rotate(&mut actual, inverse);
            for (i, v) in expected["indices"]
                .as_array()
                .unwrap()
                .iter()
                .zip(expected[name].as_array().unwrap())
            {
                assert!(
                    (actual[i.as_u64().unwrap() as usize] as f64 - v.as_f64().unwrap()).abs()
                        < 2e-6
                );
            }
            a.rotate(&mut actual, !inverse);
            assert!(input.iter().zip(&actual).all(|(a, b)| (a - b).abs() < 3e-6));
        }
        Ok(())
    }
}
