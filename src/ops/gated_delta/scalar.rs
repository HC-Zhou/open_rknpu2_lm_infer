use super::{GdnDimensions, Updates};
use anyhow::{Result, ensure};

/// Scalar oracle with the same state layout and explicit FP32 operation order.
pub fn run_scalar(
    dim: GdnDimensions,
    state: &mut [f32],
    query: &[f32],
    updates: Updates<'_>,
    output: &mut [f32],
) -> Result<()> {
    dim.validate_updates(state, updates)?;
    ensure!(
        query.len() == updates.rows * dim.key_width()
            && output.len() == updates.rows * dim.value_width(),
        "GDN query/output shape mismatch"
    );
    ensure!(query.iter().all(|v| v.is_finite()), "nonfinite GDN query");
    for t in 0..updates.rows {
        for h in 0..dim.value_heads {
            let kh = h / (dim.value_heads / dim.key_heads);
            let offset = t * dim.key_width() + kh * dim.key_dim;
            let q = &query[offset..offset + dim.key_dim];
            let k = &updates.key[offset..offset + dim.key_dim];
            let decay = updates.log_decay[t * dim.value_heads + h].exp();
            let beta = updates.beta[t * dim.value_heads + h];
            for j in 0..dim.value_dim {
                let base = (h * dim.value_dim + j) * dim.key_dim;
                let s = &mut state[base..base + dim.key_dim];
                let mut predicted = 0.0;
                for (s, k) in s.iter_mut().zip(k) {
                    *s *= decay;
                    predicted += *s * k;
                }
                let delta = (updates.value[t * dim.value_width() + h * dim.value_dim + j]
                    - predicted)
                    * beta;
                let mut result = 0.0;
                for ((s, k), q) in s.iter_mut().zip(k).zip(q) {
                    *s += delta * k;
                    result += *s * q;
                }
                output[t * dim.value_width() + h * dim.value_dim + j] =
                    result / (dim.key_dim as f32).sqrt();
            }
        }
    }
    Ok(())
}
