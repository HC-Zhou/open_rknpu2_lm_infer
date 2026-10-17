//! Qwen3.5 interleaved query/output-gate projection layout.
use anyhow::{Result, ensure};

/// q_proj stores [query_head_0, gate_head_0, query_head_1, gate_head_1, ...].
pub fn split_query_gate(
    projected: &[f32],
    head_dim: usize,
    query: &mut [f32],
    gate: &mut [f32],
) -> Result<()> {
    ensure!(
        head_dim > 0
            && query.len().is_multiple_of(head_dim)
            && query.len() == gate.len()
            && query.len().checked_mul(2) == Some(projected.len()),
        "query/gate shape mismatch"
    );
    for ((qg, q), g) in projected
        .chunks(2 * head_dim)
        .zip(query.chunks_mut(head_dim))
        .zip(gate.chunks_mut(head_dim))
    {
        q.copy_from_slice(&qg[..head_dim]);
        g.copy_from_slice(&qg[head_dim..]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::activation;
    #[test]
    fn per_head_query_gate_split_and_sigmoid_are_distinct_from_gdn_silu() {
        let qg = [1.0, 2.0, 30.0, 40.0, 5.0, 6.0, 70.0, 80.0];
        let mut q = [0.0; 4];
        let mut gate = q;
        split_query_gate(&qg, 2, &mut q, &mut gate).unwrap();
        assert_eq!(q, [1.0, 2.0, 5.0, 6.0]);
        assert_eq!(gate, [30.0, 40.0, 70.0, 80.0]);
        assert_eq!(activation::sigmoid(-1000.0), 0.0);
        assert_eq!(activation::sigmoid(1000.0), 1.0);
        assert_eq!(activation::softplus(1000.0), 1000.0);
        let mut attention = [2.0; 3];
        activation::sigmoid_gate(&mut attention, &[-1000.0, 0.0, 1000.0]).unwrap();
        assert_eq!(attention, [0.0, 1.0, 2.0]);
    }
}
