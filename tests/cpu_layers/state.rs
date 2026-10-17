use super::support::{Inputs, close, fixtures, series};
use open_rknpu2_lm_infer::{
    layers::gated_delta::{LayerState, MixerWeights, Projected, Scratch},
    ops::gated_delta::GdnDimensions,
};

fn weights(dim: GdnDimensions, kernel: usize, offset: usize) -> MixerWeights {
    MixerWeights::new(
        dim,
        kernel,
        series(dim.conv_width() * kernel, offset, 0.2),
        series(dim.value_heads(), offset + 7, 0.4),
        series(dim.value_heads(), offset + 19, 0.2),
        vec![1.0; dim.value_dim()],
        1e-6,
    )
    .unwrap()
}

#[test]
fn complete_cpu_mixer_matches_pinned_transformers_fixture() {
    let f = fixtures().mixer;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut state = LayerState::new(f.dim(), f.kernel, 8).unwrap();
    let mut scratch = Scratch::default();
    let actual = state
        .forward(&pool, &f.weights(), f.input(), &mut scratch)
        .unwrap();
    close(actual, &f.output);
    close(state.canonical(), &f.state);
    assert_eq!(state.convolution(), f.history);
    assert_eq!(state.position(), f.rows);
}

#[test]
fn prefill_chunks_and_single_token_execution_have_identical_states_and_outputs() {
    let dim = GdnDimensions::new(2, 4, 8, 5).unwrap();
    let weights = weights(dim, 4, 3);
    let input = Inputs::new(dim, 129, 7);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for chunk in [1, 2, 3, 4, 5, 7, 8, 31, 32, 33, 127, 128, 129] {
        let mut all = LayerState::new(dim, 4, 8).unwrap();
        let mut pieces = LayerState::new(dim, 4, 8).unwrap();
        let mut scratch = Scratch::default();
        let expected = all
            .forward(&pool, &weights, input.view(0, 129), &mut scratch)
            .unwrap()
            .to_vec();
        let mut output = Vec::new();
        for start in (0..129).step_by(chunk) {
            output.extend_from_slice(
                pieces
                    .forward(
                        &pool,
                        &weights,
                        input.view(start, chunk.min(129 - start)),
                        &mut scratch,
                    )
                    .unwrap(),
            );
        }
        assert_eq!(output, expected);
        assert_eq!(pieces.canonical(), all.canonical());
        assert_eq!(pieces.convolution(), all.convolution());
    }
}

#[test]
fn every_acceptance_boundary_replays_only_confirmed_prefix_and_next_call_matches_recompute() {
    let dim = GdnDimensions::new(2, 4, 8, 5).unwrap();
    let weights = weights(dim, 4, 13);
    let input = Inputs::new(dim, 64, 31);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for k in [1, 2, 3, 4, 7] {
        for accepted in 0..=k {
            let mut state = LayerState::new(dim, 4, 8).unwrap();
            let mut fresh = LayerState::new(dim, 4, 8).unwrap();
            let mut scratch = Scratch::default();
            state
                .forward(&pool, &weights, input.view(0, 11), &mut scratch)
                .unwrap();
            let before = state.canonical().to_vec();
            let history = state.convolution().to_vec();
            state
                .verify(&pool, &weights, input.view(11, k + 1), &mut scratch)
                .unwrap();
            assert_eq!(state.position(), 11);
            assert_eq!(state.canonical(), before);
            assert_eq!(state.convolution(), history);
            assert!(state.finish_round(&pool, 0).is_err());
            state.finish_round(&pool, 1 + accepted).unwrap();
            assert_eq!(state.position(), 12 + accepted);
            fresh
                .forward(&pool, &weights, input.view(0, 12 + accepted), &mut scratch)
                .unwrap();
            assert_eq!(state.canonical(), fresh.canonical());
            assert_eq!(state.convolution(), fresh.convolution());
            let actual = state
                .forward(&pool, &weights, input.view(12 + accepted, 3), &mut scratch)
                .unwrap()
                .to_vec();
            let expected = fresh
                .forward(&pool, &weights, input.view(12 + accepted, 3), &mut scratch)
                .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(state.canonical(), fresh.canonical());
        }
    }
}

#[test]
fn scratch_reuse_cannot_overwrite_other_layer_log_and_abort_keeps_canonical_history() {
    let dim = GdnDimensions::new(2, 4, 7, 5).unwrap();
    let w1 = weights(dim, 4, 13);
    let w2 = weights(dim, 4, 23);
    let input = Inputs::new(dim, 31, 71);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut a = LayerState::new(dim, 4, 8).unwrap();
    let mut b = LayerState::new(dim, 4, 8).unwrap();
    let mut scratch = Scratch::default();
    for round in 0..4 {
        let start = a.position();
        let before = a.canonical().to_vec();
        a.verify(&pool, &w1, input.view(start, 8), &mut scratch)
            .unwrap();
        b.verify(&pool, &w2, input.view(start, 8), &mut scratch)
            .unwrap();
        if round % 2 == 0 {
            a.abort_round();
            b.abort_round();
            assert_eq!(a.canonical(), before);
            assert_eq!(a.position(), start);
            a.verify(&pool, &w1, input.view(start, 8), &mut scratch)
                .unwrap();
            b.verify(&pool, &w2, input.view(start, 8), &mut scratch)
                .unwrap();
        }
        let count = round + 1;
        a.finish_round(&pool, count).unwrap();
        b.finish_round(&pool, count).unwrap();
        for (state, weight) in [(&a, &w1), (&b, &w2)] {
            let mut fresh = LayerState::new(dim, 4, 8).unwrap();
            fresh
                .forward(&pool, weight, input.view(0, state.position()), &mut scratch)
                .unwrap();
            assert_eq!(state.canonical(), fresh.canonical());
            assert_eq!(state.convolution(), fresh.convolution());
        }
    }
    assert_eq!(
        a.memory_usage().update_log_bytes,
        8 * (dim.key_width() + dim.value_width() + 2 * dim.value_heads() + dim.conv_width()) * 4
    );
    let before = a.canonical().to_vec();
    let bad = Projected {
        rows: 1,
        qkv: &[],
        z: &[],
        a: &[],
        b: &[],
    };
    assert!(a.verify(&pool, &w1, bad, &mut scratch).is_err());
    assert_eq!(a.canonical(), before);
    assert_eq!(a.pending_rows(), None);
    assert!(
        a.verify(&pool, &w1, input.view(0, 9), &mut scratch)
            .is_err()
    );
    a.reset();
    let mut fresh = LayerState::new(dim, 4, 8).unwrap();
    let actual = a
        .forward(&pool, &w1, input.view(0, 7), &mut scratch)
        .unwrap()
        .to_vec();
    let expected = fresh
        .forward(&pool, &w1, input.view(0, 7), &mut scratch)
        .unwrap();
    assert_eq!(actual, expected);
}
