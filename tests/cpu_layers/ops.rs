use super::support::{close, fixtures, series};
use open_rknpu2_lm_infer::{ops::causal_conv::ConvState, ops::norm, ops::rope::MRope};

#[test]
fn zero_centered_norm_and_interleaved_partial_rope_match_transformers() {
    let f = fixtures().ops;
    let mut output = vec![0.0; f.norm_input.len()];
    norm::rms_norm(&f.norm_input, &f.norm_weight, 1e-6, true, &mut output).unwrap();
    close(&output, &f.norm_output);
    let rope = MRope::new(f.head_dim, f.rotary_dim, 10000.0, [2, 2, 2], true).unwrap();
    let mut input = f.rope_input;
    rope.apply(&mut input, f.heads, &f.positions).unwrap();
    close(&input, &f.rope_output);
}

#[test]
fn convolution_chunks_and_history_only_replay_match_full_call() {
    for kernel in [1, 2, 4, 7] {
        let channels = 5;
        let input = series(channels * 33, 13, 0.7);
        let weights = series(channels * kernel, 29, 0.2);
        let mut all = ConvState::new(channels, kernel).unwrap();
        let mut chunked = all.clone();
        let mut replayed = all.clone();
        let mut output = vec![0.0; input.len()];
        all.run(&input, &weights, &mut output).unwrap();
        let mut actual = vec![0.0; input.len()];
        for t in 0..33 {
            let start = t * channels;
            chunked
                .run(
                    &input[start..start + channels],
                    &weights,
                    &mut actual[start..start + channels],
                )
                .unwrap();
        }
        assert_eq!(actual, output);
        replayed.append(&input).unwrap();
        assert_eq!(all, chunked);
        assert_eq!(all, replayed);
    }
}

#[test]
fn parallel_causal_convolution_matches_nonzero_history_and_chunk_boundaries() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for channels in [1, 5, 16, 8192] {
        for kernel in [1, 2, 4, 7] {
            let input = series(channels * 33, 13, 0.7);
            let weights = series(channels * kernel, 29, 0.2);
            let mut reference = ConvState::new(channels, kernel).unwrap();
            reference.append(&series(channels * 9, 37, 0.3)).unwrap();
            let mut parallel = reference.clone();
            let mut chunked = reference.clone();
            let mut expected = vec![0.0; input.len()];
            reference.run(&input, &weights, &mut expected).unwrap();
            let mut actual = vec![0.0; input.len()];
            parallel
                .run_parallel(&pool, &input, &weights, &mut actual)
                .unwrap();
            close(&actual, &expected);
            assert_eq!(parallel, reference);
            let mut offset = 0;
            for rows in [1, 2, 3, 5, 17, 5] {
                let end = offset + rows * channels;
                chunked
                    .run_parallel(
                        &pool,
                        &input[offset..end],
                        &weights,
                        &mut actual[offset..end],
                    )
                    .unwrap();
                offset = end;
            }
            close(&actual, &expected);
            assert_eq!(chunked, reference);
        }
    }
}
