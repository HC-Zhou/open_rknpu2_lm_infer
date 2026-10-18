use super::*;

#[test]
#[ignore = "requires exclusive RK3588 NPU access"]
fn floating_profiles_preserve_weights_and_reload_without_search() -> Result<()> {
    let _lock = crate::validation::BoardLock::acquire()?;
    let cache = tempfile::tempdir()?;
    let mut options = ModelOptions {
        path: std::path::PathBuf::new(),
        context: 256,
        chunk: 128,
        threads: 4,
        npu_mask: 7,
        cpu_mask: 0xf0,
        tune: true,
        tune_search: true,
        tune_cache: cache.path().into(),
        attention_tile: 512,
        vision_dtype: Default::default(),
    };
    configure_cpu(&mut options)?;
    for dtype in [RK_NPU_MM_F16, RK_NPU_MM_BF16] {
        let device = Backend::new(&options, 12)?;
        let backend = FloatBackend::with_dtype(device.clone(), dtype);
        let round = |x| {
            if dtype == RK_NPU_MM_BF16 {
                half::bf16::from_f32(x).to_f32()
            } else {
                f16::from_f32(x).to_f32()
            }
        };
        let (k, n) = (96, 160);
        let weights: Vec<_> = (0..k * n)
            .map(|i| round(((i * 13 % 71) as f32 - 35.0) * 0.007))
            .collect();
        let bias = vec![0.007; n];
        let linear = backend.load(&weights, &bias, k, n)?;
        let mut expected_outputs = Vec::new();
        for rows in [3, 129] {
            linear.prepare(rows)?;
            let input: Vec<_> = (0..rows * k)
                .map(|i| ((i * 7 % 83) as f32 - 41.0) * 0.01)
                .collect();
            let mut output = vec![0.0; rows * n];
            linear.run(&input, &mut output, rows)?;
            for r in 0..rows {
                for j in 0..n {
                    let expected: f32 = (0..k)
                        .map(|i| round(input[r * k + i]) * weights[j * k + i])
                        .sum::<f32>()
                        + bias[j];
                    assert!(
                        (output[r * n + j] - expected).abs()
                            < if dtype == RK_NPU_MM_BF16 { 0.02 } else { 0.004 }
                    );
                }
            }
            expected_outputs.push((rows, input, output));
        }
        assert_eq!(device.stats()["tune_calls"], 2);
        drop(linear);
        drop(backend);
        drop(device);
        let mut load_options = options.clone();
        load_options.tune_search = false;
        let device = Backend::new(&load_options, 12)?;
        let backend = FloatBackend::with_dtype(device.clone(), dtype);
        let linear = backend.load(&weights, &bias, k, n)?;
        for (rows, input, expected) in expected_outputs {
            let mut output = vec![0.0; expected.len()];
            linear.run(&input, &mut output, rows)?;
            assert_eq!(output, expected);
        }
        assert_eq!(device.stats()["tune_calls"], 0);
        assert_eq!(device.stats()["tune_cache_hits"], 2);
    }
    Ok(())
}
