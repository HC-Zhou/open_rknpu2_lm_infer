use super::*;
use half::f16;
fn pool() -> ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
}
#[test]
fn finite_fp16_conversion_round_trips_every_bit_pattern() {
    let input: Vec<_> = (0..=u16::MAX)
        .filter(|bits| bits & 0x7c00 != 0x7c00)
        .collect();
    assert_eq!(narrow(&widen(&input)), input);
}

#[test]
fn fp16_parallel_operations_match_across_thread_counts() -> Result<()> {
    let single = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let parallel = pool();
    let input: Vec<_> = (0..129 * 1024)
        .map(|i| f16::from_f32(((i * 37 % 257) as f32 - 128.) * 0.013).to_bits())
        .collect();
    for approximate in [false, true] {
        let mut a = input.clone();
        let mut b = a.clone();
        gelu(&single, &mut a, approximate);
        gelu(&parallel, &mut b, approximate);
        assert_eq!(a, b);
        add(&parallel, &mut b, &input)?;
        for ((x, y), actual) in a.iter().zip(&input).zip(b) {
            assert_eq!(
                actual,
                f16::from_f32(f16::from_bits(*x).to_f32() + f16::from_bits(*y).to_f32()).to_bits()
            );
        }
    }
    let weight: Vec<_> = (0..1024).map(|i| 0.5 + i as f32 / 2048.).collect();
    let bias: Vec<_> = (0..1024).map(|i| (i as f32 - 512.) / 4096.).collect();
    let mut a = vec![0; input.len()];
    let mut b = a.clone();
    layer_norm(&single, &input, &mut a, &weight, &bias, 1e-6)?;
    layer_norm(&parallel, &input, &mut b, &weight, &bias, 1e-6)?;
    assert_eq!(a, b);
    let positions = crate::vision::math::patch_positions(16, 16);
    let rotary = Rotary::new(16, 64, &positions)?;
    let mut a: Vec<_> = input.iter().copied().cycle().take(256 * 1024).collect();
    let mut b = a.clone();
    rotary.apply(&single, &mut a)?;
    rotary.apply(&parallel, &mut b)?;
    assert_eq!(a, b);
    Ok(())
}
#[test]
fn fp16_rotary_matches_scalar_rounding() -> Result<()> {
    let positions = [[0, 0], [127, 83], [3, 97]];
    for heads in [1, 2, 16] {
        for dim in [4, 16, 32, 64] {
            let input: Vec<_> = (0..positions.len() * heads * dim)
                .map(|i| f16::from_f32(((i * 37 % 257) as f32 - 128.) * 0.013).to_bits())
                .collect();
            let mut reference = input.clone();
            let mut actual = input;
            crate::vision::math::rotary_half(&mut reference, heads, dim, &positions)?;
            Rotary::new(heads, dim, &positions)?.apply(&pool(), &mut actual)?;
            assert_eq!(actual, reference, "heads={heads},dim={dim}");
        }
    }
    Ok(())
}
#[test]
fn fp16_norm_keeps_large_reductions_finite() -> Result<()> {
    for width in [7, 64, 1024] {
        let input: Vec<_> = (0..3 * width)
            .map(|i| {
                if i < width {
                    300.
                } else {
                    ((i * 17 % 113) as f32 - 56.) * 8.
                }
            })
            .collect();
        let stored = narrow(&input);
        assert_eq!(widen(&stored), input);
        let weights = vec![1.; width];
        let bias = vec![0.; width];
        let mut output = vec![0; input.len()];
        layer_norm(&pool(), &stored, &mut output, &weights, &bias, 1e-6)?;
        for (x, y) in input.chunks(width).zip(widen(&output).chunks(width)) {
            let mean = x.iter().map(|v| *v as f64).sum::<f64>() / width as f64;
            let variance = x.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / width as f64;
            for (a, b) in x.iter().zip(y) {
                let expected = (*a as f64 - mean) / (variance + 1e-6).sqrt();
                assert!(b.is_finite() && (*b as f64 - expected).abs() < 0.002);
            }
        }
    }
    Ok(())
}
#[test]
fn fp16_gelu_and_residual_match_rounded_reference() -> Result<()> {
    let input: Vec<_> = (0..259).map(|i| (i as f32 - 129.) / 16.).collect();
    for approximate in [false, true] {
        let mut values = narrow(&input);
        gelu(&pool(), &mut values, approximate);
        for (&x, y) in input.iter().zip(widen(&values)) {
            let reference = if approximate {
                0.5 * x * (1. + (0.7978846 * (x + 0.044715 * x * x * x)).tanh())
            } else {
                unsafe extern "C" {
                    fn erff(x: f32) -> f32;
                }
                0.5 * x * (1. + unsafe { erff(x * std::f32::consts::FRAC_1_SQRT_2) })
            };
            assert!((y - f16::from_f32(reference).to_f32()).abs() < 0.004);
        }
        let original = widen(&values);
        let rhs = narrow(&input);
        add(&pool(), &mut values, &rhs)?;
        for ((a, b), result) in original.iter().zip(&input).zip(widen(&values)) {
            assert_eq!(result, f16::from_f32(a + b).to_f32());
        }
    }
    Ok(())
}
