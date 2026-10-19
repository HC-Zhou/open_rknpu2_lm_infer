use anyhow::Result;
use open_rknpu2_lm_infer::{
    input::PreparedInput,
    vision::{ImageProcessor, PreparedImage},
};
use serde_json::json;
use std::sync::Arc;

fn processor() -> Result<(tempfile::TempDir, ImageProcessor)> {
    let dir = tempfile::tempdir()?;
    let config: serde_json::Value = serde_json::from_str(include_str!(
        "../llm_generated_docs/evidence/qwen35_mtp/official_config.json"
    ))?;
    std::fs::write(dir.path().join("config.json"), serde_json::to_vec(&config)?)?;
    std::fs::write(
        dir.path().join("preprocessor_config.json"),
        serde_json::to_vec(
            &json!({"patch_size":16,"merge_size":2,"temporal_patch_size":2,"size":{"shortest_edge":65536,"longest_edge":16777216},"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5]}),
        )?,
    )?;
    let processor = ImageProcessor::load(dir.path())?.unwrap();
    Ok((dir, processor))
}
#[test]
fn multi_image_expansion_uses_spatial_axes_then_compressed_text_positions() -> Result<()> {
    let (_dir, p) = processor()?;
    let images = vec![
        Arc::new(PreparedImage {
            identity: [1; 32],
            grid: [1, 4, 6],
            patches: vec![0.0; 4 * 6 * 1536].into(),
        }),
        Arc::new(PreparedImage {
            identity: [2; 32],
            grid: [1, 4, 4],
            patches: vec![0.0; 4 * 4 * 1536].into(),
        }),
    ];
    let tokens = vec![
        10,
        p.vision_start,
        p.image_token,
        p.vision_end,
        11,
        p.vision_start,
        p.image_token,
        p.vision_end,
    ];
    let input = PreparedInput::expand(tokens, images, &p)?;
    assert_eq!(input.tokens.len(), 16);
    assert_eq!(input.next_position, 11);
    assert_eq!(
        &input.positions[2..8],
        &[
            [2, 2, 2],
            [2, 2, 3],
            [2, 2, 4],
            [2, 3, 2],
            [2, 3, 3],
            [2, 3, 4]
        ]
    );
    assert_eq!(input.positions[8], [5; 3]);
    assert_eq!(input.positions[9], [6; 3]);
    assert_eq!(
        &input.positions[11..15],
        &[[8, 8, 8], [8, 8, 9], [8, 9, 8], [8, 9, 9]]
    );
    assert_eq!(input.positions[15], [10; 3]);
    assert!(PreparedInput::expand(vec![p.image_token], vec![], &p).is_err());
    assert!(PreparedInput::expand(vec![p.video_token], vec![], &p).is_err());
    Ok(())
}
#[test]
fn malformed_sources_and_extreme_dimensions_fail_before_device_work() -> Result<()> {
    let (_dir, p) = processor()?;
    assert!(p.prepare_url("https://example.org/photo.png").is_err());
    assert!(p.prepare_url("data:image/png;base64,!!!").is_err());
    assert!(p.prepare(b"not an image").is_err());
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::RgbImage::new(1, 4096).write_to(&mut encoded, image::ImageFormat::Png)?;
    assert!(p.prepare(encoded.get_ref()).is_err());
    Ok(())
}

#[test]
fn image_resolution_follows_the_checkpoint_pixel_budget() -> Result<()> {
    let (dir, p) = processor()?;
    let mut encoded = std::io::Cursor::new(Vec::new());
    image::RgbImage::new(1024, 768).write_to(&mut encoded, image::ImageFormat::Png)?;
    let image = p.prepare(encoded.get_ref())?;
    assert_eq!(image.grid, [1, 48, 64]);
    assert_eq!(image.tokens(), 768);
    image.validate()?;

    let path = dir.path().join("preprocessor_config.json");
    let mut config: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    config["size"]["longest_edge"] = (512 * 512).into();
    std::fs::write(path, serde_json::to_vec(&config)?)?;
    let limited = ImageProcessor::load(dir.path())?.unwrap();
    let image = limited.prepare(encoded.get_ref())?;
    assert_eq!(image.grid, [1, 26, 36]);
    assert_eq!(image.tokens(), 234);
    Ok(())
}

#[test]
fn jpeg_camera_orientation_precedes_resize_and_patchification() -> Result<()> {
    let (_dir, p) = processor()?;
    let image = p.prepare(include_bytes!("fixtures/vision/jpeg-orientation.jpg"))?;
    assert!(image.grid[1] > image.grid[2]);
    // EXIF=6 rotates the source's left red half to the top, blue to the bottom.
    assert!(image.patches[0] > 0.9 && image.patches[1024] < -0.9);
    let last = image.patches.len() - 1536;
    assert!(image.patches[last] < -0.9 && image.patches[last + 1024] > 0.9);
    Ok(())
}

#[test]
fn complete_png_processor_matches_pinned_pil_patch_hashes() -> Result<()> {
    use sha2::{Digest, Sha256};
    let (_dir, p) = processor()?;
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/vision/processor-comparison.json"))?;
    for (name, bytes) in [
        ("red", include_bytes!("fixtures/vision/red.png").as_slice()),
        (
            "blue",
            include_bytes!("fixtures/vision/blue.png").as_slice(),
        ),
        (
            "resize-0",
            include_bytes!("fixtures/vision/resize-0.png").as_slice(),
        ),
    ] {
        let image = p.prepare(bytes)?;
        let mut hash = Sha256::new();
        for x in image.patches.iter() {
            hash.update(x.to_le_bytes());
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            manifest[name]["reference_patches_sha256"].as_str().unwrap()
        );
    }
    Ok(())
}

#[test]
fn vision_math_matches_pinned_transformers() -> Result<()> {
    use open_rknpu2_lm_infer::{layers::layer_norm::LayerNorm, ops::gelu, vision::math};
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/vision/cpu-fixture.json"))?;
    let values = |name: &str| -> Vec<f32> { serde_json::from_value(f[name].clone()).unwrap() };
    let compare = |a: &[f32], b: &[f32]| {
        assert_eq!(a.len(), b.len());
        let max = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(max < 2e-6, "max error {max}");
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let norm = LayerNorm {
        weight: values("norm_weight"),
        bias: values("norm_bias"),
        epsilon: 1e-6,
    };
    let input = values("norm_input");
    let mut out = vec![0.0; input.len()];
    norm.run(&pool, &input, &mut out)?;
    compare(&out, &values("norm_output"));
    let mut out = values("gelu_input");
    gelu::apply(&pool, &mut out);
    compare(&out, &values("gelu_exact"));
    let positions = math::patch_positions(2, 4);
    assert_eq!(
        positions,
        serde_json::from_value::<Vec<[usize; 2]>>(f["positions"].clone())?
    );
    let mut hidden = vec![0.0; 8 * 4];
    math::add_interpolated_position(&mut hidden, &values("table"), 4, 3, 2, 4)?;
    compare(&hidden, &values("interpolated"));
    let mut q = values("rope_input");
    math::rotary(&mut q, 2, 8, &positions)?;
    compare(&q, &values("rope_output"));
    Ok(())
}
