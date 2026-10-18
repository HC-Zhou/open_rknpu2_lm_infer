//! Separable antialiased bicubic resize, matching Pillow's RGB integer path.
//! Cubic coefficient -0.5; quantized 22-bit weights and intermediate u8 rows.
use image::RgbImage;

fn cubic(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}
fn coefficients(source: usize, target: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = source as f64 / target as f64;
    let filter = scale.max(1.0);
    let support = 2.0 * filter;
    (0..target)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(source);
            let weights: Vec<_> = (first..end)
                .map(|j| cubic((j as f64 + 0.5 - center) / filter))
                .collect();
            let sum: f64 = weights.iter().sum();
            let fixed = weights
                .iter()
                .map(|w| {
                    let x = w / sum * ((1u64 << 22) as f64);
                    (x + if x < 0.0 { -0.5 } else { 0.5 }) as i32
                })
                .collect();
            (first, fixed)
        })
        .collect()
}
pub(super) fn bicubic(image: RgbImage, width: u32, height: u32) -> RgbImage {
    let (sw, sh) = image.dimensions();
    let horizontal = if sw == width {
        image
    } else {
        let coeffs = coefficients(sw as usize, width as usize);
        RgbImage::from_fn(width, sh, |x, y| {
            let (first, weights) = &coeffs[x as usize];
            image::Rgb(std::array::from_fn(|c| {
                let sum = weights.iter().enumerate().fold(1i64 << 21, |s, (i, w)| {
                    s + *w as i64 * image.get_pixel((*first + i) as u32, y)[c] as i64
                });
                (sum >> 22).clamp(0, 255) as u8
            }))
        })
    };
    if sh == height {
        return horizontal;
    }
    let coeffs = coefficients(sh as usize, height as usize);
    RgbImage::from_fn(width, height, |x, y| {
        let (first, weights) = &coeffs[y as usize];
        image::Rgb(std::array::from_fn(|c| {
            let sum = weights.iter().enumerate().fold(1i64 << 21, |s, (i, w)| {
                s + *w as i64 * horizontal.get_pixel(x, (*first + i) as u32)[c] as i64
            });
            (sum >> 22).clamp(0, 255) as u8
        }))
    })
}
