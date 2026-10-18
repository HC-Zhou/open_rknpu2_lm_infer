//! Axial vision positions and learned-grid interpolation.
use anyhow::{Result, ensure};
use half::f16;
pub fn patch_positions(h: usize, w: usize) -> Vec<[usize; 2]> {
    let mut out = Vec::with_capacity(h * w);
    for bh in 0..h / 2 {
        for bw in 0..w / 2 {
            for ih in 0..2 {
                for iw in 0..2 {
                    out.push([bh * 2 + ih, bw * 2 + iw]);
                }
            }
        }
    }
    out
}
pub fn add_interpolated_position(
    hidden: &mut [f32],
    table: &[f32],
    width: usize,
    side: usize,
    h: usize,
    w: usize,
) -> Result<()> {
    ensure!(
        h > 1
            && w > 1
            && h.is_multiple_of(2)
            && w.is_multiple_of(2)
            && table.len() == side * side * width
            && hidden.len() == h * w * width,
        "vision position shape mismatch"
    );
    for (row, [y, x]) in hidden.chunks_mut(width).zip(patch_positions(h, w)) {
        let fy = y as f32 * (side - 1) as f32 / (h - 1) as f32;
        let fx = x as f32 * (side - 1) as f32 / (w - 1) as f32;
        let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
        let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
        let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
        let taps = [
            (y0 * side + x0, (1.0 - dy) * (1.0 - dx)),
            (y0 * side + x1, (1.0 - dy) * dx),
            (y1 * side + x0, dy * (1.0 - dx)),
            (y1 * side + x1, dy * dx),
        ];
        for j in 0..width {
            row[j] += taps
                .iter()
                .map(|(p, a)| table[p * width + j] * a)
                .sum::<f32>();
        }
    }
    Ok(())
}
pub fn rotary(
    values: &mut [f32],
    heads: usize,
    dim: usize,
    positions: &[[usize; 2]],
) -> Result<()> {
    ensure!(
        dim > 0 && dim.is_multiple_of(4) && values.len() == positions.len() * heads * dim,
        "vision rotary shape mismatch"
    );
    for (row, p) in values.chunks_mut(heads * dim).zip(positions) {
        for i in 0..dim / 2 {
            let axis = i / (dim / 4);
            let j = i % (dim / 4);
            let freq = 1.0 / 10000f32.powf((2 * j) as f32 / (dim / 2) as f32);
            let (sin, cos) = (p[axis] as f32 * freq).sin_cos();
            for head in row.chunks_mut(dim) {
                let (a, b) = (head[i], head[i + dim / 2]);
                head[i] = a * cos - b * sin;
                head[i + dim / 2] = b * cos + a * sin;
            }
        }
    }
    Ok(())
}
pub fn add_interpolated_position_half(
    hidden: &mut [u16],
    table: &[f32],
    width: usize,
    side: usize,
    h: usize,
    w: usize,
) -> Result<()> {
    ensure!(
        h > 1
            && w > 1
            && h.is_multiple_of(2)
            && w.is_multiple_of(2)
            && table.len() == side * side * width
            && hidden.len() == h * w * width,
        "vision position shape mismatch"
    );
    for (row, [y, x]) in hidden.chunks_mut(width).zip(patch_positions(h, w)) {
        let fy = y as f32 * (side - 1) as f32 / (h - 1) as f32;
        let fx = x as f32 * (side - 1) as f32 / (w - 1) as f32;
        let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
        let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
        let (dy, dx) = (fy - y0 as f32, fx - x0 as f32);
        let taps = [
            (y0 * side + x0, (1. - dy) * (1. - dx)),
            (y0 * side + x1, (1. - dy) * dx),
            (y1 * side + x0, dy * (1. - dx)),
            (y1 * side + x1, dy * dx),
        ];
        for j in 0..width {
            let position = taps
                .iter()
                .map(|(p, a)| table[p * width + j] * a)
                .sum::<f32>();
            row[j] = f16::from_f32(f16::from_bits(row[j]).to_f32() + position).to_bits();
        }
    }
    Ok(())
}
#[cfg(test)]
pub fn rotary_half(
    values: &mut [u16],
    heads: usize,
    dim: usize,
    positions: &[[usize; 2]],
) -> Result<()> {
    ensure!(
        dim > 0 && dim.is_multiple_of(4) && values.len() == positions.len() * heads * dim,
        "vision rotary shape mismatch"
    );
    for (row, p) in values.chunks_mut(heads * dim).zip(positions) {
        for i in 0..dim / 2 {
            let axis = i / (dim / 4);
            let j = i % (dim / 4);
            let freq = 1. / 10000f32.powf((2 * j) as f32 / (dim / 2) as f32);
            let (sin, cos) = (p[axis] as f32 * freq).sin_cos();
            for head in row.chunks_mut(dim) {
                let (a, b) = (
                    f16::from_bits(head[i]).to_f32(),
                    f16::from_bits(head[i + dim / 2]).to_f32(),
                );
                head[i] = f16::from_f32(a * cos - b * sin).to_bits();
                head[i + dim / 2] = f16::from_f32(b * cos + a * sin).to_bits();
            }
        }
    }
    Ok(())
}
