use anyhow::{Result, ensure};

/// Positions are [temporal, height, width], independent of token/KV indices.
#[derive(Debug, Clone)]
pub struct MRope {
    head_dim: usize,
    rotary_dim: usize,
    inv_freq: Vec<f32>,
    sections: [usize; 3],
    interleaved: bool,
}

impl MRope {
    pub fn new(
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
        sections: [usize; 3],
        interleaved: bool,
    ) -> Result<Self> {
        ensure!(
            rotary_dim > 0 && rotary_dim.is_multiple_of(2) && rotary_dim <= head_dim,
            "invalid partial rotary dimension"
        );
        ensure!(theta.is_finite() && theta > 0.0, "invalid RoPE theta");
        ensure!(
            sections.iter().try_fold(0usize, |a, b| a.checked_add(*b)) == Some(rotary_dim / 2),
            "M-RoPE sections must cover half the rotary dimension"
        );
        let inv_freq = (0..rotary_dim / 2)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary_dim as f32))
            .collect();
        Ok(Self {
            head_dim,
            rotary_dim,
            inv_freq,
            sections,
            interleaved,
        })
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    fn axis(&self, i: usize) -> usize {
        if self.interleaved {
            match i % 3 {
                1 if i / 3 < self.sections[1] => 1,
                2 if i / 3 < self.sections[2] => 2,
                _ => 0,
            }
        } else if i < self.sections[0] {
            0
        } else if i < self.sections[0] + self.sections[1] {
            1
        } else {
            2
        }
    }

    pub fn apply(&self, values: &mut [f32], heads: usize, positions: &[[usize; 3]]) -> Result<()> {
        ensure!(
            heads > 0
                && positions
                    .len()
                    .checked_mul(heads)
                    .and_then(|v| v.checked_mul(self.head_dim))
                    == Some(values.len()),
            "M-RoPE input/position shape mismatch"
        );
        let half = self.rotary_dim / 2;
        for (position, row) in positions
            .iter()
            .zip(values.chunks_mut(heads * self.head_dim))
        {
            for i in 0..half {
                let angle = position[self.axis(i)] as f32 * self.inv_freq[i];
                let (sine, cosine) = angle.sin_cos();
                for head in row.chunks_mut(self.head_dim) {
                    let (a, b) = (head[i], head[i + half]);
                    head[i] = a * cosine - b * sine;
                    head[i + half] = b * cosine + a * sine;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_positions_degenerate_and_preserve_unrotated_tail() {
        let rope = MRope::new(24, 12, 10000.0, [2, 2, 2], true).unwrap();
        let plain = MRope::new(24, 12, 10000.0, [2, 2, 2], false).unwrap();
        let input: Vec<f32> = (0..2 * 3 * 24).map(|i| (i as f32 * 0.19).sin()).collect();
        let mut a = input.clone();
        let mut b = input.clone();
        rope.apply(&mut a, 3, &[[71; 3], [72; 3]]).unwrap();
        plain.apply(&mut b, 3, &[[71; 3], [72; 3]]).unwrap();
        assert_eq!(a, b);
        for (rotated, original) in a.chunks(24).zip(input.chunks(24)) {
            assert_eq!(&rotated[12..], &original[12..]);
        }
        assert!(rope.apply(&mut a, 3, &[[71; 3]]).is_err());
    }
}
