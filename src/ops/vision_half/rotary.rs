use super::*;

pub struct Rotary {
    cos: Vec<f32>,
    sin: Vec<f32>,
    rows: usize,
    heads: usize,
    dim: usize,
}
impl Rotary {
    pub fn new(heads: usize, dim: usize, positions: &[[usize; 2]]) -> Result<Self> {
        ensure!(
            heads > 0 && dim > 0 && dim.is_multiple_of(4),
            "invalid FP16 rotary geometry"
        );
        let half = dim / 2;
        let frequencies: Vec<_> = (0..half)
            .map(|i| 1. / 10000f32.powf((2 * (i % (dim / 4))) as f32 / half as f32))
            .collect();
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for p in positions {
            for (i, &frequency) in frequencies.iter().enumerate() {
                let (s, c) = (p[i / (dim / 4)] as f32 * frequency).sin_cos();
                cos.push(c);
                sin.push(s);
            }
        }
        Ok(Self {
            cos,
            sin,
            rows: positions.len(),
            heads,
            dim,
        })
    }
    pub fn apply(&self, pool: &ThreadPool, values: &mut [u16]) -> Result<()> {
        ensure!(
            values.len() == self.rows * self.heads * self.dim,
            "FP16 rotary tensor shape mismatch"
        );
        let half = self.dim / 2;
        let run = |index: usize, head: &mut [u16]| {
            let row = index / self.heads;
            let cos = &self.cos[row * half..(row + 1) * half];
            let sin = &self.sin[row * half..(row + 1) * half];
            #[cfg(target_arch = "aarch64")]
            let mut first = 0;
            #[cfg(not(target_arch = "aarch64"))]
            let first = 0;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                use std::arch::aarch64::*;
                while first + 8 <= half {
                    let (a, b) = convert::load8(head.as_ptr().add(first));
                    let (c, d) = convert::load8(head.as_ptr().add(half + first));
                    let cl = vld1q_f32(cos.as_ptr().add(first));
                    let ch = vld1q_f32(cos.as_ptr().add(first + 4));
                    let sl = vld1q_f32(sin.as_ptr().add(first));
                    let sh = vld1q_f32(sin.as_ptr().add(first + 4));
                    // Keep multiply/add separate, matching FP32 and scalar RoPE.
                    convert::store8(
                        head.as_mut_ptr().add(first),
                        vsubq_f32(vmulq_f32(a, cl), vmulq_f32(c, sl)),
                        vsubq_f32(vmulq_f32(b, ch), vmulq_f32(d, sh)),
                    );
                    convert::store8(
                        head.as_mut_ptr().add(half + first),
                        vaddq_f32(vmulq_f32(c, cl), vmulq_f32(a, sl)),
                        vaddq_f32(vmulq_f32(d, ch), vmulq_f32(b, sh)),
                    );
                    first += 8;
                }
            }
            for j in first..half {
                let a = f16::from_bits(head[j]).to_f32();
                let b = f16::from_bits(head[half + j]).to_f32();
                head[j] = f16::from_f32(a * cos[j] - b * sin[j]).to_bits();
                head[half + j] = f16::from_f32(b * cos[j] + a * sin[j]).to_bits();
            }
        };
        if values.len() >= PARALLEL_ELEMENTS {
            pool.install(|| {
                values
                    .par_chunks_mut(self.dim)
                    .enumerate()
                    .for_each(|(index, head)| run(index, head))
            });
        } else {
            for (index, head) in values.chunks_mut(self.dim).enumerate() {
                run(index, head);
            }
        }
        Ok(())
    }
}
