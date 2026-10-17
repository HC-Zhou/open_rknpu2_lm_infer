use super::ConvState;

pub(super) fn row(conv: &ConvState, input: &[f32], weights: &[f32], t: usize, output: &mut [f32]) {
    let mut c = 0;
    #[cfg(target_arch = "aarch64")]
    if conv.kernel == 4 {
        use std::arch::aarch64::*;
        // All loads are four complete channels. The first three tokens read
        // the old channel-major history; later tokens only use contiguous input.
        unsafe {
            while c + 4 <= conv.channels {
                let w = vld4q_f32(weights.as_ptr().add(c * 4));
                let history = if t < 3 {
                    Some(vld3q_f32(conv.history.as_ptr().add(c * 3)))
                } else {
                    None
                };
                let mut value = vdupq_n_f32(0.0);
                for (k, w) in [w.0, w.1, w.2, w.3].into_iter().enumerate() {
                    let x = if t + k < 3 {
                        let h = history.unwrap();
                        [h.0, h.1, h.2][t + k]
                    } else {
                        vld1q_f32(input.as_ptr().add((t + k - 3) * conv.channels + c))
                    };
                    value = vfmaq_f32(value, x, w);
                }
                vst1q_f32(output.as_mut_ptr().add(c), value);
                c += 4;
            }
        }
    }
    for (c, out) in output.iter_mut().enumerate().skip(c) {
        let mut value = 0.0;
        for k in 0..conv.kernel {
            let x = if t + k < conv.kernel - 1 {
                conv.history[c * (conv.kernel - 1) + t + k]
            } else {
                input[(t + k - (conv.kernel - 1)) * conv.channels + c]
            };
            value += x * weights[c * conv.kernel + k];
        }
        *out = value;
    }
    crate::ops::activation::silu_in_place(output);
}
