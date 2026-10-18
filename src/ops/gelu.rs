//! Exact and tanh-approximate GELU, used by different vision projections.
use rayon::{ThreadPool, prelude::*};
unsafe extern "C" {
    fn erff(x: f32) -> f32;
}
#[inline]
pub(super) fn erf(x: f32) -> f32 {
    unsafe { erff(x) }
}
pub fn apply(pool: &ThreadPool, values: &mut [f32], approximate: bool) {
    pool.install(|| {
        values.par_iter_mut().for_each(|x| {
            *x = if approximate {
                0.5 * *x * (1.0 + (0.797_884_6 * (*x + 0.044715 * *x * *x * *x)).tanh())
            } else {
                0.5 * *x * (1.0 + erf(*x * std::f32::consts::FRAC_1_SQRT_2))
            };
        })
    });
}
