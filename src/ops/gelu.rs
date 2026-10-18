//! Exact erf GELU required by the vision merger.
use rayon::{ThreadPool, prelude::*};
unsafe extern "C" {
    fn erff(x: f32) -> f32;
}
#[inline]
pub(super) fn erf(x: f32) -> f32 {
    unsafe { erff(x) }
}
pub fn apply(pool: &ThreadPool, values: &mut [f32]) {
    pool.install(|| {
        values.par_iter_mut().for_each(|x| {
            *x = 0.5 * *x * (1.0 + erf(*x * std::f32::consts::FRAC_1_SQRT_2));
        })
    });
}
