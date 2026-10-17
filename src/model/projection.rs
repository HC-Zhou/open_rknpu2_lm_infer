use super::{
    checkpoint::{QuantLoader, concatenate},
    cpu::Cpu,
};
use crate::npu::{Backend, Linear};
use anyhow::Result;
use std::sync::Arc;

pub(crate) struct Projection {
    parts: Vec<(Linear, usize)>,
    width: usize,
}

impl Projection {
    pub(crate) fn scratch_width(&self) -> usize {
        if self.parts.len() <= 1 {
            0
        } else {
            self.parts
                .iter()
                .map(|(linear, _)| linear.n())
                .max()
                .unwrap_or(0)
        }
    }
    pub(crate) fn load(
        backend: &Backend,
        loader: &Arc<QuantLoader>,
        cpu: &Cpu,
        modules: Vec<(String, usize)>,
        k: usize,
    ) -> Result<Self> {
        let mut result = Self {
            parts: Vec::new(),
            width: 0,
        };
        let mut first = 0;
        while first < modules.len() {
            let bits = loader.bits(&modules[first].0)?;
            let mut last = first + 1;
            while last < modules.len() && loader.bits(&modules[last].0)? == bits {
                last += 1;
            }
            let group = modules[first..last].to_vec();
            let reload_loader = loader.clone();
            let reload_group = group.clone();
            let reload_pool = cpu.pool.clone();
            let reload = move || {
                reload_pool.install(|| {
                    concatenate(
                        reload_group
                            .iter()
                            .map(|(name, n)| reload_loader.load(name, *n, k))
                            .collect::<Result<Vec<_>>>()?,
                    )
                })
            };
            let matrix = cpu.pool.install(|| {
                concatenate(
                    group
                        .iter()
                        .map(|(name, n)| loader.load(name, *n, k))
                        .collect::<Result<Vec<_>>>()?,
                )
            })?;
            let n = matrix.n;
            let linear = backend.load(matrix, Some(Box::new(reload)))?;
            result.parts.push((linear, result.width));
            result.width += n;
            first = last;
        }
        Ok(result)
    }
    pub(crate) fn run(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
        scratch: &mut [f32],
    ) -> Result<()> {
        if self.parts.len() == 1 {
            return self.parts[0].0.run(input, output, rows);
        }
        for (linear, offset) in &self.parts {
            let n = linear.n();
            if rows == 1 {
                linear.run(input, &mut output[*offset..*offset + n], 1)?;
            } else {
                linear.run(input, &mut scratch[..rows * n], rows)?;
                for r in 0..rows {
                    output[r * self.width + offset..r * self.width + offset + n]
                        .copy_from_slice(&scratch[r * n..(r + 1) * n]);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn run_swiglu(
        &self,
        input: &[f32],
        output: &mut [f32],
        rows: usize,
    ) -> Result<bool> {
        if self.parts.len() != 1 {
            return Ok(false);
        }
        self.parts[0].0.run_swiglu(input, output, rows)
    }
}
