use anyhow::{Context, Result};
use std::{io::Write, path::PathBuf};

pub(crate) struct Trace(Option<PathBuf>);

impl Trace {
    #[cfg(test)]
    pub fn in_directory(path: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&path)?;
        Ok(Self(Some(path)))
    }
    pub fn from_environment() -> Result<Self> {
        let directory = std::env::var_os("OPEN_RKNPU2_TRACE_DIR")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from);
        if let Some(path) = &directory {
            std::fs::create_dir_all(path)?;
        }
        Ok(Self(directory))
    }
    pub fn enabled(&self) -> bool {
        self.0.is_some()
    }
    pub fn path(&self) -> Option<&std::path::Path> {
        self.0.as_deref()
    }
    pub fn save(&self, position: usize, layer: i32, stage: &str, values: &[f32]) -> Result<()> {
        let Some(directory) = &self.0 else {
            return Ok(());
        };
        let path = directory.join(format!("p{position}-l{layer}-{stage}.f32"));
        let mut file = std::fs::File::create(&path)
            .with_context(|| format!("write trace {}", path.display()))?;
        let bytes = unsafe {
            std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
        };
        file.write_all(bytes)?;
        Ok(())
    }
    pub fn save_half(
        &self,
        position: usize,
        layer: i32,
        stage: &str,
        values: &[u16],
    ) -> Result<()> {
        if self.enabled() {
            self.save(
                position,
                layer,
                stage,
                &crate::ops::vision_half::widen(values),
            )?;
        }
        Ok(())
    }
}
