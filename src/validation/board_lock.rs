use anyhow::{Context, Result, ensure};
use std::{
    fs::{File, OpenOptions},
    os::fd::{AsRawFd, FromRawFd},
};

/// Coordinate exclusive board access for tests and benchmarks only.
pub struct BoardLock {
    _file: File,
}
impl BoardLock {
    pub fn acquire() -> Result<Self> {
        let inherited = std::env::var("OPEN_RKNPU2_LOCK_FD")
            .or_else(|_| std::env::var("LLM_INFER_LOCK_FD"))
            .ok();
        let file = if let Some(value) = inherited {
            let fd: i32 = value
                .parse()
                .context("invalid inherited NPU lock descriptor")?;
            let duplicate = unsafe { libc::dup(fd) };
            ensure!(
                duplicate >= 0,
                "cannot duplicate NPU lock descriptor: {}",
                std::io::Error::last_os_error()
            );
            unsafe { File::from_raw_fd(duplicate) }
        } else {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/rknpu_lowlevel_submit.lock")?
        };
        let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        ensure!(
            status == 0,
            "NPU is in use: /tmp/rknpu_lowlevel_submit.lock"
        );
        Ok(Self { _file: file })
    }
}
