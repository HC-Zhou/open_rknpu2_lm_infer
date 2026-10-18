//! Qwen3.5 hybrid text model: config, checkpoint mapping and decoder orchestration.
mod attention;
pub mod config;
mod decoder;
mod head;
mod load;
mod multimodal;
mod rope;
mod runtime;
mod stats;
mod timings;
pub mod vision;
pub mod weights;

pub use runtime::Qwen35;
