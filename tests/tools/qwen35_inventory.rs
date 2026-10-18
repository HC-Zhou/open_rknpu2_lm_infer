//! Print the logical text/MTP tensor contract without loading model weights.
use anyhow::Result;
use clap::Parser;
use open_rknpu2_lm_infer::layers::gated_delta::LayerState;
use open_rknpu2_lm_infer::model::qwen35::config::{Config, LayerType};
use serde_json::json;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    /// config.json or its model directory.
    config: PathBuf,
    /// Include a separately stored output head even for tied embeddings.
    #[arg(long)]
    lm_head: bool,
    #[arg(long, default_value_t = 8)]
    verify_rows: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let path = if args.config.is_dir() {
        args.config.join("config.json")
    } else {
        args.config
    };
    let value = serde_json::from_slice(&std::fs::read(path)?)?;
    let config = Config::parse(&value)?;
    let linear_layers = config
        .layers
        .iter()
        .filter(|v| **v == LayerType::LinearAttention)
        .count();
    let state = LayerState::new(config.gdn, config.conv_kernel, args.verify_rows)?;
    let memory = state.memory_usage();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "hidden_size": config.hidden_size,
            "full_attention_layers": config.layers.len() - linear_layers,
            "gdn_layers": linear_layers,
            "mtp_layers": config.mtp_layers,
            "verify_rows": args.verify_rows,
            "gdn_state_layout": "value_head,value_channel,key_channel",
            "gdn_canonical_state_bytes": memory.canonical_state_bytes * linear_layers,
            "gdn_convolution_history_bytes": memory.convolution_history_bytes * linear_layers,
            "gdn_update_log_bytes": memory.update_log_bytes * linear_layers,
            "tensors": config.text_tensors(args.lm_head),
        }))?
    );
    Ok(())
}
