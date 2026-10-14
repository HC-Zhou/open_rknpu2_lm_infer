use crate::{
    engine::BoardLock,
    model::{self, ModelOptions},
};
use anyhow::{Result, ensure};
use serde_json::json;
use std::{
    io::{BufWriter, Write},
    path::PathBuf,
};

pub fn run(
    options: ModelOptions,
    tokens_file: PathBuf,
    continuation_file: Option<PathBuf>,
    dump_logits: PathBuf,
    report: Option<PathBuf>,
) -> Result<()> {
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(tokens_file)?)?;
    let continuation: Vec<u32> = match continuation_file {
        Some(path) => serde_json::from_slice(&std::fs::read(path)?)?,
        None => vec![],
    };
    ensure!(
        !tokens.is_empty() && tokens.len() + continuation.len() <= options.context,
        "evaluation input exceeds context or is empty"
    );
    let _lock = BoardLock::acquire()?;
    let mut model = model::load(options)?;
    model.reserve(tokens.len() + continuation.len())?;
    model.prepare(tokens.len())?;
    let mut output = BufWriter::new(std::fs::File::create(&dump_logits)?);
    let mut save = |logits: &[f32]| -> Result<()> {
        for value in logits {
            output.write_all(&value.to_le_bytes())?;
        }
        Ok(())
    };
    save(model.forward(&tokens)?)?;
    for token in &continuation {
        save(model.forward(&[*token])?)?;
    }
    output.flush()?;
    let shape = json!({"shape":[1+continuation.len(),model.info().vocab_size],"dtype":"float32","byte_order":"little"});
    std::fs::write(
        format!("{}.shape.json", dump_logits.display()),
        format!("{}\n", serde_json::to_string_pretty(&shape)?),
    )?;
    let result = json!({"prompt_tokens":tokens.len(),"continuation_tokens":continuation.len(),"logits":dump_logits,"stats":model.stats()});
    println!("{}", serde_json::to_string_pretty(&result)?);
    if let Some(path) = report {
        std::fs::write(
            path,
            format!("{}\n", serde_json::to_string_pretty(&result)?),
        )?;
    }
    Ok(())
}
