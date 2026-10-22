#[path = "text/output_decoder.rs"]
mod output_decoder;
use crate::engine::Prompt;
use anyhow::{Context, Result, anyhow, ensure};
use minijinja::{Environment, context};
pub(crate) use output_decoder::OutputDecoder;
use serde_json::Value;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct TextCodec {
    pub tokenizer: Tokenizer,
    template: String,
    eos_token: Value,
    bos_token: Value,
    images: Option<crate::vision::ImageProcessor>,
}

// Transformers templates can pass ensure_ascii to tojson. Delegate all existing
// behavior (including indentation and HTML escaping) to MiniJinja unchanged.
fn template_tojson(
    value: &minijinja::Value,
    indent: Option<minijinja::Value>,
    args: minijinja::value::Kwargs,
) -> std::result::Result<minijinja::Value, minijinja::Error> {
    let ensure_ascii: Option<bool> = args.get("ensure_ascii")?;
    let rendered = minijinja::filters::tojson(value, indent, args)?;
    if ensure_ascii != Some(true) {
        return Ok(rendered);
    }
    let mut ascii = String::new();
    for ch in rendered.as_str().unwrap_or("").chars() {
        if ch.is_ascii() {
            ascii.push(ch);
        } else {
            use std::fmt::Write;
            for unit in ch.encode_utf16(&mut [0; 2]) {
                write!(ascii, "\\u{unit:04x}").unwrap();
            }
        }
    }
    Ok(minijinja::Value::from_safe_string(ascii))
}

impl TextCodec {
    pub fn load(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer: {e}"))?;
        let config: Value =
            serde_json::from_slice(&std::fs::read(path.join("tokenizer_config.json"))?)?;
        let template_path = path.join("chat_template.jinja");
        let template = if template_path.exists() {
            std::fs::read_to_string(template_path)?
        } else {
            config["chat_template"]
                .as_str()
                .context("checkpoint needs chat_template.jinja or a tokenizer_config.json chat_template string")?
                .to_owned()
        };
        Ok(Self {
            tokenizer,
            template,
            eos_token: config["eos_token"].clone(),
            bos_token: config["bos_token"].clone(),
            images: crate::vision::ImageProcessor::load(path)?,
        })
    }

    pub fn format(&self, prompt: &Prompt) -> Result<String> {
        match prompt {
            Prompt::Text(text) => Ok(text.clone()),
            Prompt::Tokens(_) => anyhow::bail!("token IDs have no chat template"),
            Prompt::Chat {
                messages,
                tools,
                enable_thinking,
            } => {
                ensure!(!messages.is_empty(), "messages must not be empty");
                let mut env = Environment::new();
                env.add_filter("tojson", template_tojson);
                env.set_trim_blocks(true);
                env.set_lstrip_blocks(true);
                env.set_unknown_method_callback(
                    minijinja_contrib::pycompat::unknown_method_callback,
                );
                env.add_function(
                    "raise_exception",
                    |message: String| -> std::result::Result<String, minijinja::Error> {
                        Err(minijinja::Error::new(
                            minijinja::ErrorKind::InvalidOperation,
                            message,
                        ))
                    },
                );
                env.add_template("chat", &self.template)?;
                let tools = tools
                    .as_ref()
                    .map(minijinja::Value::from_serialize)
                    .unwrap_or(minijinja::Value::UNDEFINED);
                Ok(env.get_template("chat")?.render(context! {
                    messages => messages, tools => tools, add_generation_prompt => true,
                    enable_thinking => enable_thinking, eos_token => self.eos_token, bos_token => self.bos_token,
                })?)
            }
        }
    }

    pub fn encode(&self, prompt: &Prompt) -> Result<Vec<u32>> {
        if let Prompt::Tokens(tokens) = prompt {
            return Ok(tokens.clone());
        }
        let formatted = self.format(prompt)?;
        let encoding = self
            .tokenizer
            .encode(formatted, false)
            .map_err(|e| anyhow!("encode prompt: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn prepare(&self, prompt: &Prompt) -> Result<crate::input::PreparedInput> {
        let images = match prompt {
            Prompt::Chat { messages, .. } => {
                crate::input::message_images(messages, self.images.as_ref())?
            }
            _ => Vec::new(),
        };
        let tokens = self.encode(prompt)?;
        if let Some(processor) = &self.images {
            crate::input::PreparedInput::expand(tokens, images, processor)
        } else {
            Ok(crate::input::PreparedInput::text(tokens))
        }
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.tokenizer
            .decode(tokens, skip_special_tokens)
            .map_err(|e| anyhow!("decode: {e}"))
    }
}

/// Delay only suffixes that could become a stop sequence in the next chunk.
/// Complete stops are removed even when a tokenizer chunk spans the stop.
pub struct StopFilter {
    stops: Vec<String>,
    pending: String,
    stopped: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Result<Self> {
        ensure!(
            stops.iter().all(|s| !s.is_empty()),
            "stop sequences must not be empty"
        );
        Ok(Self {
            stops,
            pending: String::new(),
            stopped: false,
        })
    }
    pub fn push(&mut self, chunk: &str) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.pending.push_str(chunk);
        if let Some(position) = self
            .stops
            .iter()
            .filter_map(|stop| self.pending.find(stop))
            .min()
        {
            let output = self.pending[..position].to_owned();
            self.pending.clear();
            self.stopped = true;
            return (output, true);
        }
        let mut retained = 0;
        for stop in &self.stops {
            for (length, _) in stop.char_indices().skip(1) {
                if length > retained && self.pending.ends_with(&stop[..length]) {
                    retained = length;
                }
            }
        }
        let split = self.pending.len() - retained;
        let tail = self.pending.split_off(split);
        let output = std::mem::replace(&mut self.pending, tail);
        (output, false)
    }
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn tojson_supports_ascii_keyword_without_changing_existing_templates() {
        let mut env = Environment::new();
        env.add_filter("tojson", template_tojson);
        let value = serde_json::json!({"text":"北京😀<&'", "items":[1,true]});
        for template in [
            "{{ v|tojson }}",
            "{{ v|tojson(indent=2) }}",
            "{{ v|tojson(2) }}",
        ] {
            let actual = env.render_str(template, context! { v => &value }).unwrap();
            let expected = Environment::new()
                .render_str(template, context! { v => &value })
                .unwrap();
            assert_eq!(actual, expected);
        }
        let unicode = env
            .render_str(
                "{{ v|tojson(ensure_ascii=False) }}",
                context! { v => &value },
            )
            .unwrap();
        assert!(unicode.contains("北京😀"));
        let ascii = env
            .render_str(
                "{{ v|tojson(indent=2, ensure_ascii=True) }}",
                context! { v => &value },
            )
            .unwrap();
        assert!(ascii.is_ascii());
        assert!(ascii.contains("\\ud83d\\ude00"));
        assert_eq!(serde_json::from_str::<Value>(&ascii).unwrap(), value);
        assert!(
            env.render_str("{{ v|tojson(unknown=True) }}", context! { v => &value })
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with the real template and tokenizer"]
    fn real_minicpm5_template_renders_tools_and_history() {
        let path = std::path::PathBuf::from(
            std::env::var("MINICPM5_MODEL_PATH").expect("set MINICPM5_MODEL_PATH"),
        );
        let codec = TextCodec::load(&path).unwrap();
        let rendered = codec.format(&Prompt::Chat {
            messages: vec![
                serde_json::json!({"role":"user","content":"查询北京天气"}),
                serde_json::json!({"role":"assistant","content":"", "tool_calls":[{"id":"c0","type":"function","function":{"name":"capture", "arguments":{"city":"北京"}}}]}),
                serde_json::json!({"role":"tool", "tool_call_id":"c0", "content":"晴"}),
            ],
            tools: Some(serde_json::json!([{"type":"function", "function":{"name":"capture", "description":"查询天气", "parameters":{"type":"object"}}}])),
            enable_thinking: false,
        }).unwrap();
        assert!(rendered.contains("查询天气"));
        assert!(
            rendered.contains(
                "<function name=\"capture\"><param name=\"city\">北京</param></function>"
            )
        );
        assert!(rendered.contains("<tool_response>\n晴\n</tool_response>"));
    }

    use super::*;
    #[test]
    fn native_template_tools_and_whitespace_rules() {
        let codec = TextCodec {
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            template: "{% if tools is not defined %}{% set tools = [] %}{% endif %}\n{% if tools is iterable and tools | length > 0 %}tools{% else %}plain{% endif %}".into(),
            eos_token: Value::Null,
            bos_token: Value::Null,
            images: None,
        };
        assert_eq!(
            codec
                .format(&Prompt::Chat {
                    messages: vec![serde_json::json!({"role":"user","content":"你好"})],
                    tools: None,
                    enable_thinking: false
                })
                .unwrap(),
            "plain"
        );
    }
    #[test]
    fn stops_across_chunks_without_leaking_prefixes() {
        let mut filter = StopFilter::new(vec!["结束".into(), "END".into()]).unwrap();
        assert_eq!(filter.push("你好结"), ("你好".into(), false));
        assert_eq!(filter.push("束剩余"), ("".into(), true));
        assert_eq!(filter.finish(), "");
    }
    #[test]
    fn partial_stop_is_flushed_at_length_limit() {
        let mut filter = StopFilter::new(vec!["END".into()]).unwrap();
        assert_eq!(filter.push("hello E"), ("hello ".into(), false));
        assert_eq!(filter.finish(), "E");
    }
}
