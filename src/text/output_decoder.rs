//! Keep only the explicitly enabled protocol's special markers in generated text.
use std::collections::HashSet;
use tokenizers::{
    Tokenizer, decoders::DecoderWrapper, models::ModelWrapper, normalizers::NormalizerWrapper,
    pre_tokenizers::PreTokenizerWrapper, processors::PostProcessorWrapper, tokenizer::DecodeStream,
};

pub(crate) struct OutputDecoder<'a> {
    inner: DecodeStream<
        'a,
        ModelWrapper,
        NormalizerWrapper,
        PreTokenizerWrapper,
        PostProcessorWrapper,
        DecoderWrapper,
    >,
    excluded: HashSet<u32>,
}
impl<'a> OutputDecoder<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, preserve_tool_markers: bool) -> Self {
        let excluded = if preserve_tool_markers {
            tokenizer
                .get_added_tokens_decoder()
                .into_iter()
                .filter(|(_, token)| {
                    token.special
                        && !matches!(
                            token.content.as_str(),
                            "<function" | "</function>" | "<param" | "</param>"
                        )
                })
                .map(|(id, _)| id)
                .collect()
        } else {
            HashSet::new()
        };
        Self {
            inner: tokenizer.decode_stream(!preserve_tool_markers),
            excluded,
        }
    }
    pub(crate) fn step(&mut self, token: u32) -> tokenizers::Result<Option<String>> {
        if self.excluded.contains(&token) {
            return Ok(None);
        }
        self.inner.step(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires MINICPM5_MODEL_PATH with real tokenizer.json"]
    fn real_minicpm5_tokenizer_retains_only_xml_markers() {
        let path = std::path::PathBuf::from(
            std::env::var("MINICPM5_MODEL_PATH").expect("set MINICPM5_MODEL_PATH"),
        );
        let tokenizer = Tokenizer::from_file(path.join("tokenizer.json")).unwrap();
        let text = "<function name=\"f\"><param name=\"城市\">北京</param></function>";
        let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let mut with_special = vec![tokenizer.token_to_id("<|im_start|>").unwrap()];
        with_special.extend(&ids);
        with_special.push(tokenizer.token_to_id("<|im_end|>").unwrap());
        for enabled in [false, true] {
            let mut decoder = OutputDecoder::new(&tokenizer, enabled);
            let result: String = with_special
                .iter()
                .filter_map(|&id| decoder.step(id).unwrap())
                .collect();
            if enabled {
                assert_eq!(result, text);
            } else {
                assert_eq!(result, tokenizer.decode(&with_special, true).unwrap());
            }
            assert!(!result.contains("<|im_"));
        }
    }
}
