use super::{decoder::Mixer, runtime::Qwen35};
use serde_json::{Value, json};

impl Qwen35 {
    pub(super) fn statistics(&self) -> Value {
        let mut result = crate::model::metrics::backend_stats(&self.backends, self.head.backend());
        let mut state_bytes = 0;
        let mut conv_bytes = 0;
        let mut log_bytes = 0;
        let mut kv_bytes = 0;
        for layer in &self.layers {
            match &layer.mixer {
                Mixer::Recurrent { state, .. } => {
                    let memory = state.memory_usage();
                    state_bytes += memory.canonical_state_bytes;
                    conv_bytes += memory.convolution_history_bytes;
                    log_bytes += memory.update_log_bytes;
                }
                Mixer::Attention { cache, .. } => {
                    kv_bytes += cache.bytes().expect("live Qwen3.5 native KV cache")
                }
            }
        }
        let fields = json!({
            "position":self.position,"layers":self.layers.len(),"architecture":"qwen3_5",
            "model_load_ms":self.load_ms,"decoder_ms":self.forward_ms,"head_ms":self.head_ms,
            "gdn_state_bytes":state_bytes,"gdn_convolution_bytes":conv_bytes,"gdn_update_log_bytes":log_bytes,
            "cpu_scratch_bytes":self.work.allocated_bytes()+self.logits.capacity()*4,"attention_cache_payload_bytes":kv_bytes,
            "mapped_weight_bytes":self.loader.store.mapped_bytes(),"gdn_backend":"cpu-fp32-cpp-neon",
            "attention_backend":"native-npu-gqa-stable-softmax","kv_dtype":"float16",
            "lm_head_weight_bits":self.head.bits(),"lm_head_backend":self.head.bits().map(|b| format!("open-rknpu2-w{b}a8")).unwrap_or_else(|| "cpu-floating".into()),
            "mtp_checkpoint_available":self.mtp_available,"speculation_enabled":false,
            "multimodal_enabled":false,"prefix_truncate":"reset/recompute for shorter nonzero prefixes",
            "trace_directory":self.trace.path(),"attention_tile":self.options.attention_tile,
            "tuning_enabled":self.options.tune,"tuning_search_enabled":self.options.tune_search,"tune_cache":self.options.tune_cache,
            "activation_quantization":"full-K maxabs/127, round-away, clamp[-127,127]"
        });
        result
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        result.as_object_mut().unwrap().extend(
            serde_json::to_value(&self.phases)
                .expect("finite stage totals")
                .as_object()
                .unwrap()
                .clone(),
        );
        result
    }
}
