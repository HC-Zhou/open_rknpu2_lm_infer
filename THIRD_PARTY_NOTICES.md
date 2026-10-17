# Third-party notices

Project contributions are licensed under AGPL-3.0-only; see [LICENSE](LICENSE).

The executable statically links the separately supplied `rk_npu_matmul` library. Preserve that library's [license](native/rk_npu_matmul/LICENSE) and [third-party notices](native/rk_npu_matmul/THIRD_PARTY_NOTICES.md), including the recorded attribution and permission boundary for adapted allbilly/rk3588 code. This project does not change those permissions.

Rust dependencies are pinned in Cargo.lock. Their upstream license and copyright notices must be retained in source or binary distributions. Major dependencies include Hugging Face tokenizers and safetensors, memmap2, half, Rayon, Tokio, axum, clap, ratatui, crossterm, MiniJinja, serde, rand, tracing, bindgen and cmake. Cargo package metadata and each dependency's distributed license files define their respective terms.

The CPU kernels preserve arithmetic from the adjacent native inference demo and the open operator library. Model checkpoints and tokenizer assets are supplied separately under their own licenses. Model weights, vendor runtime binaries and reference documents are excluded from this source distribution.

The Qwen3.5 C++ CPU module in `native/cpu_kernels/` adapts the gated delta loop and NEON helpers from llama.cpp / ggml. The Rust vector SiLU also adapts ggml arithmetic. Their MIT copyright and permission notice is retained in [llm_generated_docs/THIRD_PARTY_NOTICES.md](llm_generated_docs/THIRD_PARTY_NOTICES.md) and must accompany distributions of this code.
