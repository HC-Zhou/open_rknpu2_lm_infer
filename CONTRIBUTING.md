# Contributing

模型、CPU 运算、调度和用户界面使用 Rust；NPU 运算通过 `rk_npu_matmul` 公共 C ABI 接入。新增模型实现 `LanguageModel` 并在 `model::load` 注册。

算子库通过 `native/rk_npu_matmul` Git submodule 固定 [上游提交](https://github.com/happyme531/rknpu2_matmul_open)。克隆后运行 `git submodule update --init --depth 1`。算子实现和硬件布局调度维护在上游库中；本地联调可通过 `RK_NPU_MATMUL_ROOT` 指定 checkout。

在 RK3588 Linux 板上验证代码修改：

```sh
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked
```

涉及 NPU、KV 或模型前向的修改还需运行相关板端算子测试，并使用固定 token IDs 比较模型 logits。板端测试与测速共享 `/tmp/rknpu_lowlevel_submit.lock`；固定频率和恢复由 `scripts/run_board.py` 管理。

性能报告包含模型格式、上下文、chunk、CPU/NPU mask、固定频率、调优缓存状态及原始样本。分别报告 prefill、decode 和历史 KV 深度；测速包含完整模型调用和输出头。结果与复现方式见 [VALIDATION.md](llm_generated_docs/VALIDATION.md)。

PR 说明具体行为变化和验证结果。提交仅包含相关源码、文档与小型证据；模型文件、构建产物和实验目录由 `.gitignore` 排除。贡献按工程的 AGPL-3.0-only 许可证提供，第三方来源保留原许可和通知。
