# CI and releases

CI 在 GitHub ARM64 runner 的 Debian 11 容器中构建，使用 rust-toolchain.toml 的固定版本和 Cargo.lock。每次 main push、PR 或手动运行执行格式检查、Clippy、自动测试、release 构建、CLI 启动和打包。Cargo 依赖及构建产物按 toolchain 和 lockfile 缓存。

Release CI 由 v 开头的 tag 触发，复用同一套检查和构建。tag 必须与 Cargo.toml 的版本一致；全部检查完成后发布 GitHub Release。

    git tag -s v0.1.0 -m 'Release v0.1.0'
    git push origin v0.1.0

发布包含两个附件：

| 附件 | 内容 |
|---|---|
| open_rknpu2_lm_infer-vVERSION-linux-aarch64.tar.gz | ARM64 可执行程序、使用文档、固定频率脚本、工程和第三方许可证 |
| open_rknpu2_lm_infer-vVERSION-source.tar.gz | Rust 源码、CI 定义及从上游初始化的固定算子库 submodule |

二进制使用 glibc 2.31 构建，适用于 Debian 11 或更新的 ARM64 Linux。系统需要 libstdc++6、libgomp1 和可用的 RK3588 NPU DRM 驱动。解压后通过 bin/open_rknpu2_lm_infer 运行。

源码包保留 submodule 的 Git 元数据和固定提交，可直接执行 cargo build --release --locked。依赖仍由上游仓库维护。源码与文档只收录Git跟踪的文件，本地未跟踪实验不进入附件；二进制附件包含scripts/run_board.py，以便按README显式调优和固定频率测速。

## 板端验证

GitHub runner 没有 RK3588 NPU。3项硬件测试、完整模型数值与性能在 RK3588 上运行；普通 CI 执行自动测试，并保留这3项为 ignored。发布二进制还需在板上检查启动、模型加载和固定 token 的 logits，口径见 [VALIDATION.md](VALIDATION.md)。
