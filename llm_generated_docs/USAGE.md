# Usage

所有命令通过 `--model` 或 `RKNPU_MODEL` 指定 checkpoint 目录。运行用户需要有权限打开 NPU DRM 设备。

## 命令

| 命令 | 功能 |
|---|---|
| `inspect` | 查看模型元数据，不打开 NPU |
| `tokenize` | 编码文本或解码 token IDs，不打开 NPU |
| `run` | 流式生成；省略 prompt 时从 stdin 读取 |
| `chat` | 交互终端（TUI） |
| `serve` | OpenAI 兼容 HTTP 服务 |
| `bench` | 测量 prefill 与带历史的 decode |
| `eval` | 导出固定 token 序列的 FP32 logits |
| `ppl` | 对独立 token 窗口计算 PPL |

```sh
export RKNPU_MODEL=/path/to/C7-Rotate-QQQ

target/release/open_rknpu2_lm_infer inspect
target/release/open_rknpu2_lm_infer tokenize '你好' --chat-template
target/release/open_rknpu2_lm_infer run '你好，请介绍一下自己。' \
  --chat-template --temperature 0 -n 128 --stats
target/release/open_rknpu2_lm_infer chat
target/release/open_rknpu2_lm_infer serve --host 127.0.0.1 --port 8080
```

`run` 默认读取原始文本，`--chat-template` 将它作为用户消息。聊天模板读取 checkpoint 的 `chat_template.jinja` 或 `tokenizer_config.json`；`--thinking` 开启 thinking。

TUI 支持 Enter 发送、Shift+Enter 或 Alt+Enter 换行、Esc 取消生成、`/reset` 清空会话、PageUp/PageDown 滚动和 Ctrl+C 退出。

## 运行参数

| 参数 | 默认值 |
|---|---|
| `--context` | 4096 |
| `--chunk` | 64（prefill 块大小） |
| `--threads` | 4 |
| `--cpu-mask` | `0xf0` |
| `--npu-mask` | 7 |

算子库自动将 Attention 的实际核数限制在 KV head 数以内，并按硬件布局拆分逻辑 query 行。MiniCPM5 默认使用两核 Attention，Linear 仍使用三核。

OpenMP 等待默认配置为 `OMP_WAIT_POLICY=PASSIVE`、`GOMP_SPINCOUNT=0`。程序仅补充未设置的变量，保留用户显式设置的值；这些配置在 libgomp 初始化时生效。

首次加载和首次使用新的 batch bucket 会进行调优，结果保存至 `$XDG_CACHE_HOME/open_rknpu2_lm_infer` 或 `~/.cache/open_rknpu2_lm_infer`。`--tune-cache` 指定目录，`--no-tune` 使用确定的默认策略。

## HTTP 服务

服务默认仅监听本机。`--api-key` 或 `OPEN_RKNPU_API_KEY` 启用 Bearer token。单个模型实例由专用线程顺序执行请求；有界队列最多等待4个请求，满时返回503。与当前 KV 相同的 token 前缀会被复用。路由和参数支持范围见 [API.md](API.md)。

## 测速与评测

```sh
target/release/open_rknpu2_lm_infer bench --prompt 512 --generate 128 \
  --repetitions 3 --report perf.json
target/release/open_rknpu2_lm_infer bench --suite representative \
  --repetitions 3 --report representative.json
target/release/open_rknpu2_lm_infer --context 8192 bench --suite contexts \
  --repetitions 3 --report contexts.json
```

`bench` 独立测量 pp 和 tg；`--depth N` 在计时前建立 N-token KV 历史，默认为0。测速包含完整同步模型调用、输出头和 CPU 运算，排除加载、调优、历史建立、分词和采样。使用 `std::mt19937(42) % vocab_size` 对应的固定 IDs，报告原始样本、均值、标准差和资源统计。正式对照需要固定 NPU/DDR/CPU 频率，并保持模型、算子库、上下文和调优策略一致。

`--suite contexts` 测量空 KV 的 pp512/1024/4096，以及分别带512/1024/4096-token 历史的 tg128；每项预热一次。最大上下文容量至少为4224，以容纳4096-token 历史与128步 decode。

`eval --tokens-file IDS.json --continuation-file MORE.json --dump-logits logits.f32` 导出 prompt 末端与 teacher-forced continuation 的 FP32 logits，用于独立数值比较。

`ppl --tokens-file WINDOWS.json --report ppl.json` 对独立、等长的 token 窗口计算 next-token PPL。输入为二维 token ID 数组；默认从零起始 target 位置257评分至窗口末尾，对应512-token窗口的后255个 target。模型按 chunk 批量执行 NPU 输出头，NLL 使用 FP32 shifted exp 与 FP64 sum/log，报告逐 token NLL、每窗口耗时和累计 PPL。报告每窗口保存一次；同一命令可继续未完成的评测。

固定频率与板锁由 `scripts/run_board.py` 管理，用法见 [PERFORMANCE.md](PERFORMANCE.md)。

## 独立源码包

`python3 scripts/package_source.py --output /path/to/new-directory` 生成可独立发布的源码目录，并从 [rknpu2_matmul_open](https://github.com/happyme531/rknpu2_matmul_open) 初始化固定提交的 Git submodule。需要 Git 和网络连接。

克隆项目后先运行 `git submodule update --init --depth 1`，再运行 `cargo build --release --locked`。`RK_NPU_MATMUL_ROOT` 可显式指定本地算子库 checkout，用于开发联调。
