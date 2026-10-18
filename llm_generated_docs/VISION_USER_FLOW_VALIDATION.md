# 视觉调优、正式运行锁边界与真实图片交互

2026-10-18，RK3588 主板，C7 Qwen3.5 checkpoint；默认 FP16 ViT 中间张量、FP32 归约，算子库 `5b4e39c`。本轮不是文本精度基准，重点检查调优准备和真实图文交互。

## 用户入口

```sh
open_rknpu2_lm_infer -m MODEL tune --vision
open_rknpu2_lm_infer -m MODEL tune --vision --image photo.jpg --image other.png --report vision-tune.json
```

不传图片时准备 M256/384/1024；传图片时按模型预处理后的 patch 数去重。七类 Linear 的 K/N 来自模型配置，merger 自动使用 M/4，不加载文本或视觉 checkpoint 张量。沿用 `--tune-cache`、CPU/NPU 参数和 `--vision-dtype`；默认准备 FP16 host-storage profile，需要旧 FP32 中间张量对照时使用 `--vision-dtype fp32`。

FP16、FP32 两档分别命中21个 profile，红/蓝图片入口命中14个，三次均 `tune_calls=0`。普通推理只读缓存，miss 使用合法 baseline。测试中移除了子进程的 inherited-lock 环境变量与文件描述符，父 runner 仍持有全局板锁，正式 `tune` 依然成功。原始报告见 [FP16](evidence/vision_user_flow_20261018/tune-fp16.json)、[FP32](evidence/vision_user_flow_20261018/tune-fp32.json) 和 [图片输入](evidence/vision_user_flow_20261018/tune-images.json)。

## 锁边界

Engine 启动和正式 `tune` 不再获取 `/tmp/rknpu_lowlevel_submit.lock`。`run`、`chat`、`serve` 通过这个 Engine 执行；独立测试、eval/PPL、benchmark 仍通过 `validation::BoardLock` 协调板端使用。锁工具已移出 Engine。

真实图片服务和 PTY 测试同样由外层 runner 持锁，生产子应用不继承该描述符，也没有 `OPEN_RKNPU2_LOCK_FD` / `LLM_INFER_LOCK_FD` bypass。服务和 TUI 均成功启动，验证正式路径不依赖测试锁。

## OpenAI 服务

实际启动 `serve`，使用 localhost 随机空闲端口、context2048、CPU4/`0xf0`、NPU7、temperature0、max_tokens64、thinking关闭。图片为 Ultralytics 街景公交照片和 COCO 两猫照片，[来源记录](evidence/vision_user_flow_20261018/photo-sources.json)。请求发送真实图片的 inline PNG/JPEG base64。

| 场景 | 实际结果摘要 |
|---|---|
| 公交图识别 | 蓝色电动公交，提到 `cero emisiones` |
| 保留图片的颜色追问 | `Blue and white.` |
| 同一会话追加两猫图 | `There are two cats in the image.` |
| 保留两猫图的 SSE 颜色追问 | `Pink.`，收到 usage 与 `[DONE]` |
| 相同提示词分别换猫图/公交图 | 分别描述两猫和公交，没有复用错误图片 |
| 单请求同时传两图、SSE | 正确指出第二张为动物、第一张为公共交通 |
| 中文公交问答 | `这是一辆城市公交车。` |
| 图片请求之后的新纯文本请求 | `OK` |

九个请求均成功。完整回答、usage 与请求耗时见 [API 报告](evidence/vision_user_flow_20261018/api-final.json)。服务 SIGINT 后正常退出，exit0；正常推理没有触发调优搜索。本轮文本策略缓存未命中时使用 baseline，因此请求时延不代表充分调优后的文本性能；C7 本轮也未复用 decoder prefix tokens。

初次测试把躺卧表面限制为 `sofa/couch`，模型答 `A pink blanket.`，因此那个关键词断言未通过；[原始回答](evidence/vision_user_flow_20261018/initial-surface-description.json) 保留。之后改问表面颜色，得到 `Pink.`。公交小字 OCR 等细节也有偏差，不能把本轮粗粒度识别结果视为完整视觉精度验证。

## TUI

通过真正的 PTY 启动 `chat`，按键操作 `/image PATH`、输入提示词、追问、换图、`/reset` 和 Ctrl+C；pyte 重建实际 ANSI 屏幕。没有调用内部 Conversation 函数代替终端交互。

- 两猫图问数量：`2`；追问表面颜色：`Pink`。
- 追加公交图：描述蓝白电动公交；追问颜色：`Blue and white`。
- `/reset` 后发送中文“请只回复 OK。”：`OK`，prompt 从含图历史回到17 tokens。
- Ctrl+C 正常退出，exit0。公交描述达到测试设置的64-token上限，UI正确显示 `length`，后续追问仍正常。

五个场景通过，见 [TUI 报告](evidence/vision_user_flow_20261018/tui.json) 和 [实际屏幕](evidence/vision_user_flow_20261018/cats-followup.txt)。测试驱动使用 CR 模拟 Enter；LF 在 Crossterm raw 模式中是 Ctrl+J。

## 复现与验证

```sh
cargo fmt --check
cargo clippy --all-targets --release --locked -- -D warnings
cargo test --release --locked
cargo build --release --locked
python3 scripts/run_board.py --wait-lock --report results/api-clocks.json -- \
  python3 tests/board/vision_api.py --binary target/release/open_rknpu2_lm_infer \
  --model MODEL --photos PHOTOS --cache CACHE --output results/vision-api
python3 -m pip install --target results/tui-python -r tests/board/requirements.txt
python3 scripts/run_board.py --wait-lock --report results/tui-clocks.json -- \
  python3 tests/board/vision_tui.py --binary target/release/open_rknpu2_lm_infer \
  --model MODEL --photos PHOTOS --cache CACHE --python-deps results/tui-python --output results/vision-tui
```

PHOTOS 包含 `bus.jpg` 和 `cats.png`。Release fmt、全 target Clippy、70项默认测试通过，9项硬件专用测试按默认配置忽略。API、TUI、调优 runner 均 exit0，CPU/NPU/DDR 频率上下限及 governor 恢复且 `restore_errors=[]`。

独立工具源码已从 `examples/` 移到 `tests/tools/`，继续使用 Cargo `--example NAME` 显式构建/运行；默认测试不执行这些外部数据/设备工具。六份必需视觉 fixture 已放入 `tests/fixtures/vision/`，脱离忽略的报告目录。旧插桩、消融与历史汇总工具见 [实验归档](FP16_VIT_EXPERIMENT_ARCHIVE.md)。Rust CPU 迁移的精度、3.5%–4.8%时延回退及原始样本见 [FP16 存储报告](VISION_FP16_STORAGE.md)。
