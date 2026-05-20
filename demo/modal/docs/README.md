# Modal GRPO 工具智能体训练说明

本目录记录如何把当前 `demo` 工程的 GRPO tool-agent 训练路径部署到 Modal 算力平台。

第一版只支持 **单节点多 GPU**。检索服务和 `verl.trainer.main_ppo` 会运行在同一个 Modal GPU Function 内，因此现有工具配置里的 `http://127.0.0.1:8790` 仍然有效，不需要把 retrieval server 先拆成独立 Web 服务。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `modal_grpo_tool_agent.py` | Modal App 定义，包含镜像、Volume、输入检查、smoke 训练和正式 GRPO 训练入口。 |
| `upload_assets.ps1` | 把 parquet 数据、检索索引、BGE 模型、reranker 和 SFT merged model 上传到 Modal Volume。 |
| `download_outputs.ps1` | 从 Modal 输出 Volume 下载 GRPO checkpoint 和日志。 |
| `docs/SECRETS.md` | Modal Secret 命名和环境变量说明。 |
| `docs/modal显卡型号信息.md` | Modal 常用 GPU 型号、显存、价格和预算时长参考。 |

## Modal 资源

运行脚本使用以下 Modal 对象：

| 名称 | 类型 | 容器内挂载路径 |
| --- | --- | --- |
| `agentic-rag-rl-data` | Volume | `/vol/data` |
| `agentic-rag-rl-models` | Volume | `/vol/models` |
| `agentic-rag-rl-outputs` | Volume | `/vol/outputs` |
| `agentic-rag-rl-secrets` | Secret | 注入为环境变量 |

期望的 Volume 目录结构：

```text
/vol/data/
├── novel/indexes/
├── novel_eval/grpo_agentic_train.parquet
└── novel_eval/grpo_agentic_val.parquet

/vol/models/
├── bge-m3/
├── bge-reranker-v2-m3/
└── Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged/

/vol/outputs/
└── grpo_tool_agent_react_v4/
```

## 前置条件

本机先安装并登录 Modal CLI：

```powershell
python -m pip install modal
modal setup
```

按 [`SECRETS.md`](SECRETS.md) 创建 `agentic-rag-rl-secrets`。如果当前 GRPO 不需要真实 API Key，也建议创建一个占位 Secret，避免 Function 启动时找不到 Secret：

```powershell
modal secret create --force agentic-rag-rl-secrets PLACEHOLDER=1
```

上传前确认本地已有以下路径：

- `demo\data\novel_eval\grpo_agentic_train.parquet`
- `demo\data\novel_eval\grpo_agentic_val.parquet`
- `demo\data\novel\indexes`
- `demo\models\bge-m3`
- `demo\models\bge-reranker-v2-m3`
- `demo\models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged`

如果模型目录不在 `demo\models` 下，可以在上传脚本中用参数覆盖。

## 上传数据和模型

在仓库根目录执行：

```powershell
.\demo\modal\upload_assets.ps1
```

脚本会创建数据、模型、输出三个 Volume，并且只上传数据和模型产物，不会上传 `.env`。

如果本地模型目录在其他位置：

```powershell
.\demo\modal\upload_assets.ps1 `
  -BgeModelDir "D:\models\bge-m3" `
  -RerankerModelDir "D:\models\bge-reranker-v2-m3" `
  -MergedModelDir "D:\models\Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged"
```

## 远端输入检查

启动 GPU 训练前先检查 Volume 中的关键路径：

```powershell
modal run .\demo\modal\modal_grpo_tool_agent.py::check_inputs
```

期望结果：训练 parquet、验证 parquet、索引目录、BGE 模型、reranker 模型、merged model、项目代码和 verl package 都输出 `true`。

## Smoke 训练

先运行小规模 smoke：

```powershell
modal run --detach .\demo\modal\modal_grpo_tool_agent.py::train_smoke
```

smoke 默认参数：

- `gpu="H100:1"`
- `trainer.total_training_steps=1`
- `TRAIN_BATCH_SIZE=4`
- `PPO_MINI_BATCH_SIZE=2`
- `ROLLOUT_N=2`
- `SAVE_FREQ=1`
- `TEST_FREQ=1`

smoke 要验证四件事：

- retrieval server 的 `/health` 能通过；
- verl 能在 Ray/vLLM 下启动；
- tool-agent 能调用 `keyword_search`、`dense_search` 或 `hybrid_search`；
- `/vol/outputs/grpo_tool_agent_react_v4` 下能写出 checkpoint 或日志。

可以在 `--` 后追加临时 Hydra 覆盖参数：

```powershell
modal run --detach .\demo\modal\modal_grpo_tool_agent.py::train_smoke -- `
  actor_rollout_ref.rollout.max_model_len=3072
```

## 正式训练

正式入口是单节点 4 GPU：

```powershell
modal run --detach .\demo\modal\modal_grpo_tool_agent.py::train
```

默认参数：

- `gpu="H100:4"`
- `trainer.nnodes=1`
- `trainer.n_gpus_per_node=4`
- `TRAIN_BATCH_SIZE=32`
- `PPO_MINI_BATCH_SIZE=16`
- `ROLLOUT_N=4`
- `TOTAL_EPOCHS=2`
- `SAVE_FREQ=5`
- `TEST_FREQ=3`
- `trainer.resume_mode=auto`

临时调参同样可以追加 Hydra 覆盖：

```powershell
modal run --detach .\demo\modal\modal_grpo_tool_agent.py::train -- `
  trainer.total_epochs=1 `
  actor_rollout_ref.rollout.n=2 `
  trainer.save_freq=2
```

不要让多个训练任务同时写入同一个 `/vol/outputs/grpo_tool_agent_react_v4` 目录。Modal Volume 支持并发访问，但同一文件的并发写入会由最后写入者覆盖，容易破坏 checkpoint。

## 下载训练产物

smoke 或正式训练结束后执行：

```powershell
.\demo\modal\download_outputs.ps1
```

默认下载到：

```text
demo\training\outputs\modal_grpo_tool_agent_react_v4
```

## 运行注意事项

- Modal GPU Function 可能被抢占。当前脚本使用 `trainer.resume_mode=auto`、Modal 重试和基于 Volume 的输出目录来支持从 checkpoint 恢复。
- Modal Function 单次最长运行时间为 24 小时。更长训练应依赖 checkpoint + retry/resume，而不是假设一次进程持续运行到结束。
- 第一版刻意不使用 Modal 多节点集群。当前项目主线是 `trainer.nnodes=1`，先把单节点 4/8 GPU 跑稳。
- Modal 封装脚本会把仓库里的 `example/verl` 运行时代码挂到镜像中，降低 tool-agent API 与当前项目不一致的风险。
- 当前镜像基于 Modal 官方 GRPO + verl 示例使用的 `verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.1`，再叠加本项目代码和运行依赖。

## 官方参考

- [Modal GRPO + verl 示例](https://modal.com/docs/examples/grpo_verl)
- [Modal GPU 加速](https://modal.com/docs/guide/gpu)
- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal Volume CLI](https://modal.com/docs/reference/cli/volume)
- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Modal Timeouts](https://modal.com/docs/guide/timeouts)
- [Modal Preemption](https://modal.com/docs/guide/preemption)
