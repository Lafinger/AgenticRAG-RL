# Modal Secret 说明

本项目建议统一使用一个 Modal Secret：

```powershell
modal secret create --force agentic-rag-rl-secrets `
  HF_TOKEN=... `
  WANDB_API_KEY=... `
  SWANLAB_API_KEY=... `
  COMMON_API_KEY=... `
  RIGHTCODE_API_KEY=...
```

只填写当前任务真实需要的 key。当前 GRPO 主线默认只使用 console 日志器，不强依赖外部实验平台或在线模型 API。

| 变量名 | 是否必需 | 用途 |
| --- | --- | --- |
| `HF_TOKEN` | 可选 | 下载私有 Hugging Face 模型或数据时使用。 |
| `WANDB_API_KEY` | 可选 | 只有把 `trainer.logger` 覆盖为包含 `wandb` 时才需要。 |
| `SWANLAB_API_KEY` | 可选 | 未来如果在 Modal 上跑 SFT 或 SwanLab 观测时使用。 |
| `COMMON_API_KEY` | 可选 | 本地数据构造或 Judge 流程中使用的 Common 在线模型供应商。 |
| `RIGHTCODE_API_KEY` | 可选 | 本地数据构造或 Judge 流程中使用的 RightCode 在线模型供应商。 |

不要把 `.env` 上传到 Modal Volume，也不要把任何 Secret 提交到 Git。

Modal 运行脚本通过下面的方式引用这个 Secret：

```python
modal.Secret.from_name("agentic-rag-rl-secrets")
```

如果当前完全不需要真实 Secret，也建议创建一个占位 Secret，避免 Modal Function 启动时查找失败：

```powershell
modal secret create --force agentic-rag-rl-secrets PLACEHOLDER=1
```
