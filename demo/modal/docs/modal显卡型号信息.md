# Modal 显卡型号信息

> 价格按 Modal 官网基础价填写；算力为 AI 常用 Tensor Core 理论峰值，实际训练/推理吞吐会受精度、batch size、内存带宽、框架和 kernel 影响。价格会随 Modal 官方页面调整，正式训练前应重新核对。

| Modal 型号 | 显存 | 主要 AI 峰值算力 | CUDA 算力等级 | 官网基础价 | $30 大约可用时长 |
|---|---:|---:|---:|---:|---:|
| T4 GPU | 16 GB GDDR6 | FP16/FP32 mixed 65 TFLOPS | 7.5 | $0.000164/s ≈ $0.59/h | 约 51 小时 |
| L4 GPU | 24 GB GDDR6 | FP16 121 TFLOPS，FP8 242.5 TFLOPS；含稀疏约翻倍 | 8.9 | $0.000222/s ≈ $0.80/h | 约 38 小时 |
| A10 GPU | 24 GB GDDR6 | FP16 125 TFLOPS；含稀疏 250 TFLOPS | 8.6 | $0.000306/s ≈ $1.10/h | 约 27 小时 |
| L40S GPU | 48 GB GDDR6 | FP16 362 TFLOPS，FP8 733 TFLOPS；含稀疏约翻倍 | 8.9 | $0.000542/s ≈ $1.95/h | 约 15 小时 |
| A100 40GB | 40 GB HBM2 | FP16 312 TFLOPS；含稀疏 624 TFLOPS | 8.0 | $0.000583/s ≈ $2.10/h | 约 14 小时 |
| A100 80GB | 80 GB HBM2e | FP16 312 TFLOPS；含稀疏 624 TFLOPS | 8.0 | $0.000694/s ≈ $2.50/h | 约 12 小时 |
| H100 | 80 GB HBM3 | FP16 约 990 TFLOPS，FP8 约 1,979 TFLOPS；含稀疏约翻倍 | 9.0 | $0.001097/s ≈ $3.95/h | 约 7.6 小时 |
| H200 | 141 GB HBM3e | FP16 约 990 TFLOPS，FP8 约 1,979 TFLOPS；含稀疏约翻倍 | 9.0 | $0.001261/s ≈ $4.54/h | 约 6.6 小时 |
| B200 | 180 GB HBM3e | FP8 约 4.5 至 5 PFLOPS；FP4 约 9 PFLOPS；含稀疏约翻倍 | 10.0 | $0.001736/s ≈ $6.25/h | 约 4.8 小时 |

## 使用建议

- 本项目 Modal smoke 优先使用 `H100:1` 或 `H100:2`，先验证 retrieval server、verl、vLLM、tool-agent 和 reward 是否闭环。
- 当前正式 GRPO 第一版使用 `H100:2`，与 `trainer.nnodes=1`、`trainer.n_gpus_per_node=2` 的单节点配置保持一致。
- `H100:4` 或 `A100 80GB:4` 可作为后续吞吐扩容选项；扩容前应重新验证 batch、rollout 并发、checkpoint 和 resume。
- 不建议第一版直接使用 Modal 多节点；先把单节点 2 GPU 的 checkpoint、resume 和输出下载流程跑稳。

## 参考来源

- Modal GPU 加速文档：https://modal.com/docs/guide/gpu
- Modal 价格页：https://modal.com/pricing
- NVIDIA CUDA GPU / 算力等级：https://developer.nvidia.com/cuda-gpus
- NVIDIA 各型号官方规格页：T4、L4、A10、L40S、A100、H100、H200、B200。
