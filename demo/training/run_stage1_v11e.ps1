[CmdletBinding()]
param(
    [string]$ProjectDir = "",
    [string]$VerlDir = "",
    [string]$ModelPath = "",
    [string]$TrainFile = "",
    [string]$ValFile = "",
    [string]$JudgeBaseUrl = "http://localhost:8086/v1",
    [switch]$PrintOnly
)

$ErrorActionPreference = "Stop"
$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ProjectDir) {
    $ProjectDir = (Resolve-Path (Join-Path $ScriptRoot "..")).Path
}

$env:AGENTIC_RAG_REWARD_VERSION = "v6a"
$VerlDir = if ($VerlDir) { $VerlDir } else { Join-Path $ProjectDir "verl" }
$ModelPath = if ($ModelPath) { $ModelPath } else { Join-Path $ProjectDir "models\Qwen3-4B-SFT-merged" }
$TrainFile = if ($TrainFile) { $TrainFile } else { Join-Path $ProjectDir "data\financial_eval\grpo_agentic_train.parquet" }
$ValFile = if ($ValFile) { $ValFile } else { Join-Path $ProjectDir "data\financial_eval\grpo_agentic_val.parquet" }
$ToolConfig = Join-Path $ProjectDir "training\config\financial_tool_config.yaml"

$command = @"
python -m verl.trainer.main_ppo `
  algorithm.adv_estimator=grpo `
  data.train_files=$TrainFile `
  data.val_files=$ValFile `
  data.train_batch_size=32 `
  data.max_prompt_length=1024 `
  data.max_response_length=1024 `
  data.return_raw_chat=True `
  actor_rollout_ref.model.path=$ModelPath `
  actor_rollout_ref.hybrid_engine=True `
  actor_rollout_ref.rollout.name=vllm `
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent `
  actor_rollout_ref.rollout.multi_turn.enable=True `
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=7 `
  actor_rollout_ref.rollout.multi_turn.tool_config_path=$ToolConfig `
  actor_rollout_ref.rollout.multi_turn.format=hermes `
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 `
  actor_rollout_ref.rollout.n=4 `
  actor_rollout_ref.actor.use_kl_loss=True `
  actor_rollout_ref.actor.kl_loss_coef=0.05 `
  reward.custom_reward_function.path=$(Join-Path $ProjectDir "training\reward_agentic_rag.py") `
  reward.custom_reward_function.name=compute_score `
  trainer.n_gpus_per_node=4 `
  trainer.total_epochs=3 `
  trainer.save_freq=5
"@

Write-Host "Stage1 / v11e"
Write-Host "JUDGE_BASE_URL=$JudgeBaseUrl"
Write-Host $command
if (-not $PrintOnly) {
    Push-Location $VerlDir
    try {
        Invoke-Expression $command
    }
    finally {
        Pop-Location
    }
}
