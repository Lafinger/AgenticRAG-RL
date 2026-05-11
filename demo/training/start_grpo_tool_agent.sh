#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
VERL_DIR=${VERL_DIR:?"Please set VERL_DIR to your verl checkout or install directory."}

RETRIEVAL_SERVER_URL=${RETRIEVAL_SERVER_URL:-http://127.0.0.1:8790}
if ! curl -s "${RETRIEVAL_SERVER_URL}/health" | grep -q "ok"; then
  echo "ERROR: Retrieval server is not ready at ${RETRIEVAL_SERVER_URL}."
  echo "Start demo/training/tools/retrieval_server.py with --index-dir before GRPO."
  exit 1
fi

cleanup_gpu() {
  ray stop --force 2>/dev/null || true
  pkill -9 -f 'verl\.trainer|python3 -m verl|VLLM::|vllm' 2>/dev/null || true
  pkill -9 -f 'ray::|raylet|gcs_server|ray-dashboard|dashboardagent|default_worker|monitor\.py|runtime_env' 2>/dev/null || true
  sleep 3
}
cleanup_gpu

MODEL_PATH=${MODEL_PATH:-${PROJECT_DIR}/models/Qwen3-4B-Instruct-2507-Unsloth-SFT-react-v4-merged}
DATA_DIR=${DATA_DIR:-${PROJECT_DIR}/data/novel_eval}
TRAIN_FILES=${TRAIN_FILES:-${DATA_DIR}/grpo_agentic_train.parquet}
VAL_FILES=${VAL_FILES:-${DATA_DIR}/grpo_agentic_val.parquet}
TOOL_CONFIG=${TOOL_CONFIG:-${PROJECT_DIR}/training/config/novel_tool_config.yaml}
REWARD_FN=${REWARD_FN:-${PROJECT_DIR}/training/reward_agentic_rag.py}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_DIR}/training/outputs/grpo_tool_agent_react_v4}
LOG_DIR=${LOG_DIR:-${PROJECT_DIR}/logs}
LOG=${LOG:-${LOG_DIR}/grpo_tool_agent_react_v4.log}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_ASSISTANT_TURNS=${MAX_ASSISTANT_TURNS:-7}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
SAVE_FREQ=${SAVE_FREQ:-5}
TEST_FREQ=${TEST_FREQ:-3}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

mkdir -p "${LOG_DIR}"
rm -rf "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES
export ATTN_BACKEND=${ATTN_BACKEND:-flash_attn}
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export AGENTIC_RAG_REWARD_VERSION=${AGENTIC_RAG_REWARD_VERSION:-v9a}

cd "${VERL_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length=1024 \
  data.max_response_length=1024 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.05 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward.custom_reward_function.path="${REWARD_FN}" \
  reward.custom_reward_function.name=compute_score \
  algorithm.use_kl_in_reward=False \
  trainer.critic_warmup=0 \
  trainer.logger='["console"]' \
  trainer.project_name=agentic-rag-rl \
  trainer.experiment_name=qwen3_4b_grpo_tool_agent_react_v4 \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  2>&1 | tee "${LOG}"
