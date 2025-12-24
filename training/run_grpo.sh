#!/usr/bin/env bash

# Environment
RAY_RUNTIME_ENV_LOCAL_DEV_MODE=1
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"
N_GPUS_PER_NODE=4
N_NODES=1
RAY_RUNTIME_ENV_LOCAL_DEV_MODE=1
UV_PROJECT_ENVIRONMENT="$(pwd)/.venv"
PROJECT_NAME="TWIN"
EXPERIMENT_NAME="TWIN-Qwen2.5-VL-3B"
DEFAULT_LOCAL_DIR="training/data/checkpoints/"

# Data
TRAIN_FILE=""
VAL_FILE=""
MAX_PROMPT_LEN=2048
MAX_RESPONSE_LEN=2048

# Model / Optimization
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
TRAIN_BATCH_SIZE=480
PPO_MINI_BATCH_SIZE=480
KL_IN_REWARD=False
ROLLOUT_N=5
LR=1e-6
EPOCHS=1
KL_LOSS_COEF=0.01
ENTROPY_COEF=0
SAVE_FREQ=40
TEST_FREQ=20

# Performance Optimization
PPO_MICRO_BATCH_SIZE_PER_GPU=60
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=160
TENSOR_MODEL_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.6
GRADIENT_CHECKPOINTING=True
ACTIVATION_OFFLOAD=True
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True

# Reward 
REWARD_FILE_PATH="training/reward.py"
REWARD_FN="compute_score_batched"

# Data
TRAIN_FILES="training/data/train.parquet"
VAL_FILES="training/data/test.parquet"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TOKENIZERS_PARALLELISM=false
export RAY_RUNTIME_ENV_LOCAL_DEV_MODE
export UV_PROJECT_ENVIRONMENT

uv run --active -- python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.balance_binary_labels=True \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.param_dtype=bf16 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.reduce_dtype=fp32 \
    +actor_rollout_ref.ref.fsdp_config.mixed_precision.buffer_dtype=fp32 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.model.enable_activation_offload="${ACTIVATION_OFFLOAD}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEF}" \
    actor_rollout_ref.model.enable_gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_MODEL_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    algorithm.use_kl_in_reward="${KL_IN_REWARD}" \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${REWARD_FILE_PATH}" \
    custom_reward_function.name="${REWARD_FN}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.nnodes="${N_NODES}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${EPOCHS}" $@
