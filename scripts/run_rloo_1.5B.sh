#!/bin/bash
set -euxo pipefail

# =================== User Configuration =====================
# Please modify these variables according to your environment
# ============================================================
MODEL_PATH="../src/Qwen2.5-1.5B-Instruct"
TRAIN_DATA="../src/data/dapo-math-17k.parquet"
VAL_DATA="../src/data/dapo-math-17k.parquet"
SAVE_PATH="../ouput"

REWARD_FUNC=""
EXP_NAME="Qwen2.5-1.5B-Instruct-R1"
PROJECT_NAME="Reasoning"

# =================== Script Execution ===================
# You shouldn't need to modify anything below this line
# ========================================================
export WORKING_DIR="${PWD}"
export RUNTIME_ENV="./verl/trainer/runtime_env.yaml"
export NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Get script PID and setup directories
SCRIPT_PID=$$
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_DIR="${SAVE_PATH}/${EXP_NAME}/${TIMESTAMP}"

# Stop any existing ray processes
ray stop

# Start ray
echo "Starting ray..."
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
ray start --head --node-ip-address 0.0.0.0 --num-gpus ${NUM_GPUS} --disable-usage-stats

# Start training
echo "Starting training..."
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.truncation='left' \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.train_batch_size=256 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${NUM_GPUS}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    algorithm.adv_estimator=rloo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_penalty=kl \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path="${REWARD_FUNC}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${SAVE_DIR}/ckpt" \
    trainer.resume_mode=auto > >(tee -a "${SAVE_DIR}/train.log") 2>&1 &