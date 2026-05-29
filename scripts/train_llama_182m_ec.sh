#!/bin/bash
# bash scripts/train_llama_182m_ec.sh [gpus_per_node] [train_iters] [micro_batch_size] [num_experts] [granularity] [capacity_factor] [project_name]
# Expert-Choice routing variant of train_llama_182m_moe.sh.

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_ROOT=${DATA_ROOT:-/path/to/pile_gpt_test}

GPUS_PER_NODE=${1:-"4"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i25000-40000 -n1)}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 512 * 1024 * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"64"}
NUM_EXPERTS=${4:-"8"}
GRANILARITY=${5:-"1"}
CAPACITY_FACTOR=${6:-"1.0"}
PROJECT_NAME=${7:-"ec_182m_lr4e4_B64_E${NUM_EXPERTS}_CF${CAPACITY_FACTOR}"}

LOCAL_LOG_PATH="./logs/$PROJECT_NAME"
CHECKPOINT_PATH="./logs/$PROJECT_NAME"
mkdir -p "$LOCAL_LOG_PATH" "$CHECKPOINT_PATH"

TRACKER_FILE="${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt"

# Recomputed on every (re)launch so each retry picks up the latest checkpoint.
compute_load_args() {
    LOAD_ARGS=()
    if [ -f "$TRACKER_FILE" ]; then
        local tracker_value iter_dir
        tracker_value="$(tr -d '[:space:]' < "$TRACKER_FILE")"
        iter_dir=""
        if [[ "$tracker_value" =~ ^[0-9]+$ ]]; then
            iter_dir="${CHECKPOINT_PATH}/iter_$(printf "%07d" "$((10#${tracker_value}))")"
        elif [ "$tracker_value" = "release" ]; then
            iter_dir="${CHECKPOINT_PATH}/release"
        fi
        if [ -n "$iter_dir" ] && [ -f "${iter_dir}/metadata.json" ]; then
            LOAD_ARGS=(--load "$CHECKPOINT_PATH")
        fi
    fi
}

PILE_DATASET="\
1.0 \
${DATA_ROOT}/01_text_document \
1.0 \
${DATA_ROOT}/02_text_document \
1.0 \
${DATA_ROOT}/03_text_document \
1.0 \
${DATA_ROOT}/04_text_document \
1.0 \
${DATA_ROOT}/05_text_document \
1.0 \
${DATA_ROOT}/06_text_document \
1.0 \
${DATA_ROOT}/07_text_document \
1.0 \
${DATA_ROOT}/08_text_document \
1.0 \
${DATA_ROOT}/09_text_document \
1.0 \
${DATA_ROOT}/10_text_document \
1.0 \
${DATA_ROOT}/11_text_document \
1.0 \
${DATA_ROOT}/12_text_document \
1.0 \
${DATA_ROOT}/13_text_document \
1.0 \
${DATA_ROOT}/14_text_document \
1.0 \
${DATA_ROOT}/15_text_document \
1.0 \
${DATA_ROOT}/16_text_document \
1.0 \
${DATA_ROOT}/17_text_document \
1.0 \
${DATA_ROOT}/18_text_document \
1.0 \
${DATA_ROOT}/19_text_document \
1.0 \
${DATA_ROOT}/20_text_document \
1.0 \
${DATA_ROOT}/21_text_document \
1.0 \
${DATA_ROOT}/22_text_document \
1.0 \
${DATA_ROOT}/23_text_document \
1.0 \
${DATA_ROOT}/24_text_document \
1.0 \
${DATA_ROOT}/25_text_document \
1.0 \
${DATA_ROOT}/26_text_document \
1.0 \
${DATA_ROOT}/27_text_document \
1.0 \
${DATA_ROOT}/28_text_document \
1.0 \
${DATA_ROOT}/29_text_document"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 12
    --hidden-size 768
    --ffn-hidden-size $((768 * 4))
    --num-attention-heads 12
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 4
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
    --use-flash-attn
)

MOE_ARGS=(
    --num-experts $NUM_EXPERTS
    --moe-router-topk 1
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --recompute-granularity selective
    --recompute-modules moe
    --moe-granularity $GRANILARITY
    --moe-expert-choice-routing
    --moe-expert-capacity-factor $CAPACITY_FACTOR
    --moe-expert-choice-softmax-order softmax_topk
)

DATA_ARGS=(
    --vocab-file gpt2-vocab.json \
    --merge-file gpt2-merges.txt \
    --make-vocab-size-divisible-by 1024 \
    --data-path $PILE_DATASET
    --split 969,30,1
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size 512
    --lr 4e-4
    --train-iters $TRAIN_ITERS
    --lr-decay-style cosine
    --min-lr 5e-5
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --bf16
    --no-gradient-accumulation-fusion
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 10
    --log-throughput
    --save-interval 500
    --eval-interval 1000
    --eval-iters 100
    --save $CHECKPOINT_PATH
    --tensorboard-dir "${LOCAL_LOG_PATH}/tensorboard"
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "EC"
        --wandb-exp-name $PROJECT_NAME
    )
fi

KEEP_LAST_N=2

cleanup_old_checkpoints() {
    if [ -d "$CHECKPOINT_PATH" ]; then
        local checkpoints=($(ls -td "$CHECKPOINT_PATH"/iter_* 2>/dev/null))
        local num_checkpoints=${#checkpoints[@]}
        if [ $num_checkpoints -gt $KEEP_LAST_N ]; then
            for ((i=$KEEP_LAST_N; i<$num_checkpoints; i++)); do
                echo "Removing old checkpoint: ${checkpoints[$i]}"
                rm -rf "${checkpoints[$i]}"
            done
        fi
    fi
}

(
    while true; do
        sleep 3600
        cleanup_old_checkpoints
    done
) &
CLEANUP_PID=$!

trap "kill $CLEANUP_PID 2>/dev/null; cleanup_old_checkpoints" EXIT

MAX_RETRIES=${MAX_RETRIES:-50}
RETRY_BACKOFF_SECS=${RETRY_BACKOFF_SECS:-30}

log_msg() { echo "[auto-resume $(date '+%F %T')] $*" | tee -a "$LOCAL_LOG_PATH/train.log"; }

ATTEMPT=0
while :; do
    ATTEMPT=$((ATTEMPT + 1))
    compute_load_args
    if [ ${#LOAD_ARGS[@]} -gt 0 ]; then
        log_msg "attempt $ATTEMPT/$((MAX_RETRIES + 1)) — resuming from checkpoint $CHECKPOINT_PATH"
    else
        log_msg "attempt $ATTEMPT/$((MAX_RETRIES + 1)) — no valid checkpoint, starting fresh"
    fi

    torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LOAD_ARGS[@]} \
        ${LOGGING_ARGS[@]} |& tee -a $LOCAL_LOG_PATH/train.log
    EXIT_CODE=${PIPESTATUS[0]}

    case $EXIT_CODE in
        0)
            log_msg "training finished successfully on attempt $ATTEMPT."
            break
            ;;
        130|143)
            log_msg "interrupted by signal (exit $EXIT_CODE); not retrying."
            exit $EXIT_CODE
            ;;
        *)
            if [ $ATTEMPT -gt $MAX_RETRIES ]; then
                log_msg "reached MAX_RETRIES=$MAX_RETRIES; giving up (last exit $EXIT_CODE)."
                exit $EXIT_CODE
            fi
            log_msg "torchrun exited $EXIT_CODE; sleeping ${RETRY_BACKOFF_SECS}s then retrying."
            sleep $RETRY_BACKOFF_SECS
            ;;
    esac
done
