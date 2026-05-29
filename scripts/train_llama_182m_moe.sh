#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DATA_ROOT=${DATA_ROOT:-/path/to/pile_gpt_test}

GPUS_PER_NODE=${1:-"1"}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i25000-40000 -n1)}
NNODES=${SLURM_NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# 512 * 1k * 60k = 30b tokens.
TRAIN_ITERS=${2:-"60000"}
MICRO_BATCH_SIZE=${3:-"64"}
NUM_EXPERTS=${4:-"8"}
GRANILARITY=${5:-"1"}
PROJECT_NAME=${6:-"moe_train_B64_E8_lr5e4"}

CHECKPOINT_PATH="./logs/$PROJECT_NAME"
mkdir -p $CHECKPOINT_PATH

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
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather
    --overlap-grad-reduce
    --moe-router-pre-softmax
    --moe-grouped-gemm
    --recompute-granularity selective
    --recompute-modules moe
    --moe-upcycling-granularity $GRANILARITY
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
    --lr 5e-4
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
    --sequence-parallel
)

LOGGING_ARGS=(
    --log-interval 10
    --log-throughput 
    --save-interval 2000
    --eval-interval 1000
    --eval-iters 100
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard"
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "ReMoE"
        --wandb-exp-name $PROJECT_NAME
    )
fi


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]} |& tee -a $CHECKPOINT_PATH/train.log
