#!/bin/bash
# Combined evaluation script for ReMoE model.
# Runs both lm_harness tasks and LAMBADA, then combines results.

set -e

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

# Distributed env vars (required by Megatron even for single GPU eval).
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + RANDOM % 10000))
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

export PYTHONWARNINGS="ignore"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_TRUST_REMOTE_CODE=1

MODEL=moe
ITER=${ITER:-60000}
CKPT_ROOT=${CKPT_ROOT:-./logs/remoe_182m_lr4e4_B64_E8}
NUM_EXPERTS=${NUM_EXPERTS:-8}
GRANULARITY=${GRANULARITY:-1}
MOE_RELU_L1_REG_COEFF_INIT=${MOE_RELU_L1_REG_COEFF_INIT:-1e-8}
MOE_RELU_L1_REG_COEFF_MULTIPLIER=${MOE_RELU_L1_REG_COEFF_MULTIPLIER:-1.2}
LM_HARNESS_TASKS="arc_challenge,arc_easy,boolq,hellaswag,piqa,race"
LAMBADA_DATA=${LAMBADA_DATA:-/path/to/lambada_test.jsonl}

LM_HARNESS_RESULTS=${CKPT_ROOT}/eval_results_remoe_${ITER}.json
LAMBADA_LOG=${CKPT_ROOT}/eval_lambada_remoe_${ITER}.log
COMBINED_RESULTS=${CKPT_ROOT}/eval_combined_remoe_${ITER}.json

SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-false}

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 1024
    --max-position-embeddings 1024
    --num-layers 12
    --hidden-size 768
    --ffn-hidden-size 3072
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
)

MOE_ARGS=(
    --num-experts ${NUM_EXPERTS}
    --moe-router-topk 1
    --moe-router-load-balancing-type none
    --moe-token-dispatcher-type alltoall
    --moe-relu-routing
    --moe-relu-l1-reg-coeff-init ${MOE_RELU_L1_REG_COEFF_INIT}
    --moe-relu-l1-reg-coeff-multiplier ${MOE_RELU_L1_REG_COEFF_MULTIPLIER}
    --moe-grouped-gemm
    --moe-granularity ${GRANULARITY}
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
)

DATA_ARGS=(
    --vocab-file gpt2-vocab.json
    --merge-file gpt2-merges.txt
    --tokenizer-type GPT2BPETokenizer
    --make-vocab-size-divisible-by 1024
)

EVAL_ARGS=(
    --load $CKPT_ROOT
    --no-load-rng
    --no-load-optim
    --micro-batch-size 1
    --bf16
    --use-cpu-initialization
    --mock-data
    --no-gradient-accumulation-fusion
)

echo "=========================================="
echo "Combined Evaluation for 182M ReMoE Model"
echo "=========================================="
echo "Checkpoint: $CKPT_ROOT"
echo "Iteration: $ITER"
echo ""

python -c "import lm_eval" 2>/dev/null || pip install lm-eval==0.4.2

echo "=========================================="
echo "Part 1: LM Harness Evaluation"
echo "Tasks: $LM_HARNESS_TASKS"
echo "=========================================="

if [ "$SKIP_IF_EXISTS" = "true" ] && [ -f "$LM_HARNESS_RESULTS" ]; then
    echo "Skipping LM Harness evaluation - results already exist at: ${LM_HARNESS_RESULTS}"
else
    python evaluation/lm_harness_eval.py \
        ${MODEL_ARGS[@]} \
        ${MOE_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_ARGS[@]} \
        --model $MODEL \
        --tasks ${LM_HARNESS_TASKS} \
        --batch-size 64 \
        --output-path ${LM_HARNESS_RESULTS} 2>&1 | grep -v "CPU RNG state changed"
    echo ""
    echo "LM Harness results saved to: ${LM_HARNESS_RESULTS}"
fi

echo ""
echo "=========================================="
echo "Part 2: LAMBADA Evaluation"
echo "=========================================="

export MASTER_PORT=$((29500 + RANDOM % 10000))

DISTRIBUTED_ARGS=(
    --nproc_per_node 1
    --nnodes 1
    --node_rank 0
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

LAMBADA_EVAL_ARGS=(
    --load $CKPT_ROOT
    --no-load-rng
    --no-load-optim
    --micro-batch-size 8
    --bf16
    --no-gradient-accumulation-fusion
    --epochs 0
)

MODEL_ARGS_LAMBADA=(
    "${MODEL_ARGS[@]}"
    --use-flash-attn
)

MOE_ARGS_LAMBADA=(
    "${MOE_ARGS[@]}"
)

if [ "$SKIP_IF_EXISTS" = "true" ] && [ -f "$LAMBADA_LOG" ] && ! grep -q "LAMBADA_SKIPPED\|FileNotFoundError\|Traceback" "$LAMBADA_LOG" 2>/dev/null; then
    echo "Skipping LAMBADA evaluation - results already exist at: ${LAMBADA_LOG}"
elif [ -f "$LAMBADA_DATA" ]; then
    torchrun ${DISTRIBUTED_ARGS[@]} tasks/main.py \
        ${MODEL_ARGS_LAMBADA[@]} \
        ${MOE_ARGS_LAMBADA[@]} \
        ${DATA_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${LAMBADA_EVAL_ARGS[@]} \
        --task LAMBADA \
        --valid-data ${LAMBADA_DATA} \
        --strict-lambada 2>&1 | tee ${LAMBADA_LOG}

    echo ""
    echo "LAMBADA results saved to: ${LAMBADA_LOG}"
else
    echo "WARNING: LAMBADA dataset not found at ${LAMBADA_DATA}"
    echo "Skipping LAMBADA evaluation"
    echo "LAMBADA_SKIPPED" > ${LAMBADA_LOG}
fi

echo ""
echo "=========================================="
echo "Part 3: Combining Results"
echo "=========================================="

python evaluation/combine_eval_results.py \
    --lm-harness-results ${LM_HARNESS_RESULTS} \
    --lambada-log ${LAMBADA_LOG} \
    --output ${COMBINED_RESULTS} \
    --checkpoint ${CKPT_ROOT} \
    --iteration ${ITER}

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="
echo "Combined results saved to: ${COMBINED_RESULTS}"
echo ""
cat ${COMBINED_RESULTS}
