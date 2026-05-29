# STAR

Reference implementation and evaluation harness for **STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning** (Sumin Park, Noseong Park, ICML 2026). The repo is built on NVIDIA Megatron-LM.

## Routing variants

| Variant       | Flag                              | Routing rule                                                                 |
|---------------|-----------------------------------|------------------------------------------------------------------------------|
| Standard MoE  | (default top-k)                   | Each token picks top-k experts; aux load-balancing loss                      |
| ReMoE         | `--moe-relu-routing`              | ReLU gates with adaptive L1 regularization for sparsity                      |
| Expert-Choice | `--moe-expert-choice-routing`     | Each expert picks its top-k tokens; auto-balanced by construction            |
| STAR          | `--moe-star-routing`              | Linear gate + GHA-driven principal-subspace gate, interpolated per expert    |

STAR-specific flags: `--moe-star-gha-lr`, `--moe-star-gha-only`, `--moe-star-fp32-gate`.

## Repo layout

```
STAR/
├── pretrain_gpt.py                   # training entry point
├── gpt_builders.py                   # model spec builders
├── model_provider.py                 # GPT model factory
├── data_preprocessing.sh             # Pile JSONL -> tokenized .bin/.idx
├── gpt2-vocab.json, gpt2-merges.txt  # GPT-2 BPE
├── megatron/                         # slim Megatron-LM (core + training + legacy)
│   └── core/transformer/moe/         # routers: star.py, router.py, moe_layer.py
├── configs/
│   └── llama_182m_star.json          # reference 182M STAR config
├── scripts/                          # 8 launch scripts
│   ├── train_llama_182m_{moe,remoe,ec,star}.sh
│   └── train_llama_469m_{moe,remoe,ec,star}.sh
├── evaluation/                       # 8 eval scripts + 2 helpers
│   ├── lm_harness_eval.py            # lm-evaluation-harness wrapper
│   ├── combine_eval_results.py       # harness + LAMBADA -> one JSON
│   ├── run_eval_combined_{moe,remoe,ec,star}_182m.sh
│   └── run_eval_combined_{moe,remoe,ec,star}_469m.sh
├── tasks/                            # LAMBADA evaluator (from Megatron tasks/)
│   └── zeroshot_gpt/
└── tools/
    └── preprocess_data.py            # tokenizer driver for data_preprocessing.sh
```

## Model architectures

| Scale | Layers | Hidden | FFN  | Heads | GQA groups | Seq len | Experts (default) |
|-------|--------|--------|------|-------|------------|---------|-------------------|
| 182M  | 12     | 768    | 3072 | 12    | 4          | 1024    | 8                 |
| 469M  | 24     | 1024   | 4096 | 16    | 4          | 1024    | 8                 |

Common: RMSNorm, SwiGLU, RoPE (base 1e6), untied input/output embeddings, top-k=1 routing.

## Training

Each script accepts positional args: `GPUS_PER_NODE TRAIN_ITERS MICRO_BATCH_SIZE NUM_EXPERTS GRANULARITY [extras] PROJECT_NAME`. Defaults: GPUS=4, ITERS=60000, MBS=64, EXPERTS=8, GRANULARITY=1. Global batch size 512, base LR 4e-4 (5e-4 for 182M Standard MoE), cosine decay to min-LR 5e-5, warmup fraction 0.01, grad clip 1.0. 60K iters at gbs 512 = 30B tokens.

Common environment overrides: `CUDA_VISIBLE_DEVICES` (default `0,1,2,3`), `DATA_ROOT`, `PROJECT_NAME`. Checkpoints land under `./logs/$PROJECT_NAME`.

Examples:

```
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_182m_moe.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_182m_remoe.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_182m_ec.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_182m_star.sh

DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_469m_moe.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_469m_remoe.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_469m_ec.sh
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_469m_star.sh
```

The STAR launchers default to the best-reported configuration: aux loss with GHA LR 5e-5 at both scales. Override via positional arg 6 (`GHA_LR`).

## Evaluation

Six LM-evaluation-harness tasks plus LAMBADA in one combined script:

```
CKPT_ROOT=./logs/star_182m_lr4e4_aux_B64_E8_GLR5e-5 \
LAMBADA_DATA=/path/to/lambada_test.jsonl \
bash evaluation/run_eval_combined_star_182m.sh
```

Tasks: `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `piqa`, `race` (lm-eval-harness) + `LAMBADA` (Megatron task). Output JSON: `${CKPT_ROOT}/eval_combined_<variant>_<scale>_<iter>.json` with per-task accuracy, harness average, LAMBADA accuracy, and the seven-task total average.


## Acknowledgement

Built on NVIDIA Megatron-LM (https://github.com/NVIDIA/Megatron-LM). The MoE infrastructure, distributed training loop, tokenizers, and checkpointing are upstream; routing variants (ReMoE, Expert-Choice, STAR) are layered on top with minimal additions.
