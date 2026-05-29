# STAR

Reference implementation and evaluation harness for **STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning** (Sumin Park, Noseong Park, ICML 2026). The repo trains and evaluates four Mixture-of-Experts routing variants at two model scales, built on NVIDIA Megatron-LM.

## Overview

Standard MoE gating is a shallow linear projection; expert specialization depends on whether that projection is actually aware of the input distribution. STAR augments standard gating with an evolving low-dimensional routing subspace V learned online via the Generalized Hebbian Algorithm (GHA), combined with a learnable mixing matrix R, and interpolated against the linear gate via per-expert learnable coefficients alpha:

    s = Softmax( sigma(alpha) * l_linear + (1 - sigma(alpha)) * l_GHA )

with `l_linear = x * W_g^T`, `l_GHA = x * Z^T`, `Z = R * V`. V is refreshed at every forward pass by m GHA iterations; R and alpha learn via standard backprop on the task loss. STAR composes with any explicit load-balancing loss.

## Routing variants

| Variant       | Flag                              | Routing rule                                                                 |
|---------------|-----------------------------------|------------------------------------------------------------------------------|
| Standard MoE  | (default top-k)                   | Each token picks top-k experts; aux load-balancing loss                      |
| ReMoE         | `--moe-relu-routing`              | ReLU gates with adaptive L1 regularization for sparsity                      |
| Expert-Choice | `--moe-expert-choice-routing`     | Each expert picks its top-k tokens; auto-balanced by construction            |
| STAR          | `--moe-star-routing`              | Linear gate + GHA-driven principal-subspace gate, interpolated per expert    |

STAR-specific flags: `--moe-star-gha-lr`, `--moe-star-gha-lr-schedule`, `--moe-star-gha-lr-min`, `--moe-star-gha-only`, `--moe-star-fp32-gate`.

## Repo layout

```
STAR/
  README.md
  requirements.txt
  pretrain_gpt.py              GPT/LLaMA pre-training entry point
  data_preprocessing.sh        Pile JSONL -> tokenized binary
  gpt2-vocab.json              GPT-2 BPE vocab
  gpt2-merges.txt              GPT-2 BPE merges
  configs/
    llama_182m_star.json       Reference 182M STAR config
  megatron/                    Slim Megatron-LM (core + training + legacy)
    core/transformer/moe/      MoE source: router.py, moe_layer.py, star.py, ...
  scripts/                     Eight launch scripts
    train_llama_{182m,469m}_{moe,remoe,ec,star}.sh
  evaluation/
    lm_harness_eval.py         lm-evaluation-harness wrapper
    combine_eval_results.py    Aggregates LM-harness + LAMBADA into one JSON
    run_eval_combined_{moe,remoe,ec,star}_{182m,469m}.sh
  tasks/                       LAMBADA eval support
    main.py, eval_utils.py, data_utils.py, finetune_utils.py
    zeroshot_gpt/              LAMBADA + WikiText103 evaluators
  tools/
    preprocess_data.py         Tokenizer for data_preprocessing.sh
```

## Setup

Tested with PyTorch 2.4 + CUDA 12.x. Recommended base: NVIDIA NGC PyTorch 24.04+ container, which ships compatible builds of TransformerEngine, Apex, and Flash-Attention.

```
pip install -r requirements.txt
```

`transformer-engine`, `apex`, and `flash-attn` must come from CUDA-matched wheels; install them according to the upstream Megatron-LM instructions for your CUDA/PyTorch combination.

## Dataset

The Pile, raw JSONL format, tokenized to Megatron `.bin`/`.idx` shards via GPT-2 BPE.

1. Place raw Pile shards under `${PILE_RAW}/{00..29}.jsonl`.
2. Run preprocessing:

```
PILE_RAW=/path/to/pile_jsonl PILE_OUT=/path/to/pile_gpt_test bash data_preprocessing.sh
```

3. Point `DATA_ROOT` at the output directory when launching training:

```
DATA_ROOT=/path/to/pile_gpt_test bash scripts/train_llama_182m_star.sh
```

LAMBADA eval needs `lambada_test.jsonl` (https://github.com/openai/gpt-2/blob/master/src/lambada_test.jsonl); set `LAMBADA_DATA` when launching evaluation. Split: 969:30:1 train:val:test.

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

The STAR launchers default to the best-reported configurations: 182M uses aux loss with GHA LR 5e-5; 469M uses aux loss with scheduled GHA LR (2e-4 cosine-decayed to 2e-5). Override via positional arg 6 (`GHA_LR`) or, for 469M, env var `GHA_LR_MIN`.

## Evaluation

Six LM-evaluation-harness tasks plus LAMBADA in one combined script:

```
CKPT_ROOT=./logs/star_182m_lr4e4_aux_B64_E8_GLR5e-5 \
LAMBADA_DATA=/path/to/lambada_test.jsonl \
bash evaluation/run_eval_combined_star_182m.sh
```

Tasks: `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `piqa`, `race` (lm-eval-harness) + `LAMBADA` (Megatron task). Output JSON: `${CKPT_ROOT}/eval_combined_<variant>_<scale>_<iter>.json` with per-task accuracy, harness average, LAMBADA accuracy, and the seven-task total average.

## Limitations

- This repo is for fresh training. Legacy ASMG checkpoints from prior internal trees cannot be loaded as-is because every `moe_asmg_*` state-dict key was renamed to `moe_star_*`; a one-time key remap is required to resume.
- TransformerEngine is not vendored. Install a CUDA-matched build separately.
- `megatron/post_training/`, `rl/`, and `inference/` were dropped from the slim copy. The training/eval paths do not need them; if you import upstream Megatron features that do, restore those subpackages from the upstream source.

## Citation

```
@inproceedings{park2026star,
  title     = {STAR: Rethinking MoE Routing as Structure-Aware Subspace Learning},
  author    = {Park, Sumin and Park, Noseong},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```

## Acknowledgement

Built on NVIDIA Megatron-LM (https://github.com/NVIDIA/Megatron-LM). The MoE infrastructure, distributed training loop, tokenizers, and checkpointing are upstream; routing variants (ReMoE, Expert-Choice, STAR) are layered on top with minimal additions.
