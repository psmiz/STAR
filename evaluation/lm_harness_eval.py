# usage: lm_harness_eval.py [-h] [--model MODEL] [--tasks task1,task2] [--model_args MODEL_ARGS] [--num_fewshot N] 
                            # [--batch_size auto|auto:N|N] [--max_batch_size N] [--device DEVICE] [--output_path DIR|DIR/file.json] [--limit N|0<N<1] [--use_cache DIR]
#                           [--cache_requests {true,refresh,delete}] [--check_integrity] [--write_out] [--log_samples] 
                            # [--system_instruction SYSTEM_INSTRUCTION] [--apply_chat_template [APPLY_CHAT_TEMPLATE]] [--fewshot_as_multiturn] [--show_config]
#                           [--include_path DIR] [--gen_kwargs GEN_KWARGS] [--verbosity CRITICAL|ERROR|WARNING|INFO|DEBUG] 
                            # [--wandb_args WANDB_ARGS] [--hf_hub_log_args HF_HUB_LOG_ARGS] [--predict_only] [--seed SEED] [--trust_remote_code]

import sys
import torch
import subprocess
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from accelerate import Accelerator, DistributedType
import accelerate
import logging
import pandas as pd

# from models import MODEL_DICT, CONFIG_DICT
from typing import List, Literal, Optional, Tuple, Union
import os
import json
import tqdm
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(message)s',
#     handlers=[
#         logging.FileHandler('output/_logs/log.log', mode='a'),
#         logging.StreamHandler()
#     ]
# )
# logger = accelerate.logging.get_logger('lm-eval')


class BaseEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
        self,
        pretrained=None,
        max_length=2000,
        batch_size=64,
        tokenizer='EleutherAI/gpt-neox-20b',
        # tokenizer="TinyLlama/TinyLlama_v1.1",
        device=f"cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        mixed_precision_dtype=torch.bfloat16,
        softmax_dtype=torch.bfloat16,
        truncation = False,
        logits_cache = True,
        revision = "main",
        backend = 'causal',
        peft = None,
        delta = None,
        prefix_token_id = None,
        expert_nheads=None,
        shared_nheads=None,
        moe_coef=None,
        diversity_coef=None,
        gamma=None,
    ):  # training is everything 32
        LM.__init__(self)

        # Parameters
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)
        self._dtype = dtype
        # New configs for new lm-eval version
        self.mixed_precision_dtype = mixed_precision_dtype
        self.softmax_dtype = softmax_dtype
        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.add_bos_token = False
        self.vocab_size = self.tokenizer.vocab_size
                # self._device = torch.device(device)
        # self.add_bos_token = False
        self.truncation = truncation
        self.logits_cache = logits_cache
        self.vocab_size = self.tokenizer.vocab_size
        self.revision = revision
        self.peft = peft
        self.delta = delta
        self.backend = backend
        self.pretrained = pretrained
        self.custom_prefix_token_id = prefix_token_id
        ## custom heads
        self.expert_nheads=expert_nheads
        self.shared_nheads=shared_nheads
        self.moe_coef=moe_coef
        self.diversity_coef=diversity_coef
        self.gamma=gamma

    def load_model_from_index_json(self, json_path):
        with open(json_path, "r") as f:
            index_data = json.load(f)

        state_dict = {}
        for key, file_name in tqdm.tqdm(index_data["weight_map"].items()):
            file_path = os.path.join(self.pretrained, file_name)
            partial_state_dict = torch.load(file_path)
            state_dict.update(partial_state_dict)

        return state_dict

    def make_pytorch_model(self):
        script_path = os.path.join(self.pretrained, "zero_to_fp32.py")
        try:
            result = subprocess.run(
                ["python", script_path, self.pretrained, self.pretrained, "--max_shard_size=10GB"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                )
        except:
            print(f"You should run {script_path} to get pytorch_model.bin")

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, **kwargs):
        raise NotImplementedError()


@register_model("moe")
class MegatronMoEEvalWrapper(LM):
    """
    Direct Megatron-LM MoE checkpoint loader (no conversion needed).
    Loads native Megatron .distcp format checkpoints for standard MoE models.

    Usage with lm_eval:
        Pass Megatron args via command line, e.g.:
        python lm_harness_eval.py --model moe --load /path/to/checkpoint --use-mcore-models ...
    """

    _initialized = False
    _model = None
    _tokenizer = None
    _args = None

    def __init__(self, batch_size: int = 1, **kwargs):
        LM.__init__(self)
        self._batch_size = batch_size
        
        if not MegatronMoEEvalWrapper._initialized:
            print("Initializing Megatron MoE model...")
            
            # Add Megatron to path
            megatron_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            sys.path.insert(0, megatron_path)
            
            from megatron.training import get_args
            from megatron.training.initialize import initialize_megatron
            from megatron.training.checkpointing import load_checkpoint
            from megatron.training import get_model as megatron_get_model
            from megatron.training import get_tokenizer as megatron_get_tokenizer
            from model_provider import model_provider
            from gpt_builders import gpt_builder
            from functools import partial
            
            initialize_megatron()
            
            args = get_args()
            args.apply_rope_fusion = False
            MegatronMoEEvalWrapper._args = args
            
            print(f"Loading checkpoint from: {args.load}")
            
            model = megatron_get_model(
                partial(model_provider, gpt_builder),
                wrap_with_ddp=False
            )
            
            load_checkpoint(model, None, None)
            
            MegatronMoEEvalWrapper._model = model[0]
            MegatronMoEEvalWrapper._model.eval()
            
            MegatronMoEEvalWrapper._tokenizer = megatron_get_tokenizer()
            MegatronMoEEvalWrapper._initialized = True
            
            print(f"✅ Model loaded successfully")
            print(f"   Num layers: {args.num_layers}")
            print(f"   Hidden size: {args.hidden_size}")
            print(f"   Num experts: {args.num_experts}")
            print(f"   Router topk: {args.moe_router_topk}")
            print(f"   Load balancing: {args.moe_router_load_balancing_type}")
        
        self.model = MegatronMoEEvalWrapper._model
        self.tokenizer = MegatronMoEEvalWrapper._tokenizer
        self.args = MegatronMoEEvalWrapper._args
        self.vocab_size = self.tokenizer.vocab_size
        self.device = torch.cuda.current_device()
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_length(self):
        return self.args.seq_length
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eod
    
    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.tokenize(string)
    
    def tok_decode(self, tokens: list, **kwargs):
        return self.tokenizer.detokenize(tokens)
    
    def _model_call(self, inps: torch.Tensor, attention_mask=None):
        with torch.no_grad():
            batch_size, seq_len = inps.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inps.device).unsqueeze(0).expand(batch_size, -1)
            logits = self.model(inps, position_ids, attention_mask=attention_mask)
            return logits
    
    def loglikelihood(self, requests):
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        results = []
        
        batch_size = self._batch_size
        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i:i + batch_size]
            batch_inputs = []
            batch_context_lens = []
            batch_cont_lens = []
            
            for req in batch_reqs:
                context, continuation = req.args if hasattr(req, 'args') else req
                context_tokens = self.tok_encode(context)
                continuation_tokens = self.tok_encode(continuation)
                full_tokens = context_tokens + continuation_tokens
                
                batch_inputs.append(torch.tensor(full_tokens, dtype=torch.long))
                batch_context_lens.append(len(context_tokens))
                batch_cont_lens.append(len(continuation_tokens))
            
            padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=self.eot_token_id).to(self.device)
            
            logits = self._model_call(padded_inputs[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            for idx, (ctx_len, cont_len) in enumerate(zip(batch_context_lens, batch_cont_lens)):
                cont_start = max(0, ctx_len - 1)
                cont_end = cont_start + cont_len
                
                if cont_end > log_probs.shape[1]:
                    cont_end = log_probs.shape[1]
                    cont_len = cont_end - cont_start
                
                cont_logits = log_probs[idx, cont_start:cont_end]
                original_tokens = batch_inputs[idx].to(self.device)
                cont_targets = original_tokens[ctx_len:ctx_len + cont_len]
                
                cont_log_probs = cont_logits.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
                total_log_prob = cont_log_probs.sum().item()
                is_greedy = (cont_logits.argmax(-1) == cont_targets).all().item()
                
                results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        results = []
        
        batch_size = self._batch_size
        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i:i + batch_size]
            
            batch_inputs = []
            batch_seq_lens = []
            
            for req in batch_reqs:
                (string,) = req.args if hasattr(req, 'args') else req
                tokens = self.tok_encode(string)
                
                if len(tokens) < 2:
                    results.append(0.0)
                    continue
                
                batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
                batch_seq_lens.append(len(tokens))
            
            if not batch_inputs:
                continue
            
            padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=self.eot_token_id).to(self.device)
            
            logits = self._model_call(padded_inputs[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            for idx, seq_len in enumerate(batch_seq_lens):
                original_tokens = batch_inputs[idx].to(self.device)
                targets = original_tokens[1:seq_len]
                
                token_log_probs = log_probs[idx, :len(targets)].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                total_log_prob = token_log_probs.sum().item()
                
                results.append(total_log_prob)
        
        return results
    
    def generate_until(self, requests):
        raise NotImplementedError("Generation not needed for commonsense reasoning tasks")


@register_model("star_moe")
class MegatronSTAREvalWrapper(LM):
    """
    Direct Megatron-LM STAR MoE checkpoint loader (no conversion needed).
    Loads native Megatron .distcp format checkpoints.

    Usage with lm_eval:
        Pass Megatron args via command line, e.g.:
        python lm_harness_eval.py --model star_moe --load /path/to/checkpoint --use-mcore-models ...
    """

    _initialized = False
    _model = None
    _tokenizer = None
    _args = None

    def __init__(self, batch_size: int = 1, **kwargs):
        LM.__init__(self)
        self._batch_size = batch_size
        
        if not MegatronSTAREvalWrapper._initialized:
            print("Initializing Megatron STAR model...")
            
            # Add Megatron to path
            megatron_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            sys.path.insert(0, megatron_path)
            
            from megatron.training import get_args
            from megatron.training.initialize import initialize_megatron
            from megatron.training.checkpointing import load_checkpoint
            from megatron.training import get_model as megatron_get_model
            from megatron.training import get_tokenizer as megatron_get_tokenizer
            from model_provider import model_provider
            from gpt_builders import gpt_builder
            from functools import partial
            
            initialize_megatron()
            
            args = get_args()
            args.apply_rope_fusion = False
            MegatronSTAREvalWrapper._args = args
            
            print(f"Loading checkpoint from: {args.load}")
            
            model = megatron_get_model(
                partial(model_provider, gpt_builder),
                wrap_with_ddp=False
            )
            
            load_checkpoint(model, None, None)
            
            MegatronSTAREvalWrapper._model = model[0]
            MegatronSTAREvalWrapper._model.eval()
            
            MegatronSTAREvalWrapper._tokenizer = megatron_get_tokenizer()
            MegatronSTAREvalWrapper._initialized = True
            
            print(f"✅ Model loaded successfully")
            print(f"   Num layers: {args.num_layers}")
            print(f"   Hidden size: {args.hidden_size}")
            print(f"   Num experts: {args.num_experts}")
            print(f"   STAR routing: {getattr(args, 'moe_star_routing', False)}")
        
        self.model = MegatronSTAREvalWrapper._model
        self.tokenizer = MegatronSTAREvalWrapper._tokenizer
        self.args = MegatronSTAREvalWrapper._args
        self.vocab_size = self.tokenizer.vocab_size
        self.device = torch.cuda.current_device()
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def max_length(self):
        return self.args.seq_length
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eod
    
    def tok_encode(self, string: str, **kwargs):
        return self.tokenizer.tokenize(string)
    
    def tok_decode(self, tokens: list, **kwargs):
        return self.tokenizer.detokenize(tokens)
    
    def _model_call(self, inps: torch.Tensor, attention_mask=None):
        with torch.no_grad():
            batch_size, seq_len = inps.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=inps.device).unsqueeze(0).expand(batch_size, -1)
            logits = self.model(inps, position_ids, attention_mask=attention_mask)
            return logits
    
    def loglikelihood(self, requests):
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        results = []
        
        batch_size = self._batch_size
        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i:i + batch_size]
            batch_inputs = []
            batch_context_lens = []
            batch_cont_lens = []
            
            for req in batch_reqs:
                context, continuation = req.args if hasattr(req, 'args') else req
                context_tokens = self.tok_encode(context)
                continuation_tokens = self.tok_encode(continuation)
                full_tokens = context_tokens + continuation_tokens
                
                batch_inputs.append(torch.tensor(full_tokens, dtype=torch.long))
                batch_context_lens.append(len(context_tokens))
                batch_cont_lens.append(len(continuation_tokens))
            
            padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=self.eot_token_id).to(self.device)
            
            logits = self._model_call(padded_inputs[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            for idx, (ctx_len, cont_len) in enumerate(zip(batch_context_lens, batch_cont_lens)):
                cont_start = max(0, ctx_len - 1)
                cont_end = cont_start + cont_len
                
                if cont_end > log_probs.shape[1]:
                    cont_end = log_probs.shape[1]
                    cont_len = cont_end - cont_start
                
                cont_logits = log_probs[idx, cont_start:cont_end]
                original_tokens = batch_inputs[idx].to(self.device)
                cont_targets = original_tokens[ctx_len:ctx_len + cont_len]
                
                cont_log_probs = cont_logits.gather(-1, cont_targets.unsqueeze(-1)).squeeze(-1)
                total_log_prob = cont_log_probs.sum().item()
                is_greedy = (cont_logits.argmax(-1) == cont_targets).all().item()
                
                results.append((total_log_prob, is_greedy))
        
        return results
    
    def loglikelihood_rolling(self, requests):
        import torch.nn.functional as F
        from torch.nn.utils.rnn import pad_sequence
        results = []
        
        batch_size = self._batch_size
        for i in range(0, len(requests), batch_size):
            batch_reqs = requests[i:i + batch_size]
            
            batch_inputs = []
            batch_seq_lens = []
            
            for req in batch_reqs:
                (string,) = req.args if hasattr(req, 'args') else req
                tokens = self.tok_encode(string)
                
                if len(tokens) < 2:
                    results.append(0.0)
                    continue
                
                batch_inputs.append(torch.tensor(tokens, dtype=torch.long))
                batch_seq_lens.append(len(tokens))
            
            if not batch_inputs:
                continue
            
            padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=self.eot_token_id).to(self.device)
            
            logits = self._model_call(padded_inputs[:, :-1])
            log_probs = F.log_softmax(logits, dim=-1)
            
            for idx, seq_len in enumerate(batch_seq_lens):
                original_tokens = batch_inputs[idx].to(self.device)
                targets = original_tokens[1:seq_len]
                
                token_log_probs = log_probs[idx, :len(targets)].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                total_log_prob = token_log_probs.sum().item()
                
                results.append(total_log_prob)
        
        return results
    
    def generate_until(self, requests):
        raise NotImplementedError("Generation not needed for commonsense reasoning tasks")


if __name__ == "__main__":
    import argparse
    from lm_eval import evaluator
    
    # Parse lm-eval specific args
    parser = argparse.ArgumentParser(description='Evaluate STAR MoE model')
    parser.add_argument('--tasks', type=str, required=True,
                        help='Comma-separated list of tasks')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for evaluation')
    parser.add_argument('--output-path', type=str, default=None,
                        help='Path to save results')
    parser.add_argument('--model', type=str, default='star_moe',
                        help='Model name')
    
    # Parse only known args (rest go to Megatron)
    eval_args, remaining = parser.parse_known_args()
    
    # Put remaining args back for Megatron
    sys.argv = [sys.argv[0]] + remaining
    
    print("=" * 80)
    print("STAR MoE Model Evaluation")
    print("=" * 80)
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=eval_args.model,
        model_args=f"batch_size={eval_args.batch_size}",
        tasks=eval_args.tasks.split(','),
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    for task_name, task_results in results['results'].items():
        if 'acc' in task_results:
            accuracy = task_results['acc'] * 100
            print(f"{task_name:20s}: {accuracy:.2f}%")
        elif 'acc_norm' in task_results:
            accuracy = task_results['acc_norm'] * 100
            print(f"{task_name:20s}: {accuracy:.2f}%")
    
    # Save results
    if eval_args.output_path:
        import json
        os.makedirs(os.path.dirname(eval_args.output_path), exist_ok=True)
        with open(eval_args.output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to: {eval_args.output_path}")
    
    print("=" * 80)