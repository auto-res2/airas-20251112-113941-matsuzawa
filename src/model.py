import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple

__all__ = ["load_model_and_tokenizer"]

# --------------------------------------------------------------------------------
#  4-bit QLoRA BACKBONE LOADING
# --------------------------------------------------------------------------------

def _bnb_config(cfg):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.model.quantization.scheme,
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def _load_backbone(cfg, device):
    return AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        cache_dir=".cache",
        torch_dtype=torch.bfloat16 if cfg.model.dtype == "bfloat16" else torch.float16,
        quantization_config=_bnb_config(cfg),
        device_map="auto" if device.type == "cuda" else {"": device.type},
    )

# --------------------------------------------------------------------------------
#  PUBLIC API
# --------------------------------------------------------------------------------

def load_model_and_tokenizer(cfg, tokenizer=None, device: torch.device | None = None) -> Tuple["transformers.PreTrainedModel", "transformers.PreTrainedModel", "transformers.PreTrainedTokenizer"]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, cache_dir=".cache", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    base_model = _load_backbone(cfg, device)
    base_model.eval().requires_grad_(False)

    trainable = _load_backbone(cfg, device)
    lora_cfg = LoraConfig(
        r=cfg.model.peft.rank,
        lora_alpha=cfg.model.peft.alpha,
        lora_dropout=cfg.model.peft.dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    trainable = get_peft_model(trainable, lora_cfg)
    trainable.to(device)
    trainable.print_trainable_parameters()

    return trainable, base_model, tokenizer