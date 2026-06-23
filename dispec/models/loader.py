"""Model + tokenizer loading helpers.

Thin wrappers over HuggingFace. We use HF *model definitions* for the raw forward
pass; everything around them (KV cache, scheduling, sampling, speculation) is
implemented from scratch in DiSpec.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dispec.config import DRAFT_MODEL, DTYPE, TARGET_MODEL

_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(name: str, dtype: str = DTYPE, device: str = "cuda"):
    """Load a causal LM in eval mode on the given device."""
    model = AutoModelForCausalLM.from_pretrained(
        name,
        dtype=_DTYPE_MAP[dtype],
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_tokenizer(name: str):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_target(dtype: str = DTYPE, device: str = "cuda"):
    return load_model(TARGET_MODEL, dtype, device), load_tokenizer(TARGET_MODEL)


def load_draft(dtype: str = DTYPE, device: str = "cuda"):
    return load_model(DRAFT_MODEL, dtype, device), load_tokenizer(DRAFT_MODEL)


def build_prompt(tokenizer, user_msg: str) -> str:
    """Apply the chat template so instruct models behave."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
