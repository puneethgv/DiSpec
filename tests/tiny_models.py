"""Tiny randomly initialised models for tests that need a real architecture but no weights.

float32 on CPU, so comparisons can use tight tolerances and run anywhere. The output
head is re-initialised with a large scale: HF's default std of 0.02 leaves logits so
flat that argmax near-ties would make token-level comparisons flaky for no real reason.
"""

from __future__ import annotations

import torch

_DIMS = dict(vocab_size=128, hidden_size=64, intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
             tie_word_embeddings=False)


def _finish(model, seed: int):
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.randn(model.lm_head.weight.shape, generator=g))
    model.eval()
    model.requires_grad_(False)
    return model


def tiny_qwen3(num_layers: int = 2, seed: int = 0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(num_hidden_layers=num_layers, **_DIMS))
    # Non-unit q/k norm gains, so a forward that skipped the norms could not match HF
    # by accident of an all-ones initialisation.
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        for layer in model.model.layers:
            for norm in (layer.self_attn.q_norm, layer.self_attn.k_norm):
                norm.weight.copy_(1.0 + 0.3 * torch.randn(norm.weight.shape, generator=g))
    return _finish(model, seed)


def tiny_qwen2(num_layers: int = 2, seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    return _finish(Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=num_layers, **_DIMS)), seed)
