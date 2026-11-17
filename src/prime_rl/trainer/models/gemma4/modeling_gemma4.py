"""Gemma 4 (dense, with vision and audio towers) for the PrimeRL trainer.

Lumbridge addition. The layers are transformers' own Gemma 4 modules; this file only adapts them to the
trainer's contract:

- packed rows: documents are concatenated in one row and described by ``seq_lens``. We build the
  block-diagonal causal mask (and the 512-token sliding-window variant) ourselves and hand it to the HF
  model as a ready mask mapping, so no document attends to another.
- attention runs through SDPA. Gemma 4's global layers use head_dim 512, which flash-attention 2 does
  not support, so the trainer's ``attn`` setting is ignored for this model.
- module names match the HF checkpoint exactly (``model.language_model.*``, ``model.vision_tower.*``,
  ``model.audio_tower.*``), so weights load through DCP with no conversion.
- the LM head is tied to the input embedding and keeps Gemma's final-logit softcap, which the PrimeRL
  head applies.

The checkpoint also carries k/v projections for the KV-shared layers (``model-frozen-shared-kv``); HF never
builds those modules, and DCP only reads the keys the model asks for, so they are skipped.
"""

import torch
from torch import Tensor, nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4Model, Gemma4PreTrainedModel

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL

ATTN_IMPLEMENTATION = "sdpa"


def positions_from_seq_lens(seq_lens: Tensor, total: int) -> tuple[Tensor, Tensor]:
    """Per-token position within its document, and the document index of each token, for one packed row."""
    seq_lens = seq_lens.to(torch.long)
    if int(seq_lens.sum()) != total:
        raise ValueError(f"seq_lens sum to {int(seq_lens.sum())} but the row has {total} tokens")
    doc_ids = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=seq_lens.device), seq_lens)
    starts = torch.cumsum(seq_lens, 0) - seq_lens
    positions = torch.arange(total, device=seq_lens.device) - starts[doc_ids]
    return positions, doc_ids


def packed_causal_masks(doc_ids: Tensor, sliding_window: int | None) -> dict[str, Tensor]:
    """Boolean SDPA masks (True = may attend) of shape (1, 1, T, T) for full and sliding-window layers."""
    idx = torch.arange(doc_ids.numel(), device=doc_ids.device)
    q, k = idx[:, None], idx[None, :]
    full = (doc_ids[:, None] == doc_ids[None, :]) & (k <= q)
    masks = {"full_attention": full[None, None]}
    if sliding_window is not None:
        # Same rule as HF's sliding_window_overlay: the key is within the last `sliding_window` positions.
        masks["sliding_attention"] = (full & (k > q - sliding_window))[None, None]
    return masks


class Gemma4PrimeModel(Gemma4Model):
    """HF ``Gemma4Model`` with the PrimeRL packed-batch forward."""

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        routed_experts: torch.LongTensor | None = None,
        *,
        seq_lens: torch.LongTensor,
        seq_lens_are_pre_shard: bool = False,
        **mm_kwargs,
    ):
        if routed_experts is not None:
            raise ValueError("Gemma 4 dense has no experts to route")
        if seq_lens_are_pre_shard:
            raise ValueError("Gemma 4 does not support context parallelism in lumbridge-rl yet")
        reference = input_ids if input_ids is not None else inputs_embeds
        if reference.shape[0] != 1:
            raise ValueError(f"PrimeRL packs one row per micro-batch; got batch size {reference.shape[0]}")

        positions, doc_ids = positions_from_seq_lens(seq_lens.to(reference.device), reference.shape[1])
        text_config = self.config.get_text_config()
        masks = packed_causal_masks(doc_ids, getattr(text_config, "sliding_window", None))

        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=positions[None],
            attention_mask=masks,
            use_cache=False,
            **mm_kwargs,
        )


class Gemma4ForConditionalGeneration(PreTrainedModelPrimeRL, Gemma4PreTrainedModel):
    config_class = Gemma4Config
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Gemma4Config):
        _set_attn_implementation(config)
        super().__init__(config)
        text_config = config.get_text_config()
        self.model = Gemma4PrimeModel(config)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
        # inject_prime_lm_head keeps this softcap on the PrimeRL head
        self.lm_head.softcap = text_config.final_logit_softcapping
        self.post_init()
        self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False, allow_all_kernels: bool = False
    ) -> str:
        return ATTN_IMPLEMENTATION

    @classmethod
    def cp_blocker(cls, config, cp: int) -> str | None:
        return "Gemma 4 packs documents with explicit SDPA masks, which lumbridge-rl does not shard yet"

    @classmethod
    def is_hf_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        return True

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, Tensor]) -> bool:
        # Same key names as the HF checkpoint, so DCP loads directly with no conversion step.
        return False

    @torch.no_grad()
    def init_buffers_post_meta(self) -> None:
        # Non-persistent buffers (embedding scales, RoPE frequencies, audio timescales and softcap) are not in the
        # checkpoint. Gemma 4's own init fills them; it also touches that module's weights, which load afterwards.
        for module in self.modules():
            if module._non_persistent_buffers_set:
                Gemma4PreTrainedModel._init_weights(self, module)


def _set_attn_implementation(config: Gemma4Config) -> None:
    config._attn_implementation = ATTN_IMPLEMENTATION
    # The audio tower has its own chunked attention and supports only eager.
    for key, impl in (
        ("text_config", ATTN_IMPLEMENTATION),
        ("vision_config", ATTN_IMPLEMENTATION),
        ("audio_config", "eager"),
    ):
        sub = getattr(config, key, None)
        if sub is not None:
            sub._attn_implementation = impl


__all__ = ["Gemma4ForConditionalGeneration", "Gemma4PrimeModel", "packed_causal_masks", "positions_from_seq_lens"]
