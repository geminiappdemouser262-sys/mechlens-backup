"""T5Gemma 2 architecture adapter (SCAFFOLD — see CONFIRM markers before enabling).

T5Gemma2ForConditionalGeneration: Gemma-3-derived encoder-decoder.
  * Encoder: bidirectional Gemma-3 blocks (QK-norm, pre+post RMSNorm, GQA, local/global RoPE)
    + SigLIP vision tower; image tokens are encoder positions.
  * Decoder: ONE merged attention per block (K/V over [decoder ; encoder]) + gated MLP.
  * Embeddings: encoder embed, decoder embed and lm_head share ONE matrix.

Structure follows t5gemma.py (encoder-decoder skeleton) and gemma3_multimodal.py (block
internals, vision). Everything that depends on HF attribute names is in ``_HF`` so it can be
fixed in one place after reading transformers/models/t5gemma2/modeling_t5gemma2.py.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from transformer_lens.conversion_utils.conversion_steps import (
    ArithmeticTensorConversion,
    RearrangeTensorConversion,
    TransposeTensorConversion,
)
from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    OperationTypes,
)
from transformer_lens.conversion_utils.param_processing_conversion import (
    ParamProcessingConversion,
)
from transformer_lens.model_bridge.architecture_adapter import ArchitectureAdapter
from transformer_lens.model_bridge.generalized_components import (
    BlockBridge,
    EmbeddingBridge,
    GatedMLPBridge,
    LinearBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
    SiglipVisionEncoderBridge,
    UnembeddingBridge,
    VisionProjectionBridge,
)
from transformer_lens.model_bridge.generalized_components.merged_attention import (
    MergedAttentionBridge,
)
from transformer_lens.model_bridge.generalized_components.position_embeddings_attention import (
    PositionEmbeddingsAttentionBridge,
)

# --------------------------------------------------------------------------------------------
# HF module paths. Every entry is a best guess from T5Gemma / Gemma-3 naming.  # CONFIRM ALL
# Derive with:  for n, _ in model.named_modules(): print(n)   on google/t5gemma-2-270m-270m
# --------------------------------------------------------------------------------------------
_HF: Dict[str, str] = {
    "encoder_embed": "model.encoder.embed_tokens",
    "encoder_vision_tower": "model.encoder.vision_tower",
    "encoder_projector": "model.encoder.multi_modal_projector",  # may not exist (paper mentions none)
    "encoder_rotary": "model.encoder.rotary_emb",
    "encoder_rotary_local": "model.encoder.rotary_emb_local",  # Gemma-3 dual RoPE; may be absent
    "encoder_layers": "model.encoder.layers",
    "encoder_norm": "model.encoder.norm",
    "decoder_embed": "model.decoder.embed_tokens",
    "decoder_rotary": "model.decoder.rotary_emb",
    "decoder_rotary_local": "model.decoder.rotary_emb_local",
    "decoder_layers": "model.decoder.layers",
    "decoder_norm": "model.decoder.norm",
    "unembed": "lm_head",  # T5Gemma used lm_head.out_proj
    # per-layer submodules
    "self_attn": "self_attn",  # decoder merged attention may be named differently
    "ln1": "pre_self_attn_layernorm",  # Gemma-3 names: input_layernorm
    "ln1_post": "post_self_attn_layernorm",  #             post_attention_layernorm
    "ln2": "pre_feedforward_layernorm",
    "ln2_post": "post_feedforward_layernorm",
    "mlp": "mlp",
}


def _rearrange_qkvo(n_heads: int, n_kv_heads: int) -> Dict[str, ParamProcessingConversion]:
    """Standard HF -> TL rearranges for one attention module (q,k,v,o)."""
    return {
        "q": ParamProcessingConversion(tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_heads)),
        "k": ParamProcessingConversion(tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads)),
        "v": ParamProcessingConversion(tensor_conversion=RearrangeTensorConversion("(n h) m -> n m h", n=n_kv_heads)),
        "o": ParamProcessingConversion(tensor_conversion=RearrangeTensorConversion("m (n h) -> n h m", n=n_heads)),
    }


def _plus_one() -> ParamProcessingConversion:
    """Gemma RMSNorm stores w with the (1 + w) convention."""
    return ParamProcessingConversion(tensor_conversion=ArithmeticTensorConversion(OperationTypes.ADDITION, 1.0))


def _transpose() -> ParamProcessingConversion:
    return ParamProcessingConversion(tensor_conversion=TransposeTensorConversion())


class T5Gemma2ArchitectureAdapter(ArchitectureAdapter):
    """Architecture adapter for T5Gemma2ForConditionalGeneration."""

    # SigLIP image processing needs torchvision (same as Gemma 3 multimodal / Gemma 3n).
    required_libraries = ["torchvision"]
    required_libraries_group = "multimodal"

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)

        # fold_ln is impossible for the merged attention: the shared W_k / W_v would have to
        # absorb BOTH the decoder pre-norm (for the X slice) and encoder_ln_final (for the H
        # slice). The 3-way tied embedding rules it out on the unembed side as well.
        self.supports_fold_ln = False
        self.supports_center_writing_weights = False  # RMS model; no-op anyway

        self.cfg.is_multimodal = True
        self.cfg.is_encoder_decoder = True
        self.cfg.normalization_type = "RMS"
        self.cfg.uses_rms_norm = True
        self.cfg.rmsnorm_uses_offset = True  # (1 + w), encoder AND decoder, incl. q_norm/k_norm
        self.cfg.final_rms = True
        self.cfg.positional_embedding_type = "rotary"
        self.cfg.gated_mlp = True
        self.cfg.attn_only = False
        self.cfg.act_fn = "gelu_pytorch_tanh"
        self.cfg.attn_implementation = "eager"  # output_attentions -> hook_pattern
        self.cfg.use_qk_norm = True  # CONFIRM attribute consumed by PositionEmbeddingsAttentionBridge
        # Gemma-3 scales attention by query_pre_attn_scalar**-0.5, not d_head**-0.5.
        qpa = getattr(cfg, "query_pre_attn_scalar", None)
        if qpa is not None:
            self.cfg.attn_scale = float(qpa) ** 0.5  # CONFIRM: copy whatever gemma3.py does
        # Encoder/decoder may differ in depth; decoder drives cfg.n_layers (LM side).
        self.cfg.encoder_n_layers = getattr(cfg, "encoder_n_layers", self.cfg.n_layers)
        self.cfg.mm_tokens_per_image = getattr(cfg, "mm_tokens_per_image", 256)
        if hasattr(cfg, "vision_config") and cfg.vision_config is not None:
            self.cfg.vision_hidden_size = getattr(cfg.vision_config, "hidden_size", None)
            self.cfg.vision_num_layers = getattr(cfg.vision_config, "num_hidden_layers", None)
            self.cfg.vision_num_heads = getattr(cfg.vision_config, "num_attention_heads", None)

        n_heads = self.cfg.n_heads
        n_kv = getattr(self.cfg, "n_key_value_heads", None) or n_heads

        # ---------------------------------------------------------------- conversions ---------
        conv: Dict[str, ParamProcessingConversion] = {}
        for prefix in ("encoder_blocks", "decoder_blocks"):
            qkvo = _rearrange_qkvo(n_heads, n_kv)
            for slot in ("q", "k", "v", "o"):
                conv[f"{prefix}.{{i}}.attn.{slot}.weight"] = qkvo[slot]
            for ln in ("ln1", "ln1_post", "ln2", "ln2_post", "attn.q_norm", "attn.k_norm"):
                conv[f"{prefix}.{{i}}.{ln}.weight"] = _plus_one()
            for slot in ("gate", "in", "out"):
                conv[f"{prefix}.{{i}}.mlp.{slot}.weight"] = _transpose()
        conv["encoder_ln_final.weight"] = _plus_one()
        conv["decoder_ln_final.weight"] = _plus_one()
        conv["unembed.weight"] = _transpose()
        # Deliberately NO entries for vision_encoder.* / vision_projector.* — SigLIP uses plain
        # LayerNorm; the (1 + w) offset must not leak (see test_gemma3_vision_layernorm_offsets).
        # Deliberately NO decoder_blocks.{i}.cross_attn.* — there is no cross-attention module.
        # Note the SAME k/v conversion key covers both the decoder and encoder slices: one weight.
        self.weight_processing_conversions = conv

        # ---------------------------------------------------------------- components ----------
        def _attn_block(attn_bridge: Any) -> Dict[str, Any]:
            return {
                "ln1": RMSNormalizationBridge(name=_HF["ln1"], config=self.cfg),
                "ln1_post": RMSNormalizationBridge(name=_HF["ln1_post"], config=self.cfg),
                "attn": attn_bridge,
                "ln2": RMSNormalizationBridge(name=_HF["ln2"], config=self.cfg),
                "ln2_post": RMSNormalizationBridge(name=_HF["ln2_post"], config=self.cfg),
                "mlp": GatedMLPBridge(
                    name=_HF["mlp"],
                    config=self.cfg,
                    submodules={
                        "gate": LinearBridge(name="gate_proj"),
                        "in": LinearBridge(name="up_proj"),
                        "out": LinearBridge(name="down_proj"),
                    },
                ),
            }

        def _qkvo_norm_submodules() -> Dict[str, Any]:
            return {
                "q": LinearBridge(name="q_proj"),
                "k": LinearBridge(name="k_proj"),
                "v": LinearBridge(name="v_proj"),
                "o": LinearBridge(name="o_proj"),
                "q_norm": RMSNormalizationBridge(name="q_norm", config=self.cfg),
                "k_norm": RMSNormalizationBridge(name="k_norm", config=self.cfg),
            }

        encoder_attn = PositionEmbeddingsAttentionBridge(
            name=_HF["self_attn"],
            config=self.cfg,
            submodules=_qkvo_norm_submodules(),
            requires_attention_mask=True,
            requires_position_embeddings=True,
            is_causal=False,  # bidirectional encoder
        )
        decoder_attn = MergedAttentionBridge(
            name=_HF["self_attn"],
            config=self.cfg,
            submodules=_qkvo_norm_submodules(),
            is_causal=True,  # over the decoder slice; encoder slice fully visible  # CONFIRM
        )

        self.component_mapping = {
            # ---- encoder (text + vision) ----
            "encoder_embed": EmbeddingBridge(name=_HF["encoder_embed"]),
            "vision_encoder": SiglipVisionEncoderBridge(name=_HF["encoder_vision_tower"], config=self.cfg),
            "vision_projector": VisionProjectionBridge(name=_HF["encoder_projector"]),  # CONFIRM exists
            "encoder_rotary_emb": RotaryEmbeddingBridge(name=_HF["encoder_rotary"]),
            "encoder_blocks": BlockBridge(name=_HF["encoder_layers"], config=self.cfg, submodules=_attn_block(encoder_attn)),
            "encoder_ln_final": RMSNormalizationBridge(name=_HF["encoder_norm"], config=self.cfg),
            # ---- decoder (merged attention) ----
            "decoder_embed": EmbeddingBridge(name=_HF["decoder_embed"]),  # same tensor as encoder_embed
            "decoder_rotary_emb": RotaryEmbeddingBridge(name=_HF["decoder_rotary"]),
            # Two sub-blocks (attn, mlp) => BlockBridge's resid_pre/mid/post hooks are exactly right.
            # Only write a T5Gemma2DecoderBlockBridge if BlockBridge cannot pass
            # encoder_hidden_states through to the attention module.  # CONFIRM
            "decoder_blocks": BlockBridge(name=_HF["decoder_layers"], config=self.cfg, submodules=_attn_block(decoder_attn)),
            "decoder_ln_final": RMSNormalizationBridge(name=_HF["decoder_norm"], config=self.cfg),
            "unembed": UnembeddingBridge(name=_HF["unembed"], config=self.cfg),
        }

    # -------------------------------------------------------------------- weight processing --
    #: TL-format keys that alias ONE underlying HF tensor (3-way tie).
    TIED_EMBED_KEYS = ("encoder_embed.weight", "decoder_embed.weight", "unembed.weight")

    def preprocess_weights(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Untie the 3-way embedding and refuse fold_ln explicitly.

        TransformerBridge.process_weights already clones unembed when it shares storage with
        embed.weight; T5Gemma2 has a THIRD alias (encoder_embed) and different key names, so
        untie here: every processing step on ``unembed.weight`` must leave both embeddings
        bit-identical. The clones are only taken when storage is actually shared, so untied
        fine-tunes pass through unchanged.
        """
        if getattr(self, "_fold_ln_requested", False):
            raise ValueError(
                "T5Gemma2 does not support fold_ln: the decoder's merged attention shares W_k/W_v "
                "between the decoder slice (pre-norm ln1) and the encoder slice (encoder_ln_final), "
                "so a single norm cannot be folded into them. Call "
                "enable_compatibility_mode(fold_ln=False) (center_unembed etc. still work)."
            )
        present = [k for k in self.TIED_EMBED_KEYS if k in state_dict]
        if len(present) >= 2:
            base = state_dict[present[0]]
            for k in present[1:]:
                if state_dict[k].data_ptr() == base.data_ptr():
                    state_dict[k] = state_dict[k].clone()
        return state_dict

    # -------------------------------------------------------------------- testing hooks -------
    def setup_component_testing(self, hf_model: Any, bridge_model: Any = None) -> None:
        """Wire encoder/decoder rotary embeddings and eager attention for component tests.

        Mirrors t5gemma.py; additionally records on each MergedAttentionBridge that the
        encoder slice of K is (or is not) rotated, once CONFIRMed from HF.
        """
        if hasattr(hf_model, "config") and hasattr(hf_model.config, "_attn_implementation"):
            hf_model.config._attn_implementation = "eager"

        def _get(path: str) -> Any:
            cur: Any = hf_model
            for part in path.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    return None
            return cur

        enc_rot, dec_rot = _get(_HF["encoder_rotary"]), _get(_HF["decoder_rotary"])
        enc_rot_local, dec_rot_local = _get(_HF["encoder_rotary_local"]), _get(_HF["decoder_rotary_local"])
        if bridge_model is not None:
            for block in getattr(bridge_model, "encoder_blocks", []):
                block.attn._rotary_emb = enc_rot
                block.attn._rotary_emb_local = enc_rot_local  # CONFIRM attribute name used by the bridge
            for block in getattr(bridge_model, "decoder_blocks", []):
                block.attn._rotary_emb = dec_rot
                block.attn._rotary_emb_local = dec_rot_local
                block.attn.encoder_slice_rotated = False  # CONFIRM from modeling_t5gemma2.py
        else:
            for name in ("encoder_blocks.0.attn", "decoder_blocks.0.attn"):
                comp = self.get_generalized_component(name)
                if comp is not None:
                    comp._rotary_emb = enc_rot if name.startswith("encoder") else dec_rot
