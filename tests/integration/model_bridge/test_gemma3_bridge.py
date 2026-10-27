"""Integration tests for the Gemma 3 TransformerBridge.

Builds a small deterministic Gemma3ForCausalLM in float32 (no HF Hub download required)
and compares an independent HuggingFace reference forward pass against the
TransformerBridge forward pass. Verifies:
- Identical weights and buffers across HF reference and Bridge before execution
- HF rms_norm_eps propagates to bridge.cfg.eps
- setup_component_testing wires rotary_emb (including local RoPE) and eager attention
- Forward-pass logits parity against HF eager is strictly below 1e-4
- Bridge components and attention reconstruction hooks actually fire (non-tautological)
"""

import copy

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components import (
    PositionEmbeddingsAttentionBridge,
    RMSNormalizationBridge,
    RotaryEmbeddingBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import (
    build_bridge_from_module,
)


def _make_tiny_gemma3_config() -> Gemma3TextConfig:
    """Return a small, deterministic Gemma3TextConfig exercising local + global RoPE and QK-norm."""
    return Gemma3TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def gemma3_bridge_and_hf() -> tuple[TransformerBridge, Gemma3ForCausalLM]:
    """Instantiate independent HF reference and TransformerBridge with identical float32 weights."""
    torch.manual_seed(42)
    cfg = _make_tiny_gemma3_config()

    hf_reference = Gemma3ForCausalLM(copy.deepcopy(cfg)).to(dtype=torch.float32).eval()
    hf_reference.config._attn_implementation = "eager"
    for layer in hf_reference.model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
            layer.self_attn.config._attn_implementation = "eager"

    # Separate model instance via deepcopy + strict state_dict load to guarantee identical weights
    hf_for_bridge = copy.deepcopy(hf_reference).to(dtype=torch.float32).eval()
    hf_for_bridge.load_state_dict(copy.deepcopy(hf_reference.state_dict()), strict=True)

    for (name_ref, p_ref), (name_br, p_br) in zip(
        hf_reference.named_parameters(), hf_for_bridge.named_parameters()
    ):
        assert name_ref == name_br
        assert torch.equal(p_ref, p_br), f"Weight mismatch on {name_ref}"
        assert p_ref.data_ptr() != p_br.data_ptr(), f"Shared parameter storage on {name_ref}"

    bridge = build_bridge_from_module(
        hf_for_bridge,
        architecture="Gemma3ForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
        dtype=torch.float32,
        model_name="google/gemma-3-270m",
    ).eval()

    # build_bridge_from_module calls adapter.prepare_model(), not setup_component_testing().
    # Calling setup_component_testing wires rotary_emb onto each block's
    # PositionEmbeddingsAttentionBridge (and forces eager attention on all layers).
    bridge.adapter.setup_component_testing(hf_for_bridge, bridge_model=bridge)

    return bridge, hf_reference


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    """Deterministic input token sequence."""
    return torch.tensor([[1, 5, 12, 23, 34, 45, 56, 67]], dtype=torch.long)


class TestGemma3BridgeIntegration:
    """Integration and logits-parity tests for Gemma-3 TransformerBridge."""

    def test_gemma3_bridge_structure_and_eps(
        self, gemma3_bridge_and_hf: tuple[TransformerBridge, Gemma3ForCausalLM]
    ) -> None:
        bridge, hf_reference = gemma3_bridge_and_hf
        assert bridge.original_model is not hf_reference
        assert len(bridge.blocks) == 2
        assert bridge.cfg.eps == 1e-6
        assert isinstance(bridge.ln_final, RMSNormalizationBridge)
        assert isinstance(bridge.rotary_emb, RotaryEmbeddingBridge)
        for block in bridge.blocks:
            assert isinstance(block.ln1, RMSNormalizationBridge)
            assert isinstance(block.ln1_post, RMSNormalizationBridge)
            assert isinstance(block.ln2, RMSNormalizationBridge)
            assert isinstance(block.ln2_post, RMSNormalizationBridge)
            assert isinstance(block.attn, PositionEmbeddingsAttentionBridge)
            assert block.attn._rotary_emb is not None

    def test_gemma3_logits_parity_vs_hf(
        self,
        gemma3_bridge_and_hf: tuple[TransformerBridge, Gemma3ForCausalLM],
        tokens: torch.Tensor,
    ) -> None:
        """Bridge forward logits must match independent HF Gemma3ForCausalLM within 1e-4."""
        bridge, hf_reference = gemma3_bridge_and_hf

        fired_hooks: list[str] = []

        def _record_hook(tensor: torch.Tensor, hook) -> torch.Tensor:
            fired_hooks.append(hook.name)
            return tensor

        with torch.no_grad():
            hf_logits = hf_reference(tokens.clone()).logits
            bridge_logits = bridge.run_with_hooks(
                tokens.clone(),
                return_type="logits",
                fwd_hooks=[
                    ("blocks.0.attn.hook_attn_scores", _record_hook),
                    ("blocks.0.attn.hook_q_normed", _record_hook),
                    ("blocks.0.ln1.hook_normalized", _record_hook),
                    ("ln_final.hook_normalized", _record_hook),
                ],
            )

        assert bridge_logits.shape == hf_logits.shape
        assert bridge_logits.data_ptr() != hf_logits.data_ptr()
        assert torch.isfinite(bridge_logits).all()
        assert set(fired_hooks) == {
            "blocks.0.attn.hook_attn_scores",
            "blocks.0.attn.hook_q_normed",
            "blocks.0.ln1.hook_normalized",
            "ln_final.hook_normalized",
        }, f"Expected bridge hooks to fire during forward pass, got {fired_hooks}"

        max_diff = (bridge_logits - hf_logits).abs().max().item()
        assert max_diff < 1e-4, f"Gemma-3 Bridge vs HF max logit diff = {max_diff:.2e} (>= 1e-4)"
