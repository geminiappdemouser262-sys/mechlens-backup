"""Gemma-3 multimodal: RMSNorm (1 + w) offsets must stay out of the SigLIP vision LayerNorms.

Guards the fix in upstream PR #1897. The Gemma-3 language model uses RMSNorm with the
``(1 + weight)`` convention (``cfg.rmsnorm_uses_offset = True``) and the adapter's
conversion table adds 1.0 to every LM norm weight. The SigLIP vision tower uses plain
``nn.LayerNorm`` (mean-centred, with bias, weight used as-is). Three things can go wrong:

  1. an ADDITION(1.0) conversion keyed at a vision path,
  2. a vision LayerNorm wrapped as RMSNorm because the LM-level ``uses_rms_norm=True`` leaks,
  3. compatibility mode (fold_ln etc.) touching vision-tower parameters.

Everything runs offline on CPU with a tiny synthetic ``Gemma3Config`` (text + vision);
no Hub download, no HF_TOKEN.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Gemma3Config, Gemma3ForConditionalGeneration

from transformer_lens.conversion_utils.conversion_steps import ArithmeticTensorConversion
from transformer_lens.conversion_utils.conversion_steps.arithmetic_tensor_conversion import (
    OperationTypes,
)
from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.normalization import (
    NormalizationBridge,
)
from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module
from transformer_lens.model_bridge.supported_architectures.gemma3_multimodal import (
    Gemma3MultimodalArchitectureAdapter,
)

TOL = 1e-5
IMAGE_SIZE = 16
PATCH_SIZE = 8  # -> 4 patches per image


def _tiny_cfg() -> Gemma3Config:
    text_config = dict(
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
    vision_config = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        layer_norm_eps=1e-6,
        attn_implementation="eager",
    )
    return Gemma3Config(
        text_config=text_config,
        vision_config=vision_config,
        mm_tokens_per_image=4,
        image_token_index=255,
        boi_token_index=253,
        eoi_token_index=254,
    )


@pytest.fixture(scope="module")
def models() -> tuple[TransformerBridge, Gemma3ForConditionalGeneration]:
    torch.manual_seed(7)
    cfg = _tiny_cfg()
    hf_ref = Gemma3ForConditionalGeneration(copy.deepcopy(cfg)).to(torch.float32).eval()
    # Make the vision LayerNorm weights/biases clearly non-trivial so "(1 + w)" vs "w"
    # and "with bias" vs "no bias" produce visibly different outputs.
    with torch.no_grad():
        for name, p in hf_ref.model.vision_tower.named_parameters():
            if "layernorm" in name or "layer_norm" in name:
                p.copy_(torch.randn_like(p) * 0.5 + (1.0 if name.endswith("weight") else 0.0))

    hf_for_bridge = copy.deepcopy(hf_ref).eval()
    hf_for_bridge.load_state_dict(copy.deepcopy(hf_ref.state_dict()), strict=True)

    bridge = build_bridge_from_module(
        hf_for_bridge,
        architecture="Gemma3ForConditionalGeneration",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
        dtype=torch.float32,
        model_name="google/gemma-3-4b-pt",
    ).eval()
    bridge.adapter.setup_component_testing(hf_for_bridge, bridge_model=bridge)
    return bridge, hf_ref


@pytest.fixture(scope="module")
def pixel_values() -> torch.Tensor:
    torch.manual_seed(11)
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _post_ln(bridge: TransformerBridge) -> NormalizationBridge:
    enc = bridge.vision_encoder
    return enc.submodules["post_layernorm"] if "post_layernorm" in enc.submodules else enc.post_layernorm


def _last_hidden(out):
    out = getattr(out, "last_hidden_state", out)
    return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------- 1. conversion table --------


def test_no_offset_conversion_targets_vision_paths(models):
    bridge, _ = models
    adapter = bridge.adapter
    assert isinstance(adapter, Gemma3MultimodalArchitectureAdapter)
    assert adapter.cfg.rmsnorm_uses_offset is True  # LM side still needs the offset
    offset_keys = []
    for key, conv in adapter.weight_processing_conversions.items():
        tc = getattr(conv, "tensor_conversion", None)
        if isinstance(tc, ArithmeticTensorConversion) and tc.operation == OperationTypes.ADDITION:
            offset_keys.append(key)
        assert not key.startswith(("vision_encoder", "vision_projector", "model.vision_tower")), (
            f"conversion table must not touch vision paths, found {key!r}"
        )
    # Sanity: the LM offsets are present (otherwise the table itself regressed).
    assert "ln_final.weight" in offset_keys
    assert any(k.startswith("blocks.{i}.ln") for k in offset_keys)
    assert all(not ("vision" in k or "layer_norm" in k or "layernorm" in k) for k in offset_keys)


# ---------------------------------------------------------------- 2. vision norms are LayerNorm


def test_vision_post_layernorm_is_layernorm_not_rmsnorm(models):
    bridge, hf_ref = models
    post_ln = _post_ln(bridge)
    assert isinstance(post_ln, NormalizationBridge)
    assert post_ln.uses_rms_norm is False, "LM uses_rms_norm=True leaked into the SigLIP post_layernorm"
    assert isinstance(post_ln.original_component, torch.nn.LayerNorm)


def test_vision_tower_output_matches_hf(models, pixel_values):
    bridge, hf_ref = models
    with torch.no_grad():
        ref = hf_ref.model.vision_tower(pixel_values.clone()).last_hidden_state
        got = bridge.vision_encoder(pixel_values.clone())
    got = _last_hidden(got)
    assert got.shape == ref.shape == (1, (IMAGE_SIZE // PATCH_SIZE) ** 2, 32)
    assert torch.isfinite(got).all()
    assert _maxdiff(got, ref) < TOL, f"vision tower max|diff|={_maxdiff(got, ref):.2e}"


def test_vision_post_layernorm_math_is_layernorm(models, pixel_values):
    """hook_out of the SigLIP post_layernorm == LayerNorm(x) (centred, w*x+b), != RMSNorm(x) or (1+w)."""
    bridge, hf_ref = models
    ln = hf_ref.model.vision_tower.vision_model.post_layernorm
    captured: dict[str, torch.Tensor] = {}
    h_in = ln.register_forward_pre_hook(lambda m, inp: captured.__setitem__("x", inp[0].detach().clone()))
    h_out = ln.register_forward_hook(lambda m, inp, out: captured.__setitem__("y", out.detach().clone()))
    hooks = {}
    try:
        with torch.no_grad():
            hf_ref.model.vision_tower(pixel_values.clone())
            _post_ln(bridge).hook_in.add_hook(
                lambda t, hook: hooks.__setitem__("in", t.detach().clone())
            )
            _post_ln(bridge).hook_out.add_hook(
                lambda t, hook: hooks.__setitem__("out", t.detach().clone())
            )
            bridge.vision_encoder(pixel_values.clone())
    finally:
        h_in.remove()
        h_out.remove()
        _post_ln(bridge).hook_in.remove_hooks()
        _post_ln(bridge).hook_out.remove_hooks()

    x, y_hf = captured["x"], captured["y"]
    assert _maxdiff(hooks["in"], x) < TOL
    assert _maxdiff(hooks["out"], y_hf) < TOL, f"post_layernorm max|diff|={_maxdiff(hooks['out'], y_hf):.2e}"

    # Independent reference: proper LayerNorm.
    expected = torch.nn.functional.layer_norm(x, x.shape[-1:], ln.weight, ln.bias, ln.eps)
    assert _maxdiff(hooks["out"], expected) < TOL

    # Non-tautologies: the two wrong formulas must be visibly different.
    rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + ln.eps) * ln.weight
    offset = torch.nn.functional.layer_norm(x, x.shape[-1:], 1.0 + ln.weight, ln.bias, ln.eps)
    assert _maxdiff(hooks["out"], rms) > 1e-3, "post_layernorm is computing RMSNorm"
    assert _maxdiff(hooks["out"], offset) > 1e-3, "post_layernorm is applying the (1 + w) offset"


def test_text_only_logits_parity_on_multimodal_model(models):
    """Sanity that the LM half (where the offsets *are* applied) still matches HF."""
    bridge, hf_ref = models
    tokens = torch.tensor([[2, 5, 12, 23, 34, 45]], dtype=torch.long)  # no image tokens
    with torch.no_grad():
        hf_logits = hf_ref(input_ids=tokens.clone()).logits
        bridge_logits = bridge(tokens.clone(), return_type="logits")
    assert bridge_logits.shape == hf_logits.shape
    assert _maxdiff(bridge_logits, hf_logits) < 1e-4


# ---------------------------------------------------------------- 3. compatibility mode --------


def test_compatibility_mode_leaves_vision_tower_untouched(models, pixel_values):
    bridge, hf_ref = models
    before = {
        n: p.detach().clone() for n, p in bridge.original_model.model.vision_tower.named_parameters()
    }
    with torch.no_grad():
        out_before = bridge.vision_encoder(pixel_values.clone())
    out_before = _last_hidden(out_before)

    bridge.enable_compatibility_mode()

    after = dict(bridge.original_model.model.vision_tower.named_parameters())
    assert set(after) == set(before)
    for n, p in before.items():
        assert torch.equal(after[n].detach(), p), f"compatibility mode modified vision param {n}"
    with torch.no_grad():
        out_after = bridge.vision_encoder(pixel_values.clone())
    out_after = _last_hidden(out_after)
    assert _maxdiff(out_after, out_before) == 0.0

    # And the vision norms are still LayerNorm afterwards.
    assert _post_ln(bridge).uses_rms_norm is False
