"""Gemma-3 RMSNorm hook parity: hook_scale / hook_normalized / hook_out vs. HuggingFace (Issue #1526).

Regression form of debugging/gemma3_norm_hook_parity.py. Builds a tiny deterministic
Gemma3ForCausalLM from Gemma3TextConfig (no Hub download, no HF_TOKEN, float32, eager)
and an independent HF reference with identical weights, then at every RMSNorm site
(ln1, ln1_post, ln2, ln2_post per block, ln_final) checks:

    hook_scale      == sqrt(mean(x^2) + eps)        (denominator convention, float32)
    hook_normalized == x * rsqrt(mean(x^2) + eps)   (pre-weight, HF's _norm(x))
    hook_out        == HF module output             ((1 + w) * normalized)

where x is the exact input the HF module received and eps is the HF module's own eps.
Uses a non-default rms_norm_eps so that an eps-propagation miss cannot pass by accident.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module

RMS_NORM_EPS = 0.125  # deliberately far from the 1e-5 / 1e-6 defaults
TOL = 1e-5


def _tiny_cfg() -> Gemma3TextConfig:
    return Gemma3TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=RMS_NORM_EPS,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        attn_implementation="eager",
    )


@pytest.fixture(scope="module")
def models() -> tuple[TransformerBridge, Gemma3ForCausalLM]:
    torch.manual_seed(42)
    cfg = _tiny_cfg()
    hf_ref = Gemma3ForCausalLM(copy.deepcopy(cfg)).to(torch.float32).eval()
    hf_ref.config._attn_implementation = "eager"

    hf_for_bridge = copy.deepcopy(hf_ref).eval()
    hf_for_bridge.load_state_dict(copy.deepcopy(hf_ref.state_dict()), strict=True)

    bridge = build_bridge_from_module(
        hf_for_bridge,
        architecture="Gemma3ForCausalLM",
        hf_config=copy.deepcopy(cfg),
        tokenizer=None,
        device="cpu",
        dtype=torch.float32,
        model_name="google/gemma-3-270m",
    ).eval()
    bridge.adapter.setup_component_testing(hf_for_bridge, bridge_model=bridge)
    return bridge, hf_ref


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[1, 5, 12, 23, 34, 45, 56, 67, 78, 89]], dtype=torch.long)


def _sites(hf: Gemma3ForCausalLM) -> dict[str, torch.nn.Module]:
    sites: dict[str, torch.nn.Module] = {}
    for i, layer in enumerate(hf.model.layers):
        sites[f"blocks.{i}.ln1"] = layer.input_layernorm
        sites[f"blocks.{i}.ln1_post"] = layer.post_attention_layernorm
        sites[f"blocks.{i}.ln2"] = layer.pre_feedforward_layernorm
        sites[f"blocks.{i}.ln2_post"] = layer.post_feedforward_layernorm
    sites["ln_final"] = hf.model.norm
    return sites


@pytest.fixture(scope="module")
def captured(models, tokens):
    """One forward pass each; HF norm inputs/outputs captured with hooks, bridge via run_with_cache."""
    bridge, hf_ref = models
    sites = _sites(hf_ref)
    hf_in: dict[str, torch.Tensor] = {}
    hf_out: dict[str, torch.Tensor] = {}
    handles = []
    for name, mod in sites.items():
        handles.append(
            mod.register_forward_pre_hook(
                lambda m, inp, n=name: hf_in.__setitem__(n, inp[0].detach().clone())
            )
        )
        handles.append(
            mod.register_forward_hook(
                lambda m, inp, out, n=name: hf_out.__setitem__(n, out.detach().clone())
            )
        )
    try:
        with torch.no_grad():
            hf_logits = hf_ref(tokens.clone()).logits
            bridge_logits, cache = bridge.run_with_cache(tokens.clone())
    finally:
        for h in handles:
            h.remove()
    assert set(hf_in) == set(sites), "HF pre-hooks did not fire at every norm site"
    return sites, hf_in, hf_out, hf_logits, bridge_logits, cache


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


SITE_NAMES = [f"blocks.{i}.{n}" for i in range(2) for n in ("ln1", "ln1_post", "ln2", "ln2_post")] + [
    "ln_final"
]


def test_eps_propagates_to_bridge_cfg(models):
    bridge, hf_ref = models
    assert bridge.cfg.eps == RMS_NORM_EPS
    assert hf_ref.config.rms_norm_eps == RMS_NORM_EPS


@pytest.mark.parametrize("site", SITE_NAMES)
def test_hook_scale_matches_sqrt_mean_sq_plus_eps(captured, site):
    sites, hf_in, _, _, _, cache = captured
    x = hf_in[site].float()
    eps = float(getattr(sites[site], "eps", getattr(sites[site], "variance_epsilon", RMS_NORM_EPS)))
    assert eps == RMS_NORM_EPS
    ref_scale = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    got = cache[f"{site}.hook_scale"]
    assert got.shape == ref_scale.shape
    assert _maxdiff(got, ref_scale) < TOL, f"{site}.hook_scale max|diff|={_maxdiff(got, ref_scale):.2e}"
    # Non-tautology: the default eps would give a visibly different scale.
    wrong = (x.pow(2).mean(-1, keepdim=True) + 1e-6).sqrt()
    assert _maxdiff(got, wrong) > 1e-3


@pytest.mark.parametrize("site", SITE_NAMES)
def test_hook_normalized_matches_hf_norm(captured, site):
    sites, hf_in, _, _, _, cache = captured
    x = hf_in[site].float()
    eps = float(getattr(sites[site], "eps", getattr(sites[site], "variance_epsilon", RMS_NORM_EPS)))
    ref_normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    got = cache[f"{site}.hook_normalized"]
    assert got.shape == ref_normed.shape
    assert _maxdiff(got, ref_normed) < TOL, (
        f"{site}.hook_normalized max|diff|={_maxdiff(got, ref_normed):.2e}"
    )


@pytest.mark.parametrize("site", SITE_NAMES)
def test_hook_out_matches_hf_module_output(captured, site):
    _, _, hf_out, _, _, cache = captured
    got = cache[f"{site}.hook_out"]
    assert got.shape == hf_out[site].shape
    assert _maxdiff(got, hf_out[site]) < TOL, (
        f"{site}.hook_out max|diff|={_maxdiff(got, hf_out[site]):.2e}"
    )


def test_norm_inputs_match_hf(captured):
    """The bridge fed each norm the same residual the HF reference did (catches upstream drift)."""
    _, hf_in, _, _, _, cache = captured
    for site in SITE_NAMES:
        got = cache[f"{site}.hook_in"]
        assert _maxdiff(got, hf_in[site]) < TOL, f"{site}.hook_in max|diff|={_maxdiff(got, hf_in[site]):.2e}"


def test_logits_parity(captured):
    _, _, _, hf_logits, bridge_logits, _ = captured
    assert bridge_logits.shape == hf_logits.shape
    assert torch.isfinite(bridge_logits).all()
    assert _maxdiff(bridge_logits, hf_logits) < 1e-4
