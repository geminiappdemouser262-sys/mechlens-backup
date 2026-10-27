"""Gemma-3: weight processing must preserve the model's predictions through the final RMSNorm fold.

Guards the behaviour fixed in upstream PR #1903. Gemma-3 is the awkward case for
``fold_ln`` on ``ln_final``:

  * RMSNorm uses the ``(1 + w)`` convention, so the fold must scale the unembedding by
    ``(1 + w)`` and set ``ln_final.weight`` to the identity under that convention — zeros,
    not ones;
  * ``lm_head.weight`` is tied to ``embed_tokens.weight``, so folding into the unembedding
    must not also scale the input embedding (that silently corrupts every residual stream).

The invariant: for any processing combination, next-token log-probabilities (hence argmax,
loss, generation) are identical to the unprocessed HF model, and the parts of the network
that should be untouched are bit-identical. Offline, CPU, tiny synthetic Gemma3TextConfig;
no Hub download, no HF_TOKEN.
"""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from transformer_lens.model_bridge.bridge import TransformerBridge
from transformer_lens.model_bridge.sources._bridge_builder import build_bridge_from_module

TOL = 1e-5
LOGPROB_TOL = 1e-4


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
        rms_norm_eps=1e-6,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"],
        attn_implementation="eager",
        tie_word_embeddings=True,  # Gemma-3's real setting; the case the fold must respect
    )


@pytest.fixture(scope="module")
def hf_ref() -> Gemma3ForCausalLM:
    torch.manual_seed(3)
    model = Gemma3ForCausalLM(_tiny_cfg()).to(torch.float32).eval()
    # Give ln_final a weight that is clearly not the identity under either convention, so a
    # fold that uses `w` instead of `(1 + w)`, or resets to ones instead of zeros, is visible.
    with torch.no_grad():
        model.model.norm.weight.copy_(torch.linspace(-0.5, 1.5, model.config.hidden_size))
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), (
        "test setup expects tied embed/unembed weights"
    )
    return model


@pytest.fixture(scope="module")
def tokens() -> torch.Tensor:
    return torch.tensor([[2, 5, 12, 23, 34, 45, 56, 67, 78, 89, 100, 111]], dtype=torch.long)


def _fresh_bridge(hf_ref: Gemma3ForCausalLM) -> TransformerBridge:
    hf_for_bridge = copy.deepcopy(hf_ref).eval()
    hf_for_bridge.load_state_dict(copy.deepcopy(hf_ref.state_dict()), strict=True)
    bridge = build_bridge_from_module(
        hf_for_bridge,
        architecture="Gemma3ForCausalLM",
        hf_config=copy.deepcopy(hf_ref.config),
        tokenizer=None,
        device="cpu",
        dtype=torch.float32,
        model_name="google/gemma-3-270m",
    ).eval()
    bridge.adapter.setup_component_testing(hf_for_bridge, bridge_model=bridge)
    return bridge


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


PROCESSING_MODES = {
    "fold_ln_only": dict(fold_ln=True, center_writing_weights=False, center_unembed=False, fold_value_biases=False),
    "fold_ln+center_unembed": dict(fold_ln=True, center_writing_weights=False, center_unembed=True, fold_value_biases=False),
    "defaults": {},  # fold_ln + center_writing_weights + center_unembed + fold_value_biases
}


@pytest.fixture(scope="module")
def reference(hf_ref, tokens):
    with torch.no_grad():
        logits = hf_ref(tokens.clone()).logits
    return {
        "logits": logits,
        "logprobs": torch.log_softmax(logits.float(), dim=-1),
        "embed": hf_ref.model.embed_tokens.weight.detach().clone(),
        "lm_head": hf_ref.lm_head.weight.detach().clone(),
        "ln_final_w": hf_ref.model.norm.weight.detach().clone(),
    }


def test_unprocessed_bridge_matches_hf(hf_ref, tokens, reference):
    bridge = _fresh_bridge(hf_ref)
    with torch.no_grad():
        logits = bridge(tokens.clone(), return_type="logits")
    assert _maxdiff(logits, reference["logits"]) < LOGPROB_TOL


@pytest.mark.parametrize("mode", list(PROCESSING_MODES))
def test_predictions_preserved_through_processing(hf_ref, tokens, reference, mode):
    bridge = _fresh_bridge(hf_ref)
    bridge.enable_compatibility_mode(disable_warnings=True, **PROCESSING_MODES[mode])

    with torch.no_grad():
        logits = bridge(tokens.clone(), return_type="logits")
    assert torch.isfinite(logits).all()
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    # Centering the unembed only shifts each row by a constant; log-probs must be invariant.
    assert _maxdiff(logprobs, reference["logprobs"]) < LOGPROB_TOL, (
        f"[{mode}] next-token distribution changed: max|dlogp|={_maxdiff(logprobs, reference['logprobs']):.2e}"
    )
    assert torch.equal(logprobs.argmax(-1), reference["logprobs"].argmax(-1))
    if not PROCESSING_MODES[mode].get("center_unembed", True):
        # Without centering, raw logits must be reproduced exactly (fold is a pure reparametrisation).
        assert _maxdiff(logits, reference["logits"]) < LOGPROB_TOL


@pytest.mark.parametrize("mode", list(PROCESSING_MODES))
def test_ln_final_fold_uses_offset_convention(hf_ref, mode):
    """After fold_ln, ln_final.weight is the (1+w)-identity (zeros) and W_U absorbed (1+w)."""
    bridge = _fresh_bridge(hf_ref)
    w_before = bridge.ln_final.weight.detach().clone()
    wu_before = bridge.unembed.weight.detach().clone()  # HF layout: [vocab, d_model]
    bridge.enable_compatibility_mode(disable_warnings=True, **PROCESSING_MODES[mode])

    w_after = bridge.ln_final.weight.detach()
    assert torch.equal(w_after, torch.zeros_like(w_after)), (
        f"[{mode}] ln_final.weight should be all-zeros (identity under (1+w)), got "
        f"min={w_after.min():.3f} max={w_after.max():.3f}"
    )
    # Effective scale folded into W_U must be (1 + w), not w. Check before any centering by
    # comparing row-wise ratios on the un-centred mode only.
    if not PROCESSING_MODES[mode].get("center_unembed", True) and not PROCESSING_MODES[mode].get(
        "center_writing_weights", True
    ):
        expected = wu_before * (1.0 + w_before)[None, :]
        assert _maxdiff(bridge.unembed.weight.detach(), expected) < TOL, (
            f"[{mode}] W_U was not scaled by (1 + ln_final.w)"
        )


@pytest.mark.parametrize("mode", list(PROCESSING_MODES))
def test_tied_embedding_is_not_scaled_by_the_fold(hf_ref, reference, mode):
    """Folding ln_final into lm_head must not leak into the tied embed_tokens."""
    bridge = _fresh_bridge(hf_ref)
    bridge.enable_compatibility_mode(disable_warnings=True, **PROCESSING_MODES[mode])

    embed_after = bridge.original_model.model.embed_tokens.weight.detach()
    assert torch.equal(embed_after, reference["embed"]), (
        f"[{mode}] embed_tokens.weight changed during processing — the ln_final fold leaked "
        "through the embed/unembed weight tie"
    )
    # And the tie really was broken on the unembed side (otherwise the above could only hold
    # if the fold did nothing).
    lm_head_after = bridge.original_model.lm_head.weight.detach()
    assert lm_head_after.data_ptr() != embed_after.data_ptr()
    assert not torch.equal(lm_head_after, reference["lm_head"]), "lm_head was not folded at all"


def test_ln_final_hook_normalized_unchanged_by_processing(hf_ref, tokens):
    """Processing re-parametrises weights; the normalised final residual must be identical."""
    clean = _fresh_bridge(hf_ref)
    processed = _fresh_bridge(hf_ref)
    processed.enable_compatibility_mode(disable_warnings=True)
    with torch.no_grad():
        _, c0 = clean.run_with_cache(tokens.clone())
        _, c1 = processed.run_with_cache(tokens.clone())
    assert _maxdiff(c1["ln_final.hook_normalized"], c0["ln_final.hook_normalized"]) < TOL
    assert _maxdiff(c1["ln_final.hook_scale"], c0["ln_final.hook_scale"]) < TOL
