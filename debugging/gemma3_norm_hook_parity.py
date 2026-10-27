"""Gemma-3 RMSNorm hook parity check: hook_scale / hook_normalized vs. HuggingFace (Issue #1526).

Loads a Gemma-3 checkpoint through TransformerBridge and an independent HF reference,
then for every RMSNorm site (ln1, ln1_post, ln2, ln2_post per block, plus ln_final) checks:

  hook_scale      == sqrt(mean(x^2) + eps)          (denominator convention, float32)
  hook_normalized == x * rsqrt(mean(x^2) + eps)     (pre-weight, i.e. HF's _norm(x))
  hook_out        == HF module output               ((1 + w) * normalized, cast back)

where x is the exact float32 input the HF module received (captured with a forward pre-hook)
and eps is the HF module's own eps. Runs on CPU in float32 so the comparison is not
confounded by dtype. Default checkpoint is the 270m model so it fits anywhere.

Usage:
    python debugging/gemma3_norm_hook_parity.py                       # google/gemma-3-270m
    python debugging/gemma3_norm_hook_parity.py google/gemma-3-1b-it  # any Gemma-3 text ckpt
    python debugging/gemma3_norm_hook_parity.py google/gemma-3-270m --tol 1e-5

Exit code 0 = all sites within tolerance; 1 = at least one site diverged (details printed).
"""

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens.model_bridge.bridge import TransformerBridge

PROMPT = "The quick brown fox jumps over the lazy dog because the lazy dog would not move."


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="google/gemma-3-270m")
    ap.add_argument("--tol", type=float, default=1e-5, help="max-abs tolerance (float32)")
    args = ap.parse_args()

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    tokens = tok(PROMPT, return_tensors="pt").input_ids

    # Independent HF reference (not the module the bridge wraps), eager attention, float32.
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()
    bridge = TransformerBridge.boot_transformers(args.model, device="cpu", dtype=torch.float32)
    bridge.eval()

    # Map bridge hook prefixes -> HF RMSNorm modules.
    sites: dict[str, torch.nn.Module] = {}
    for i, layer in enumerate(hf.model.layers):
        sites[f"blocks.{i}.ln1"] = layer.input_layernorm
        sites[f"blocks.{i}.ln1_post"] = layer.post_attention_layernorm
        sites[f"blocks.{i}.ln2"] = layer.pre_feedforward_layernorm
        sites[f"blocks.{i}.ln2_post"] = layer.post_feedforward_layernorm
    sites["ln_final"] = hf.model.norm

    # Capture each HF norm's input and output.
    hf_in: dict[str, torch.Tensor] = {}
    hf_out: dict[str, torch.Tensor] = {}
    handles = []
    for name, mod in sites.items():
        handles.append(mod.register_forward_pre_hook(lambda m, inp, n=name: hf_in.__setitem__(n, inp[0].detach().clone())))
        handles.append(mod.register_forward_hook(lambda m, inp, out, n=name: hf_out.__setitem__(n, out.detach().clone())))

    with torch.no_grad():
        hf_logits = hf(tokens).logits
        bridge_logits, cache = bridge.run_with_cache(tokens)
    for h in handles:
        h.remove()

    bad = 0

    def check(label: str, got: torch.Tensor, want: torch.Tensor) -> None:
        nonlocal bad
        diff = (got.float() - want.float()).abs().max().item()
        ok = diff <= args.tol
        bad += 0 if ok else 1
        print(f"{'ok ' if ok else 'BAD'}  {label:<34} max|diff| = {diff:.3e}")

    for name, mod in sites.items():
        x = hf_in[name].float()
        eps = float(getattr(mod, "eps", getattr(mod, "variance_epsilon", 1e-6)))
        var = x.pow(2).mean(-1, keepdim=True)
        ref_scale = (var + eps).sqrt()
        ref_normed = x * torch.rsqrt(var + eps)
        check(f"{name}.hook_scale", cache[f"{name}.hook_scale"], ref_scale)
        check(f"{name}.hook_normalized", cache[f"{name}.hook_normalized"], ref_normed)
        check(f"{name}.hook_out vs HF output", cache[f"{name}.hook_out"], hf_out[name])

    check("logits vs HF", bridge_logits, hf_logits)

    print(f"\nbridge.cfg.eps = {bridge.cfg.eps}   HF rms_norm_eps = {hf.config.rms_norm_eps}")
    print("RESULT:", "PASS" if bad == 0 else f"FAIL ({bad} comparisons over tol={args.tol:g})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
