"""Gemma-3 (text-only) with TransformerBridge — minimal demo.

Target: transformer-lens 4.3.0 (TransformerBridge only; HookedTransformer was removed in 4.0).
Replaces the unmerged Gemma-3 demo notebook (PR #1109).

    pip install "transformer-lens==4.3.0"      # pulls transformers>=5.4
    export HF_TOKEN=...                        # Gemma is gated; accept the model agreement on the Hub
    python demos/Gemma3_Text_Demo.py [google/gemma-3-270m] [--device cuda]

Four things, in order: load, generate, cache + inspect, intervene. Everything else is
in the migration guide (docs/source/content/migrating_to_v3.md).
"""

import argparse

import torch

from transformer_lens.model_bridge import TransformerBridge

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="google/gemma-3-270m")  # also: -270m-it, -1b-pt, -1b-it, -4b-*, -12b-*
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
torch.manual_seed(0)

# 1. Load ---------------------------------------------------------------------------------
# The bridge wraps the HF Gemma3ForCausalLM as-is: raw weights, no fold_ln / centering on load.
model = TransformerBridge.boot_transformers(args.model, device=args.device, dtype=torch.float32)
model.eval()
print(f"{args.model}: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
      f"d_model={model.cfg.d_model}, eps={model.cfg.eps}")

# 2. Generate -----------------------------------------------------------------------------
prompt = "The capital of France is"
print(model.generate(prompt, max_new_tokens=8, do_sample=False, verbose=False))

# 3. Cache + inspect ----------------------------------------------------------------------
# Same hook names as HookedTransformer: blocks.{i}.hook_resid_pre/post, attn.hook_pattern,
# attn.hook_z, mlp.hook_post, ln1.hook_normalized, ... plus Gemma-3's ln1_post / ln2_post.
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)
str_toks = model.to_str_tokens(prompt)

pattern = cache["blocks.0.attn.hook_pattern"][0]          # [head, query, key]
print("layer-0 attention pattern:", tuple(pattern.shape))
print("resid_post L0 norm per token:",
      [f"{v:.1f}" for v in cache["blocks.0.hook_resid_post"][0].norm(dim=-1).tolist()])

top = logits[0, -1].topk(5)
print("next-token top-5:", [(model.to_string(i), f"{p:.2f}") for i, p in
                             zip(top.indices.tolist(), top.values.softmax(-1).tolist())])

# 4. Intervene ----------------------------------------------------------------------------
# Zero-ablate every attention head's output in the last layer and watch the top-1 change.
last = model.cfg.n_layers - 1

def ablate_heads(z, hook):           # z: [batch, pos, head, d_head]
    return torch.zeros_like(z)

ablated = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{last}.attn.hook_z", ablate_heads)])
print("top-1 clean   :", repr(model.to_string(logits[0, -1].argmax())))
print("top-1 ablated :", repr(model.to_string(ablated[0, -1].argmax())))

# Optional: analyses in HookedTransformer's folded/centered coordinate system (logit lens,
# direct logit attribution, resid norms) need the compatibility layer. Generation, loss and
# argmax are identical with or without it.
#   model.enable_compatibility_mode()
