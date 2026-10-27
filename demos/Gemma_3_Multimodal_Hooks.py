"""Gemma-3 multimodal with TransformerBridge — minimal demo with a synthetic image.

Target: transformer-lens 4.3.0 (TransformerBridge only). Uses a generated image so it
runs anywhere with no downloads beyond the checkpoint itself.

    pip install "transformer-lens==4.3.0" pillow     # pulls transformers>=5.4
    export HF_TOKEN=...                               # Gemma is gated; accept the agreement on the Hub
    python demos/Gemma_3_Multimodal_Hooks.py [google/gemma-3-4b-it] [--device cuda]

Steps: load → synthetic image → prepare inputs → generate → cache + read vision hooks →
intervene on the vision tower. For a notebook version with real images see
demos/Gemma3_Multimodal.ipynb.
"""

import argparse

import torch
from PIL import Image, ImageDraw

from transformer_lens.model_bridge import TransformerBridge

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", default="google/gemma-3-4b-it")  # any Gemma3ForConditionalGeneration ckpt
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
torch.manual_seed(0)
dtype = torch.bfloat16 if args.device == "cuda" else torch.float32

# 1. Load ---------------------------------------------------------------------------------
model = TransformerBridge.boot_transformers(args.model, device=args.device, dtype=dtype)
model.eval()
assert getattr(model.cfg, "is_multimodal", False), f"{args.model} is not a multimodal checkpoint"
print(f"{args.model}: {model.cfg.n_layers} LM layers, vision tokens/image = {model.cfg.mm_tokens_per_image}")

# 2. Synthetic image ----------------------------------------------------------------------
# A red circle on a blue background — unambiguous enough that an -it model will describe it.
image = Image.new("RGB", (224, 224), (30, 60, 200))
ImageDraw.Draw(image).ellipse([56, 56, 168, 168], fill=(220, 40, 40))

# 3. Prepare inputs -----------------------------------------------------------------------
# The processor inserts <start_of_image> + mm_tokens_per_image image tokens for us.
prompt = "<start_of_turn>user\n<start_of_image>What shape and colours are in this image?<end_of_turn>\n<start_of_turn>model\n"
inputs = model.prepare_multimodal_inputs(text=prompt, images=image)
input_ids, pixel_values = inputs["input_ids"], inputs["pixel_values"]
extra = {k: v for k, v in inputs.items() if k not in ("input_ids", "pixel_values")}  # e.g. token_type_ids
print(f"input_ids {tuple(input_ids.shape)}  pixel_values {tuple(pixel_values.shape)}")

# 4. Generate -----------------------------------------------------------------------------
print(model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=24,
                     do_sample=False, return_type="str", verbose=False, **extra))

# 5. Cache + vision hooks -----------------------------------------------------------------
with torch.no_grad():
    logits, cache = model.run_with_cache(input_ids, pixel_values=pixel_values, **extra)

# Vision-side hook names (SigLIP tower lives under `vision_encoder`, the projector under
# `vision_projector`):
#   vision_encoder.hook_in                      raw pixel_values
#   vision_encoder.hook_vision_embed            patch + position embeddings (alias of embeddings.hook_out)
#   vision_encoder.encoder_layers.{i}.hook_resid_post   per SigLIP layer (aliases: hook_attn_out, hook_mlp_out)
#   vision_encoder.post_layernorm.hook_normalized / hook_scale / hook_out   final LayerNorm
#   vision_encoder.hook_vision_out              tower output (alias of hook_out)
#   vision_projector.hook_out                   pooled + projected tokens that enter the LM residual stream
vision_keys = sorted(k for k in cache if k.startswith(("vision_encoder", "vision_projector")))
print(f"{len(vision_keys)} vision cache entries, e.g. {vision_keys[:4]}")
for k in ("vision_encoder.hook_vision_embed", "vision_encoder.hook_vision_out", "vision_projector.hook_out"):
    if k in cache:
        print(f"  {k:<40} {tuple(cache[k].shape)}")

# Which LM positions hold the image? Compare the LM residual norm at image vs text tokens.
hf_cfg = model.original_model.config
image_token_id = getattr(hf_cfg, "image_token_id", None) or getattr(hf_cfg, "image_token_index")
is_img = (input_ids[0] == image_token_id).cpu()
resid = cache["blocks.0.hook_resid_pre"][0].float().norm(dim=-1).cpu()
print(f"resid_pre L0 norm — image tokens: {resid[is_img].mean():.1f}   text tokens: {resid[~is_img].mean():.1f}")

# 6. Intervene on the vision tower --------------------------------------------------------
# Zero the projected image tokens before they enter the LM: the model should no longer "see".
def blind(proj, hook):
    return torch.zeros_like(proj)

blind_logits = model.run_with_hooks(input_ids, pixel_values=pixel_values, **extra,
                                    fwd_hooks=[("vision_projector.hook_out", blind)])
print("top-1 next token, sighted:", repr(model.to_string(logits[0, -1].argmax())))
print("top-1 next token, blinded:", repr(model.to_string(blind_logits[0, -1].argmax())))
