#!/usr/bin/env bash
# Materialise the Gemma-3 demo notebooks and offline pytest files into a TransformerLens /
# mechlens working tree. Run from the repo root (or pass the root as $1).
#
#   bash materialize_gemma3_files.sh [/path/to/repo]
#
# Writes (overwriting if present):
#   demos/Gemma_3_Text.ipynb                       demos/Gemma_3_Multimodal.ipynb
#   tests/integration/demos/__init__.py
#   tests/integration/demos/test_gemma3_text_demo_smoke.py
#   tests/integration/demos/test_gemma3_multimodal_demo_smoke.py
#   tests/integration/model_bridge/test_gemma3_norm_hook_parity.py
#   tests/integration/model_bridge/test_gemma3_vision_layernorm_offsets.py
#   tests/integration/model_bridge/test_gemma3_final_rms_weight_processing.py
#
# All five tests run offline on CPU (tiny synthetic Gemma3 configs, mock processor); no HF_TOKEN.
# Afterwards:  pytest tests/integration/demos tests/integration/model_bridge/test_gemma3_*.py -v
#
# Generated from branch gemma3-regression-tests of geminiappdemouser262-sys/mechlens-backup.
set -euo pipefail
ROOT="${1:-.}"
[[ -d "$ROOT/transformer_lens" && -d "$ROOT/demos" ]] || { echo "not a TransformerLens repo root: $ROOT" >&2; exit 1; }
cd "$ROOT"
mkdir -p demos tests/integration/demos tests/integration/model_bridge

echo '  writing demos/Gemma_3_Text.ipynb'
cat > 'demos/Gemma_3_Text.ipynb' <<'__GEMMA3_FILE_0__'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gemma-3 (text-only) with TransformerBridge\n",
    "\n",
    "Minimal walkthrough for `transformer-lens` **4.3.0**, where `TransformerBridge` is the only load path (`HookedTransformer` was removed in 4.0). Replaces the unmerged Gemma-3 demo notebook (PR #1109).\n",
    "\n",
    "Four things, in order: **load → generate → cache + inspect → intervene**. Everything else is in the [migration guide](../docs/source/content/migrating_to_v3.md).\n",
    "\n",
    "```\n",
    "pip install \"transformer-lens==4.3.0\"   # pulls transformers>=5.4\n",
    "export HF_TOKEN=...                     # Gemma is gated; accept the model agreement on the Hub first\n",
    "```"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "\n",
    "from transformer_lens.model_bridge import TransformerBridge\n",
    "\n",
    "MODEL = \"google/gemma-3-270m\"  # also: -270m-it, -1b-pt, -1b-it, -4b-*, -12b-*\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "torch.manual_seed(0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load\n",
    "\n",
    "The bridge wraps the HF `Gemma3ForCausalLM` as-is: raw weights, no `fold_ln` / centering on load."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = TransformerBridge.boot_transformers(MODEL, device=DEVICE, dtype=torch.float32)\n",
    "model.eval()\n",
    "print(f\"{MODEL}: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, \"\n",
    "      f\"d_model={model.cfg.d_model}, eps={model.cfg.eps}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Generate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = \"The capital of France is\"\n",
    "print(model.generate(prompt, max_new_tokens=8, do_sample=False, verbose=False))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Cache + inspect\n",
    "\n",
    "Same hook names as `HookedTransformer`: `blocks.{i}.hook_resid_pre/post`, `attn.hook_pattern`, `attn.hook_z`, `mlp.hook_post`, `ln1.hook_normalized`, … plus Gemma-3's `ln1_post` / `ln2_post`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "tokens = model.to_tokens(prompt)\n",
    "logits, cache = model.run_with_cache(tokens)\n",
    "str_toks = model.to_str_tokens(prompt)\n",
    "\n",
    "pattern = cache[\"blocks.0.attn.hook_pattern\"][0]  # [head, query, key]\n",
    "print(\"layer-0 attention pattern:\", tuple(pattern.shape))\n",
    "print(\"resid_post L0 norm per token:\",\n",
    "      [f\"{v:.1f}\" for v in cache[\"blocks.0.hook_resid_post\"][0].norm(dim=-1).tolist()])\n",
    "\n",
    "top = logits[0, -1].topk(5)\n",
    "print(\"next-token top-5:\", [(model.to_string(i), f\"{p:.2f}\") for i, p in\n",
    "                             zip(top.indices.tolist(), top.values.softmax(-1).tolist())])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Intervene\n",
    "\n",
    "Zero-ablate every attention head's output in the last layer and watch the top-1 prediction change."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "last = model.cfg.n_layers - 1\n",
    "\n",
    "\n",
    "def ablate_heads(z, hook):  # z: [batch, pos, head, d_head]\n",
    "    return torch.zeros_like(z)\n",
    "\n",
    "\n",
    "ablated = model.run_with_hooks(tokens, fwd_hooks=[(f\"blocks.{last}.attn.hook_z\", ablate_heads)])\n",
    "print(\"top-1 clean   :\", repr(model.to_string(logits[0, -1].argmax())))\n",
    "print(\"top-1 ablated :\", repr(model.to_string(ablated[0, -1].argmax())))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optional: compatibility mode\n",
    "\n",
    "Analyses in `HookedTransformer`'s folded/centered coordinate system (logit lens, direct logit attribution, residual-stream norms) need the compatibility layer. Generation, loss and argmax are identical with or without it.\n",
    "\n",
    "```python\n",
    "model.enable_compatibility_mode()\n",
    "```"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "transformer-lens",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
__GEMMA3_FILE_0__

echo '  writing demos/Gemma_3_Multimodal.ipynb'
cat > 'demos/Gemma_3_Multimodal.ipynb' <<'__GEMMA3_FILE_1__'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Gemma-3 multimodal with TransformerBridge\n",
    "\n",
    "Minimal walkthrough for `transformer-lens` **4.3.0** (`TransformerBridge` only) using a **synthetic image**, so the only download is the checkpoint. Steps: **load → synthetic image → prepare inputs → generate → cache + vision hooks → intervene on the vision tower**.\n",
    "\n",
    "For a longer version with real photos and attention visualisation see `Gemma3_Multimodal.ipynb`.\n",
    "\n",
    "```\n",
    "pip install \"transformer-lens==4.3.0\" pillow   # pulls transformers>=5.4\n",
    "export HF_TOKEN=...                            # Gemma is gated; accept the agreement on the Hub first\n",
    "```"
   ],
   "id": "66b1aa5c"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "from PIL import Image, ImageDraw\n",
    "\n",
    "from transformer_lens.model_bridge import TransformerBridge\n",
    "\n",
    "MODEL = \"google/gemma-3-4b-it\"  # any Gemma3ForConditionalGeneration checkpoint (4b / 12b / 27b, -pt or -it)\n",
    "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
    "DTYPE = torch.bfloat16 if DEVICE == \"cuda\" else torch.float32\n",
    "torch.manual_seed(0)"
   ],
   "id": "4484624a"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Load"
   ],
   "id": "e2b1447c"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = TransformerBridge.boot_transformers(MODEL, device=DEVICE, dtype=DTYPE)\n",
    "model.eval()\n",
    "assert getattr(model.cfg, \"is_multimodal\", False), f\"{MODEL} is not a multimodal checkpoint\"\n",
    "print(f\"{MODEL}: {model.cfg.n_layers} LM layers, vision tokens/image = {model.cfg.mm_tokens_per_image}\")"
   ],
   "id": "c2e4c0b3"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Synthetic image\n",
    "\n",
    "A red circle on a blue background — unambiguous enough that an `-it` model will describe it."
   ],
   "id": "102e38f1"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "image = Image.new(\"RGB\", (224, 224), (30, 60, 200))\n",
    "ImageDraw.Draw(image).ellipse([56, 56, 168, 168], fill=(220, 40, 40))\n",
    "image"
   ],
   "id": "c7f98c59"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Prepare inputs\n",
    "\n",
    "The processor inserts `<start_of_image>` plus `mm_tokens_per_image` image soft tokens for us. Extra processor outputs (e.g. `token_type_ids`) are forwarded to every call."
   ],
   "id": "421f1a48"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "prompt = (\n",
    "    \"<start_of_turn>user\\n\"\n",
    "    \"<start_of_image>What shape and colours are in this image?<end_of_turn>\\n\"\n",
    "    \"<start_of_turn>model\\n\"\n",
    ")\n",
    "inputs = model.prepare_multimodal_inputs(text=prompt, images=image)\n",
    "input_ids, pixel_values = inputs[\"input_ids\"], inputs[\"pixel_values\"]\n",
    "extra = {k: v for k, v in inputs.items() if k not in (\"input_ids\", \"pixel_values\")}\n",
    "print(f\"input_ids {tuple(input_ids.shape)}  pixel_values {tuple(pixel_values.shape)}\")"
   ],
   "id": "4298d1d9"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Generate"
   ],
   "id": "41672735"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=24,\n",
    "                     do_sample=False, return_type=\"str\", verbose=False, **extra))"
   ],
   "id": "0d3394ef"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Cache + vision hooks\n",
    "\n",
    "Vision-side hook names (the SigLIP tower lives under `vision_encoder`, the projector under `vision_projector`):\n",
    "\n",
    "| hook | what it holds |\n",
    "|---|---|\n",
    "| `vision_encoder.hook_in` | raw `pixel_values` |\n",
    "| `vision_encoder.hook_vision_embed` | patch + position embeddings (alias of `embeddings.hook_out`) |\n",
    "| `vision_encoder.encoder_layers.{i}.hook_resid_post` | per SigLIP layer (aliases: `hook_attn_out`, `hook_mlp_out`) |\n",
    "| `vision_encoder.post_layernorm.hook_normalized` / `hook_scale` / `hook_out` | final LayerNorm |\n",
    "| `vision_encoder.hook_vision_out` | tower output (alias of `hook_out`) |\n",
    "| `vision_projector.hook_out` | pooled + projected tokens that enter the LM residual stream |"
   ],
   "id": "7e3aa765"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with torch.no_grad():\n",
    "    logits, cache = model.run_with_cache(input_ids, pixel_values=pixel_values, **extra)\n",
    "\n",
    "vision_keys = sorted(k for k in cache if k.startswith((\"vision_encoder\", \"vision_projector\")))\n",
    "print(f\"{len(vision_keys)} vision cache entries, e.g. {vision_keys[:4]}\")\n",
    "for k in (\"vision_encoder.hook_vision_embed\", \"vision_encoder.hook_vision_out\", \"vision_projector.hook_out\"):\n",
    "    if k in cache:\n",
    "        print(f\"  {k:<40} {tuple(cache[k].shape)}\")"
   ],
   "id": "e6c15539"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Which LM positions hold the image? Compare the layer-0 residual norm at image vs. text tokens."
   ],
   "id": "8faa0949"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "hf_cfg = model.original_model.config\n",
    "image_token_id = getattr(hf_cfg, \"image_token_id\", None) or getattr(hf_cfg, \"image_token_index\")\n",
    "is_img = (input_ids[0] == image_token_id).cpu()\n",
    "resid = cache[\"blocks.0.hook_resid_pre\"][0].float().norm(dim=-1).cpu()\n",
    "print(f\"resid_pre L0 norm — image tokens: {resid[is_img].mean():.1f}   text tokens: {resid[~is_img].mean():.1f}\")"
   ],
   "id": "e9bd1bb2"
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Intervene on the vision tower\n",
    "\n",
    "Zero the projected image tokens before they enter the LM — the model should no longer \"see\"."
   ],
   "id": "17180c88"
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def blind(proj, hook):\n",
    "    return torch.zeros_like(proj)\n",
    "\n",
    "\n",
    "blind_logits = model.run_with_hooks(\n",
    "    input_ids, pixel_values=pixel_values, **extra,\n",
    "    fwd_hooks=[(\"vision_projector.hook_out\", blind)],\n",
    ")\n",
    "print(\"top-1 next token, sighted:\", repr(model.to_string(logits[0, -1].argmax())))\n",
    "print(\"top-1 next token, blinded:\", repr(model.to_string(blind_logits[0, -1].argmax())))"
   ],
   "id": "fc82632d"
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "transformer-lens",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
__GEMMA3_FILE_1__

echo '  writing tests/integration/demos/__init__.py'
: > tests/integration/demos/__init__.py

echo '  writing tests/integration/demos/test_gemma3_text_demo_smoke.py'
cat > 'tests/integration/demos/test_gemma3_text_demo_smoke.py' <<'__GEMMA3_FILE_3__'
"""Smoke test for demos/Gemma_3_Text.ipynb — runs offline, no HF_TOKEN, no Hub download.

Executes every code cell of the notebook, in order, in one namespace, with
``TransformerBridge.boot_transformers`` redirected to a tiny synthetic
``Gemma3ForCausalLM`` (built from ``Gemma3TextConfig``) and a tiny locally-trained
tokenizer. Because the real ``boot_transformers`` still runs (it just receives
``hf_model=`` / ``tokenizer=``), this exercises the same load path the demo uses
against the Hub, minus the network.

Checks that the notebook runs end to end, that each section printed what it
promises, and that the bridge picked up the synthetic config's ``rms_norm_eps``.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import Gemma3ForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

from transformer_lens.model_bridge.bridge import TransformerBridge

NOTEBOOK = Path(__file__).resolve().parents[3] / "demos" / "Gemma_3_Text.ipynb"
RMS_NORM_EPS = 1e-6  # Gemma-3's real value; the test asserts it propagates


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Word-level tokenizer trained on a few sentences; has BOS/EOS/PAD like Gemma's."""
    corpus = [
        "The capital of France is Paris .",
        "The quick brown fox jumps over the lazy dog .",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z",
    ]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tok.train_from_iterator(corpus, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
        add_bos_token=True,
    )


def _tiny_gemma3(vocab_size: int) -> Gemma3ForCausalLM:
    """Deterministic 2-layer Gemma-3 (local + global RoPE, QK-norm, dual norms), float32, eager."""
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=vocab_size,
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
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return Gemma3ForCausalLM(cfg).to(torch.float32).eval()


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def notebook_run(tmp_path_factory):
    """Run the notebook once; return (namespace, stdout)."""
    tokenizer = _tiny_tokenizer()
    hf_model = _tiny_gemma3(vocab_size=len(tokenizer))
    original_boot = TransformerBridge.boot_transformers

    def boot_offline(model_name, *args, **kwargs):
        # Same code path as the demo, with the Hub model/tokenizer swapped for local ones.
        kwargs.pop("hf_model", None)
        kwargs.pop("tokenizer", None)
        kwargs["device"] = "cpu"
        kwargs["dtype"] = torch.float32
        return original_boot(
            model_name,
            hf_model=copy.deepcopy(hf_model),
            tokenizer=tokenizer,
            **kwargs,
        )

    cells = _code_cells(NOTEBOOK)
    assert len(cells) >= 4, "expected load / generate / cache / intervene cells"

    namespace: dict = {"__name__": "__notebook__"}
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TransformerBridge, "boot_transformers", staticmethod(boot_offline))
        with contextlib.redirect_stdout(buf):
            for i, src in enumerate(cells):
                try:
                    exec(compile(src, f"{NOTEBOOK.name}[code cell {i}]", "exec"), namespace)
                except Exception as e:  # noqa: BLE001 — surface the cell that failed
                    raise AssertionError(f"code cell {i} failed: {e}\n--- cell ---\n{src}") from e
    return namespace, buf.getvalue()


def test_notebook_exists_and_has_no_outputs():
    nb = json.loads(NOTEBOOK.read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, "commit the notebook with outputs stripped"


def test_notebook_runs_end_to_end(notebook_run):
    ns, out = notebook_run
    assert isinstance(ns["model"], TransformerBridge)
    # Section 1: load
    assert ns["model"].cfg.n_layers == 2
    assert ns["model"].cfg.eps == RMS_NORM_EPS, "rms_norm_eps did not propagate to bridge.cfg.eps"
    assert "layers" in out and "eps=" in out
    # Section 3: cache + inspect
    assert tuple(ns["pattern"].shape)[0] == 4, "layer-0 pattern should have one slice per head"
    assert ns["logits"].shape[-1] == len(ns["model"].tokenizer)
    assert torch.isfinite(ns["logits"]).all()
    assert "next-token top-5" in out
    # Section 4: intervene — hook actually ran and produced finite logits of the same shape
    assert ns["ablated"].shape == ns["logits"].shape
    assert torch.isfinite(ns["ablated"]).all()
    assert "top-1 clean" in out and "top-1 ablated" in out


def test_notebook_uses_only_bridge_api():
    """Guard against the demo regressing to the removed HookedTransformer path."""
    src = "\n".join(_code_cells(NOTEBOOK))
    assert "boot_transformers(" in src
    assert "HookedTransformer" not in src
__GEMMA3_FILE_3__

echo '  writing tests/integration/demos/test_gemma3_multimodal_demo_smoke.py'
cat > 'tests/integration/demos/test_gemma3_multimodal_demo_smoke.py' <<'__GEMMA3_FILE_4__'
"""Smoke test for demos/Gemma_3_Multimodal.ipynb — offline, CPU, no HF_TOKEN, no Hub download.

Executes every code cell of the notebook in order with:
  * ``TransformerBridge.boot_transformers`` redirected to a tiny synthetic
    ``Gemma3ForConditionalGeneration`` built from ``Gemma3Config`` (2-layer text + 2-layer SigLIP,
    16x16 images, 4 image tokens) and a locally-trained tokenizer that knows Gemma's chat and
    image special tokens;
  * ``transformers.AutoProcessor.from_pretrained`` replaced by a mock processor that does what
    the real Gemma3Processor does for the demo: expands ``<start_of_image>`` into
    boi + ``mm_tokens_per_image`` image soft tokens + eoi, resizes the PIL image to the model's
    ``image_size`` and returns ``input_ids`` / ``pixel_values`` / ``attention_mask`` /
    ``token_type_ids``.

The real ``boot()`` path still runs (it just receives ``hf_model=`` / ``tokenizer=`` and the
mocked processor), so ``prepare_multimodal_inputs``, ``generate(pixel_values=...)``,
``run_with_cache(pixel_values=...)`` and the vision hooks are exercised for real.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import Gemma3Config, Gemma3ForConditionalGeneration, PreTrainedTokenizerFast

from transformer_lens.model_bridge.bridge import TransformerBridge

NOTEBOOK = Path(__file__).resolve().parents[3] / "demos" / "Gemma_3_Multimodal.ipynb"
IMAGE_SIZE, PATCH_SIZE = 16, 8
MM_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<start_of_turn>", "<end_of_turn>",
           "<start_of_image>", "<end_of_image>", "<image_soft_token>"]


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    corpus = [
        "user model What shape and colours are in this image ? red circle blue background",
        "a b c d e f g h i j k l m n o p q r s t u v w x y z . , \\n",
    ]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, trainers.WordLevelTrainer(special_tokens=SPECIAL))
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>",
        unk_token="<unk>", add_bos_token=True,
        additional_special_tokens=SPECIAL[4:],
    )


def _tiny_model(tokenizer: PreTrainedTokenizerFast) -> Gemma3ForConditionalGeneration:
    ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
    torch.manual_seed(0)
    cfg = Gemma3Config(
        text_config=dict(
            vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
            rms_norm_eps=1e-6, sliding_window=16, layer_types=["sliding_attention", "full_attention"],
            attn_implementation="eager", bos_token_id=ids["<bos>"], eos_token_id=ids["<eos>"],
            pad_token_id=ids["<pad>"],
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            num_channels=3, image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, layer_norm_eps=1e-6,
            attn_implementation="eager",
        ),
        mm_tokens_per_image=MM_TOKENS,
        # transformers 5.x names, plus the 4.x names (stored harmlessly) for safety
        image_token_id=ids["<image_soft_token>"], boi_token_id=ids["<start_of_image>"],
        eoi_token_id=ids["<end_of_image>"],
        image_token_index=ids["<image_soft_token>"], boi_token_index=ids["<start_of_image>"],
        eoi_token_index=ids["<end_of_image>"],
        bos_token_id=ids["<bos>"], eos_token_id=ids["<eos>"], pad_token_id=ids["<pad>"],
    )
    # from-config models have no `architectures`; boot() needs it to pick the multimodal adapter
    cfg.architectures = ["Gemma3ForConditionalGeneration"]
    return Gemma3ForConditionalGeneration(cfg).to(torch.float32).eval()


class MockGemma3Processor:
    """Minimal stand-in for Gemma3Processor: text with <start_of_image> + a PIL image -> tensors."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer
        self.boi = tokenizer.convert_tokens_to_ids("<start_of_image>")
        self.eoi = tokenizer.convert_tokens_to_ids("<end_of_image>")
        self.img = tokenizer.convert_tokens_to_ids("<image_soft_token>")
        self.calls = 0

    def __call__(self, text, images=None, return_tensors="pt", **_):
        self.calls += 1
        assert isinstance(text, str), "demo passes a single prompt"
        images = [] if images is None else (images if isinstance(images, list) else [images])
        ids: list[int] = [self.tokenizer.bos_token_id]
        types: list[int] = [0]
        n_images = 0
        for chunk in text.split("<start_of_image>"):
            if n_images:  # every chunk after the first follows an image placeholder
                ids += [self.boi] + [self.img] * MM_TOKENS + [self.eoi]
                types += [0] + [1] * MM_TOKENS + [0]
            n_images += 1
            toks = self.tokenizer(chunk, add_special_tokens=False)["input_ids"]
            ids += toks
            types += [0] * len(toks)
        assert n_images - 1 == len(images), "number of <start_of_image> must match images"
        out = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
            "token_type_ids": torch.tensor([types], dtype=torch.long),
        }
        if images:
            arrs = []
            for im in images:
                im = im.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
                a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0  # SigLIP-style [-1, 1]
                arrs.append(torch.from_numpy(a).permute(2, 0, 1))
            out["pixel_values"] = torch.stack(arrs)
        return out


def _code_cells(path: Path) -> list[str]:
    nb = json.loads(path.read_text())
    assert nb["nbformat"] == 4 and nb["nbformat_minor"] >= 5
    assert all("id" in c for c in nb["cells"]), "nbformat 4.5 cells need ids"
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def notebook_run():
    tokenizer = _tiny_tokenizer()
    hf_model = _tiny_model(tokenizer)
    processor = MockGemma3Processor(tokenizer)
    original_boot = TransformerBridge.boot_transformers

    def boot_offline(model_name, *args, **kwargs):
        kwargs.pop("hf_model", None)
        kwargs.pop("tokenizer", None)
        kwargs["device"], kwargs["dtype"] = "cpu", torch.float32
        return original_boot(model_name, hf_model=copy.deepcopy(hf_model), tokenizer=tokenizer, **kwargs)

    cells = _code_cells(NOTEBOOK)
    namespace: dict = {"__name__": "__notebook__"}
    buf = io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TransformerBridge, "boot_transformers", staticmethod(boot_offline))
        # boot() does `from transformers import AutoProcessor; AutoProcessor.from_pretrained(model_name)`
        mp.setattr(transformers.AutoProcessor, "from_pretrained", classmethod(lambda cls, *a, **k: processor))
        mp.setenv("HF_HUB_OFFLINE", "1")
        mp.setenv("TRANSFORMERS_OFFLINE", "1")
        with contextlib.redirect_stdout(buf):
            for i, src in enumerate(cells):
                try:
                    exec(compile(src, f"{NOTEBOOK.name}[code cell {i}]", "exec"), namespace)
                except Exception as e:  # noqa: BLE001
                    raise AssertionError(f"code cell {i} failed: {e}\n--- cell ---\n{src}") from e
    return namespace, buf.getvalue(), processor


def test_notebook_is_clean_nbformat_45():
    nb = json.loads(NOTEBOOK.read_text())
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None, "commit with outputs stripped"


def test_notebook_runs_end_to_end(notebook_run):
    ns, out, processor = notebook_run
    model = ns["model"]
    assert isinstance(model, TransformerBridge) and model.cfg.is_multimodal
    assert model.processor is processor and processor.calls >= 1
    # Section 3: inputs
    assert ns["pixel_values"].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert int((ns["input_ids"][0] == processor.img).sum()) == MM_TOKENS
    # Section 5: cache has vision entries and LM logits over the whole sequence
    assert ns["logits"].shape[:2] == ns["input_ids"].shape
    assert torch.isfinite(ns["logits"]).all()
    assert any(k.startswith("vision_encoder") for k in ns["cache"])
    assert "vision_projector.hook_out" in ns["cache"]
    assert ns["cache"]["vision_projector.hook_out"].shape[-2] == MM_TOKENS
    assert ns["is_img"].sum() == MM_TOKENS
    # Section 6: blinding hook actually changed the forward pass at image-conditioned positions
    assert ns["blind_logits"].shape == ns["logits"].shape
    assert torch.isfinite(ns["blind_logits"]).all()
    assert not torch.allclose(ns["blind_logits"], ns["logits"]), "zeroing vision_projector.hook_out had no effect"
    for marker in ("vision cache entries", "resid_pre L0 norm", "sighted:", "blinded:"):
        assert marker in out, f"missing expected print: {marker!r}"


def test_vision_projector_output_feeds_image_positions(notebook_run):
    """The projected image tokens are what sits at the image positions of the LM input (Gemma-3 scatter)."""
    ns, _, processor = notebook_run
    model = ns["model"]
    proj = ns["cache"]["vision_projector.hook_out"][0].float()  # [MM_TOKENS, d_model]
    resid_pre = ns["cache"]["blocks.0.hook_resid_pre"][0].float()
    img_pos = (ns["input_ids"][0] == processor.img).nonzero().flatten()
    assert len(img_pos) == MM_TOKENS
    assert torch.allclose(resid_pre[img_pos], proj, atol=1e-5, rtol=1e-5)


def test_notebook_uses_only_bridge_api():
    src = "\n".join(_code_cells(NOTEBOOK))
    assert "boot_transformers(" in src and "prepare_multimodal_inputs(" in src
    assert "HookedTransformer" not in src and "requests." not in src, "demo must stay offline-capable"
__GEMMA3_FILE_4__

echo '  writing tests/integration/model_bridge/test_gemma3_norm_hook_parity.py'
cat > 'tests/integration/model_bridge/test_gemma3_norm_hook_parity.py' <<'__GEMMA3_FILE_5__'
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
__GEMMA3_FILE_5__

echo '  writing tests/integration/model_bridge/test_gemma3_vision_layernorm_offsets.py'
cat > 'tests/integration/model_bridge/test_gemma3_vision_layernorm_offsets.py' <<'__GEMMA3_FILE_6__'
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
__GEMMA3_FILE_6__

echo '  writing tests/integration/model_bridge/test_gemma3_final_rms_weight_processing.py'
cat > 'tests/integration/model_bridge/test_gemma3_final_rms_weight_processing.py' <<'__GEMMA3_FILE_7__'
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
__GEMMA3_FILE_7__

python3 - <<'PY'
import json, py_compile, sys
ok = True
for nb in ("demos/Gemma_3_Text.ipynb", "demos/Gemma_3_Multimodal.ipynb"):
    d = json.load(open(nb)); assert d["nbformat"] == 4, nb
for t in (
    "tests/integration/demos/test_gemma3_text_demo_smoke.py",
    "tests/integration/demos/test_gemma3_multimodal_demo_smoke.py",
    "tests/integration/model_bridge/test_gemma3_norm_hook_parity.py",
    "tests/integration/model_bridge/test_gemma3_vision_layernorm_offsets.py",
    "tests/integration/model_bridge/test_gemma3_final_rms_weight_processing.py",
):
    py_compile.compile(t, doraise=True)
print("all files written and syntax-checked")
PY
echo "next: pytest tests/integration/demos tests/integration/model_bridge/test_gemma3_*.py -v"
