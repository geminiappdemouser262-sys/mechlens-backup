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
