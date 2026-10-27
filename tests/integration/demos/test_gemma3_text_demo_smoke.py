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
