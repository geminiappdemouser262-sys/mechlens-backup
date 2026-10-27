#!/usr/bin/env bash
# One-shot Gemma-3 verification: regression tests + docs registration/build check.
# Offline-safe: every test builds a tiny synthetic Gemma-3; no HF_TOKEN, no Hub download.
#
#   scripts/verify_gemma3.sh              # tests + docs check
#   scripts/verify_gemma3.sh --skip-docs  # tests only
#   scripts/verify_gemma3.sh --docs-only  # docs check only
#
# Used by .github/workflows/demo-smoke.yml; run it locally before opening a PR.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-uv run}"           # set RUN="" to use the current environment's python/pytest
SKIP_DOCS=0; DOCS_ONLY=0
for a in "$@"; do
  case "$a" in
    --skip-docs) SKIP_DOCS=1 ;;
    --docs-only) DOCS_ONLY=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

GEMMA3_TESTS=(
  tests/unit/model_bridge/supported_architectures/test_gemma3_adapter.py
  tests/unit/model_bridge/supported_architectures/test_gemma3_multimodal_adapter.py
  tests/unit/model_bridge/generalized_components/test_normalization_bridge.py
  tests/integration/model_bridge/test_gemma3_bridge.py
  tests/integration/model_bridge/test_gemma3_norm_hook_parity.py
  tests/integration/model_bridge/test_gemma3_vision_layernorm_offsets.py
  tests/integration/model_bridge/test_gemma3_final_rms_weight_processing.py
  tests/integration/demos/test_gemma3_text_demo_smoke.py
  tests/integration/demos/test_gemma3_multimodal_demo_smoke.py
)

if [[ $DOCS_ONLY -eq 0 ]]; then
  echo "== Gemma-3 regression tests (${#GEMMA3_TESTS[@]} files) =="
  $RUN pytest "${GEMMA3_TESTS[@]}" -v -p no:cacheprovider -W ignore::DeprecationWarning
fi

if [[ $SKIP_DOCS -eq 0 ]]; then
  echo "== Docs: demo registration =="
  $RUN python - <<'PY'
import re, sys
from pathlib import Path
problems = []
for nb in ("Gemma_3_Text.ipynb", "Gemma_3_Multimodal.ipynb"):
    stem = nb.removesuffix(".ipynb")
    if not Path("demos", nb).exists():
        problems.append(f"demos/{nb} missing"); continue
    mk = Path("docs/make_docs.py").read_text()
    if f'"{nb}"' not in mk.split("def copy_demos", 1)[1].split("def ", 1)[0]:
        problems.append(f"{nb} not in copy_demos() notebooks_to_copy (docs/make_docs.py)")
    if not re.search(rf"^generated/demos/{stem}\s*$", Path("docs/source/index.md").read_text(), re.M):
        problems.append(f"generated/demos/{stem} not in docs/source/index.md toctree")
    if nb not in Path("docs/source/content/tutorials.md").read_text():
        problems.append(f"{nb} not linked from docs/source/content/tutorials.md")
    import json
    for c in json.loads(Path("demos", nb).read_text())["cells"]:
        if c["cell_type"] == "code" and (c["outputs"] or c["execution_count"] is not None):
            problems.append(f"{nb} has unstripped outputs"); break
if problems:
    print("\n".join("  - " + p for p in problems)); sys.exit(1)
print("  ok: both notebooks registered in make_docs.py, index.md toctree, tutorials.md; outputs stripped")
PY

  echo "== Docs: sphinx build (nbsphinx_execute=never; model tables skipped) =="
  # build_docs() also regenerates the model-properties table from the Hub, which is slow and
  # needs network; here we only copy the demos, render the bridge-models page from the local
  # registry, and run sphinx. Warnings are allowed (same as `make build-docs`); errors are not.
  $RUN python - <<'PY'
import os, subprocess, sys
sys.path.insert(0, "docs")
import make_docs as m
m.copy_demos()
try:
    m.generate_bridge_models_page()
except Exception as e:  # local registry only; never fatal for the demo check
    print(f"  (bridge-models page skipped: {e})")
for nb in ("Gemma_3_Text.ipynb", "Gemma_3_Multimodal.ipynb"):
    out = m.GENERATED_DIR / "demos" / nb
    assert out.exists(), f"copy_demos() did not produce {out}"
subprocess.run([sys.executable, "-m", "sphinx", "-b", "html", "-q", str(m.SOURCE_PATH), str(m.BUILD_PATH)], check=True)
for stem in ("Gemma_3_Text", "Gemma_3_Multimodal"):
    html = m.BUILD_PATH / "generated" / "demos" / f"{stem}.html"
    assert html.exists(), f"sphinx did not render {html}"
    print(f"  ok: rendered {html.resolve()}")
PY
fi
echo "== verify_gemma3: PASS =="
