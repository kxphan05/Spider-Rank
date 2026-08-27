"""Materialize every local runtime asset the agent needs, in one command.

`docs/submission_rules.md` § Reproducibility requires dependency installation
steps and one command to run the agent, and warns that the organizer may run
the submission under network restrictions. The agent therefore must reach
scoring time with all weights and indexes already on disk.

Two assets are not in git (they are far too large -- the bge-small weight blob
alone is 128 MB, over GitHub's 100 MB per-file hard limit):

  model/                  bge-small-en-v1.5 encoder weights, via HuggingFace
  data/dense_index/       50k catalog embeddings, built locally from the above

This script fetches the first and builds the second. **It needs network
access, and it is the only part of this project that does.** Run it once at
setup time; after it completes, `scripts/preflight.py --strict` should pass
and the agent runs fully offline.

Usage:
    uv run python3 scripts/fetch_assets.py              # encoder + dense index
    uv run python3 scripts/fetch_assets.py --with-lm    # also distilbert (see below)
    uv run python3 scripts/fetch_assets.py --skip-index

`--with-lm` is off by default deliberately. The masked-LM attribute inference
costs 256 MB of weights, ~350 MB of peak RSS and ~47 ms per turn, and is worth
+0.001 TechnicalScore -- inside the noise floor of the 200-sample public set.
See `docs/team_report.md` § Ablations.
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from _common import (DEFAULT_CATALOG, DEFAULT_INDEX_DIR, DEFAULT_MODEL_DIR,  # noqa: F401
                     REPO_ROOT)

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

MODEL_DIR = DEFAULT_MODEL_DIR
INDEX_DIR = DEFAULT_INDEX_DIR
CATALOG = DEFAULT_CATALOG

ENCODER = "BAAI/bge-small-en-v1.5"
MASKED_LM = "distilbert-base-uncased"


def fetch_encoder() -> None:
    print(f"[1/3] fetching {ENCODER} into {MODEL_DIR}/ ...")
    from sentence_transformers import SentenceTransformer

    # cache_folder must match starter/retrieval.py:load_embedding_model().
    SentenceTransformer(ENCODER, cache_folder=str(MODEL_DIR))
    print("      done.")


def fetch_masked_lm() -> None:
    print(f"[2/3] fetching {MASKED_LM} into {MODEL_DIR}/ ...")
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    AutoTokenizer.from_pretrained(MASKED_LM, cache_dir=str(MODEL_DIR))
    AutoModelForMaskedLM.from_pretrained(MASKED_LM, cache_dir=str(MODEL_DIR))
    print("      done.")


def build_index() -> int:
    print(f"[3/3] building the dense index into {INDEX_DIR}/ ...")
    if not CATALOG.exists():
        print(f"      SKIPPED: {CATALOG} is missing. Download the catalog first "
              "(see README.md 'Download the Catalog'), then re-run this script.",
              file=sys.stderr)
        return 1
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_dense_index.py")],
        cwd=REPO_ROOT,
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-lm", action="store_true",
                        help="also fetch distilbert for masked-LM attribute inference (+256 MB, ~noise)")
    parser.add_argument("--skip-index", action="store_true",
                        help="fetch weights only, don't build the dense index")
    args = parser.parse_args()

    fetch_encoder()
    if args.with_lm:
        fetch_masked_lm()
    else:
        print(f"[2/3] skipping {MASKED_LM} (pass --with-lm to include it)")

    code = 0 if args.skip_index else build_index()
    if args.skip_index:
        print("[3/3] skipping the dense index build (--skip-index)")

    print("\nNow verify with:  uv run python3 scripts/preflight.py --strict")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
