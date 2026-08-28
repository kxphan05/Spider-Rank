"""Where the agent looks for its local model weights.

Both the encoder and the masked LM used a bare `"./model"`, which resolves
against the *current working directory*. That is fine from the repo root and
silently wrong anywhere else: the weights are not found, the loader falls back
to a network fetch, and with no network the component goes dark without
raising (see `scripts/preflight.py`). A submitted bundle is exactly the case
where the harness may run from a different directory, so resolve explicitly.

Order: TECHJAM_MODEL_DIR, then ./model relative to the working directory,
then model/ beside the package. The working directory is checked before the
package-relative fallback so behaviour from the repo root is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL_DIR_ENV = "TECHJAM_MODEL_DIR"
_PACKAGE_PARENT = Path(__file__).resolve().parent.parent


def model_cache_dir() -> str:
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        return override
    cwd_relative = Path("model")
    if cwd_relative.is_dir():
        return str(cwd_relative)
    beside_package = _PACKAGE_PARENT / "model"
    if beside_package.is_dir():
        return str(beside_package)
    return str(cwd_relative)  # conventional path; loader decides how to fail
