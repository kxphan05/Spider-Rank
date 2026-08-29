"""Where the agent looks for its local model weights, and how it fingerprints
the catalog those assets were derived from.

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

import hashlib
import os
from pathlib import Path
from .config import MODEL_DIR_ENV

_FINGERPRINT_CHUNK = 1 << 20
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


def catalog_fingerprint(catalog_path: Path) -> str:
    """Content hash of the catalog, for checking a dense index still matches it.

    This replaces a filesystem mtime comparison, which cannot survive leaving
    this machine. A prebuilt index shipped to the organizer would be checked
    against a catalog they had just checked out, so its mtime is newer than the
    index's by construction -- the guard fired on every correct setup while
    still not reliably catching a genuinely different catalog. Content is the
    only fingerprint that means the same thing on both machines.

    Read in chunks: the catalog is ~60 MB and there is no reason to hold it in
    memory. Cost is ~1 s on the 50k catalog, paid once at index load.
    """
    digest = hashlib.sha256()
    with catalog_path.open("rb") as handle:
        while chunk := handle.read(_FINGERPRINT_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
