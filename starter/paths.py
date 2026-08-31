"""Where the agent looks for its local model weights, and how it fingerprints
the catalog those assets were derived from.

A bare "./model" resolves against cwd, which breaks silently if the harness
runs from elsewhere. Order: TECHJAM_MODEL_DIR, then ./model, then model/
beside the package.
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

    Replaces an mtime comparison, which false-fires on a freshly checked-out
    catalog. Read in chunks since the file is ~60 MB.
    """
    digest = hashlib.sha256()
    with catalog_path.open("rb") as handle:
        while chunk := handle.read(_FINGERPRINT_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
