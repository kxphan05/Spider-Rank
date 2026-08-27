"""Shared bootstrap for the scripts in this directory.

Every script here needs the same three things, and each used to re-derive
them: the repo root on `sys.path` (so `starter` / `evaluator` import when the
script is run directly), the two dataset paths, and -- for anything that
constructs an `Agent` -- an isolated user-profile store.

Import this first; it puts the repo root on `sys.path` as an import side
effect, so the `starter` / `evaluator` imports that follow resolve:

    from _common import REPO_ROOT, DEFAULT_CATALOG, DEFAULT_DATASET  # noqa: F401
    from starter.agent import Agent  # noqa: E402

Paths are anchored to the repo root rather than the working directory, so
scripts work from anywhere instead of only from the repo root.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CATALOG = REPO_ROOT / "data" / "catalog.jsonl"
DEFAULT_DATASET = REPO_ROOT / "data" / "public_set.jsonl"
DEFAULT_MODEL_DIR = REPO_ROOT / "model"
DEFAULT_INDEX_DIR = REPO_ROOT / "data" / "dense_index"


def isolate_profile_store(path: str | None = None, *, announce: bool = True) -> str:
    """Point the long-term profile store at a fresh temp file for this run.

    A store carried across runs is a measured contamination source: history
    accumulated by an earlier run changes what later runs see (CLAUDE.md
    "Known open problems" #5). Diagnostics should start clean unless they are
    deliberately testing accumulation, so this is the default everywhere.

    Returns the path that was set, so callers can report or reuse it.
    """
    from starter.user_profile import STORE_PATH_ENV

    if path is None:
        path = str(Path(tempfile.mkdtemp(prefix="techjam-profiles-")) / "user_profiles.json")
        if announce:
            print(f"isolating profile store at {path}", file=sys.stderr)
    os.environ[STORE_PATH_ENV] = path
    return path
