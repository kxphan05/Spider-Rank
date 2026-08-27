"""Verify every runtime asset is present and the pipeline comes up whole.

The failure this exists to catch is silent. With a cold HuggingFace cache and
no network, `Agent.__init__` degrades by design rather than crashing
(`starter/agent.py`: "degrade-don't-crash"): the encoder fails to load, and
with it the dense index, both embedding classifiers and the masked LM all go
dark. The agent still starts, still answers, and still returns ten
recommendations -- it just silently stops being the system that was measured.
`docs/submission_rules.md` warns the organizer may disable network access for
official scoring, which is exactly the condition that triggers this.

So: run this before any scored run. It reports which components are actually
live, and with --strict exits non-zero if the pipeline is not whole.

Usage:
    uv run python3 scripts/preflight.py
    uv run python3 scripts/preflight.py --strict     # non-zero exit on any gap
    uv run python3 scripts/preflight.py --require-lm # treat the masked LM as required
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CATALOG = REPO_ROOT / "data" / "catalog.jsonl"
INDEX_DIR = REPO_ROOT / "data" / "dense_index"
MODEL_DIR = REPO_ROOT / "model"
ENCODER_DIR = MODEL_DIR / "models--BAAI--bge-small-en-v1.5"
LM_DIR = MODEL_DIR / "models--distilbert-base-uncased"

OK, WARN, FAIL = "  ok  ", " warn ", " FAIL "


def line(status: str, name: str, detail: str) -> None:
    print(f"[{status}] {name:<22} {detail}")


def check_static() -> list[str]:
    """File-level checks that need no imports and no network."""
    problems: list[str] = []

    if CATALOG.exists():
        size_mb = CATALOG.stat().st_size / (1024 * 1024)
        line(OK, "catalog", f"{CATALOG.relative_to(REPO_ROOT)} ({size_mb:.0f} MB)")
    else:
        line(FAIL, "catalog", f"missing {CATALOG.relative_to(REPO_ROOT)} -- see README.md 'Download the Catalog'")
        problems.append("catalog")

    if ENCODER_DIR.is_dir():
        line(OK, "encoder weights", f"{ENCODER_DIR.relative_to(REPO_ROOT)}")
    else:
        line(FAIL, "encoder weights", "missing bge-small -- run scripts/fetch_assets.py")
        problems.append("encoder")

    meta_path = INDEX_DIR / "meta.json"
    if (INDEX_DIR / "embeddings.npy").exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            line(OK, "dense index", f"{meta.get('count', '?')} rows, model={meta.get('model_name')}")
        except (OSError, json.JSONDecodeError) as exc:
            line(FAIL, "dense index", f"meta.json unreadable: {exc}")
            problems.append("dense index")
    else:
        line(FAIL, "dense index", "missing -- run scripts/build_dense_index.py")
        problems.append("dense index")

    if LM_DIR.is_dir():
        line(OK, "masked LM", "distilbert present (optional; ~noise, see team_report.md)")
    else:
        line(WARN, "masked LM", "absent -- attribute inference disabled (optional, worth ~+0.001)")

    return problems


def check_live(require_lm: bool) -> list[str]:
    """Instantiate the real Agent offline and report what actually came up."""
    problems: list[str] = []
    # Force the same condition the grader may impose, so a warm cache cannot
    # mask a missing local asset by quietly downloading it.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("\n-- instantiating the agent with network access disabled --")
    try:
        from starter.agent import Agent
        agent = Agent()
    except Exception as exc:  # pragma: no cover - a hard init failure is itself the finding
        line(FAIL, "agent init", f"raised {type(exc).__name__}: {exc}")
        return ["agent init"]

    for name, live, required in (
        ("dense retrieval", agent.dense is not None, True),
        ("intent classifier", agent.intent_classifier is not None, True),
        ("override detector", agent.override_detector is not None, True),
        ("masked LM", agent.lm_scorer is not None, require_lm),
    ):
        if live:
            line(OK, name, "live")
        elif required:
            line(FAIL, name, "DARK -- the agent will run degraded and score differently than measured")
            problems.append(name)
        else:
            line(WARN, name, "dark (optional)")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strict", action="store_true", help="exit non-zero if anything required is missing")
    parser.add_argument("--require-lm", action="store_true", help="treat the masked LM as required, not optional")
    parser.add_argument("--static-only", action="store_true", help="file checks only; don't load the models")
    args = parser.parse_args()

    print("preflight: checking local runtime assets\n")
    problems = check_static()
    if not args.static_only and "encoder" not in problems:
        problems += check_live(args.require_lm)
    elif "encoder" in problems:
        print("\n-- skipping the live check: the encoder is missing, everything downstream would be dark --")

    print()
    if problems:
        print(f"INCOMPLETE: {len(problems)} problem(s): {', '.join(problems)}")
        print("Fix with:  uv run python3 scripts/fetch_assets.py")
        return 1 if args.strict else 0
    print("OK: all runtime assets present; the agent runs fully offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
