"""Assemble the submission bundle, then verify it actually runs.

`docs/submission_rules.md` requires the submitted package to stand on its own:
an entry file exporting `Agent`, its helper modules, dependency installation
steps, one command to run in the official harness, and a report. It warns that
"if your code cannot be reproduced from the submitted bundle and instructions,
the organizer may treat the run as invalid."

The bundle is *generated*, never hand-maintained, so it cannot drift from the
source tree. Layout follows the rules' recommended shape:

    submission/
      agent.py          entry point; exports Agent
      requirements.txt  pinned runtime dependencies
      README.md         setup, run command, environment variables
      REPORT.md         the required method/limitations/cost report
      src/              the agent package (copied from starter/)
      tools/            fetch_assets.py, preflight.py, and their bootstrap

`data/` and `model/` are deliberately *not* copied -- together they are ~460 MB
and the catalog is organizer-provided. `tools/fetch_assets.py` materializes
them into the bundle, and `tools/preflight.py` verifies the result offline.

Usage:
    uv run python3 scripts/build_submission.py
    uv run python3 scripts/build_submission.py --verify     # also run the harness
    uv run python3 scripts/build_submission.py --out /tmp/sub
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT  # noqa: F401

DEFAULT_OUT = REPO_ROOT / "dist" / "submission"
AGENT_PACKAGE = REPO_ROOT / "starter"
TOOLS = ["fetch_assets.py", "preflight.py", "_common.py"]

ENTRY = '''"""Submission entry point.

The organizer's harness imports `Agent` from here. The implementation lives in
`src/`, which is the agent package copied verbatim from the source repository.
"""
from src.agent import Agent

__all__ = ["Agent"]
'''

README = '''# TechJam Conversational Search — Submission

An AI shopping agent that finds a simulated customer's hidden target product
within 10 conversational turns, asking clarifying questions along the way.

**Score on the 200-sample public set: TechnicalScore {score}**
(HitRate@10 {hit} / MRR {mrr} / MTTC {mttc}).

See `REPORT.md` for the method, model choice, limitations, and the required
latency / token-usage / cost disclosure.

## Requirements

Python **3.12 or later**.

## Setup

```bash
# 1. install dependencies (CPU-only torch)
pip install -r requirements.txt

# 2. place the organizer's catalog at data/catalog.jsonl
mkdir -p data && cp /path/to/catalog.jsonl data/catalog.jsonl

# 3. fetch model weights and build the dense index
#    THIS IS THE ONLY STEP THAT NEEDS NETWORK ACCESS.
python3 tools/fetch_assets.py

# 4. verify the pipeline comes up whole with the network disabled
python3 tools/preflight.py --strict
```

Step 4 is not optional. With missing weights the agent **degrades silently**:
dense retrieval, both intent classifiers and the masked LM all go dark while
the agent still starts and still returns ten recommendations. `preflight.py`
loads the agent with `HF_HUB_OFFLINE=1` and exits non-zero if any required
component is dark.

## Running

The harness imports `Agent` from `agent.py`:

```python
from agent import Agent
```

To run the official local harness with this bundle on the path:

```bash
PYTHONPATH=. python3 -m evaluator.local_evaluator
```

## Network access

**None required at scoring time.** Both models are local, frozen and CPU-only,
and no hosted LLM API is called on any turn — reported token usage is
genuinely zero. Only the one-time `tools/fetch_assets.py` setup step reaches
the network.

## Environment variables

All optional; every one has a working default.

| variable | purpose |
|---|---|
| `TECHJAM_CATALOG` | catalog path, if not `data/catalog.jsonl` |
| `TECHJAM_DENSE_INDEX` | dense index directory, if not `data/dense_index` |
| `TECHJAM_MODEL_DIR` | model weights directory, if not `model/` |
| `TECHJAM_PROFILE_STORE` | long-term user-profile store path |

Paths resolve against the working directory first, then against this bundle's
own directory, so the harness may be run from either.
'''


def rename_package_refs(text: str) -> str:
    """Rewrite `starter` -> `src` for the bundle.

    Imports must be rewritten for the code to run at all. Prose references are
    rewritten too: the bundle is read by the organizer, and a docstring
    pointing at a `starter/` directory that does not exist in the bundle is
    a stale instruction, not a harmless comment.
    """
    text = re.sub(r"\bfrom starter\.", "from src.", text)
    text = re.sub(r"\bimport starter\.", "import src.", text)
    text = re.sub(r"\bstarter/", "src/", text)
    text = re.sub(r"`starter`", "`src`", text)
    return text


def copy_package(out: Path) -> None:
    """Copy starter/ to src/, rewriting package references."""
    dest = out / "src"
    shutil.copytree(AGENT_PACKAGE, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for path in dest.rglob("*.py"):
        text = path.read_text()
        rewritten = rename_package_refs(text)
        if rewritten != text:
            path.write_text(rewritten)


def copy_tools(out: Path) -> None:
    """Copy the setup tools, repointing them at the bundle's own layout."""
    dest = out / "tools"
    dest.mkdir(parents=True, exist_ok=True)
    for name in TOOLS:
        text = rename_package_refs((REPO_ROOT / "scripts" / name).read_text())
        # In the bundle these live in tools/, not scripts/.
        text = re.sub(r"\bscripts/", "tools/", text)
        (dest / name).write_text(text)


def read_score() -> dict:
    """Pull the headline metrics out of the report so they cannot drift."""
    report = (REPO_ROOT / "docs" / "team_report.md").read_text()
    match = re.search(
        r"HitRate@10\s+([\d.]+)\s+MRR\s+([\d.]+)\s+MTTC\s+([\d.]+).*?TechnicalScore\s+([\d.]+)",
        report, re.S)
    if not match:
        return {"hit": "?", "mrr": "?", "mttc": "?", "score": "?"}
    hit, mrr, mttc, score = match.groups()
    return {"hit": hit, "mrr": mrr, "mttc": mttc, "score": score}


def build(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    (out / "agent.py").write_text(ENTRY)
    copy_package(out)
    copy_tools(out)
    shutil.copy2(REPO_ROOT / "requirements.txt", out / "requirements.txt")
    shutil.copy2(REPO_ROOT / "docs" / "team_report.md", out / "REPORT.md")
    (out / "README.md").write_text(README.format(**read_score()))

    files = sorted(p for p in out.rglob("*") if p.is_file())
    total_kb = sum(p.stat().st_size for p in files) / 1024
    print(f"built {out} -- {len(files)} files, {total_kb:.0f} KB")
    for p in files:
        print(f"    {p.relative_to(out)}")


def verify(out: Path, limit: int | None = None) -> int:
    """Import the bundle's Agent from a neutral directory and score it.

    Runs in a subprocess with the bundle first on sys.path, from a working
    directory that is *not* the repo root -- the point is to prove the bundle
    stands on its own, so anything it accidentally inherits from the source
    tree has to fail here rather than at scoring time.
    """
    print("\n-- verifying: importing Agent from the bundle and scoring --", flush=True)
    slice_line = f"samples = samples[:{limit}]" if limit else "pass"
    script = f"""
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, {str(out)!r})
sys.path.insert(0, {str(REPO_ROOT)!r})   # the organizer-provided evaluator
os.environ["TECHJAM_PROFILE_STORE"] = str(Path(tempfile.mkdtemp()) / "p.json")
os.environ["TECHJAM_CATALOG"] = {str(REPO_ROOT / "data" / "catalog.jsonl")!r}
os.environ["TECHJAM_DENSE_INDEX"] = {str(REPO_ROOT / "data" / "dense_index")!r}
os.environ["TECHJAM_MODEL_DIR"] = {str(REPO_ROOT / "model")!r}

from agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

agent = Agent()
assert agent.dense is not None, "dense retrieval dark in the bundle"
assert agent.intent_classifier is not None, "intent classifier dark in the bundle"
assert agent.override_detector is not None, "override detector dark in the bundle"
print("components: dense/intent/override all live")

samples = load_jsonl({str(REPO_ROOT / "data" / "public_set.jsonl")!r})
{slice_line}
cid, cats, prods = catalog_index({str(REPO_ROOT / "data" / "catalog.jsonl")!r})
report = evaluate(agent, samples, cid, cats, prods)
print("BUNDLE n=%d TechnicalScore=%.6f HitRate=%.4f MRR=%.6f MTTC=%.3f" % (
    len(samples), report["recommended_technical_score"], report["hit_rate_at_10"],
    report["mrr"], report["mttc"]))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO_ROOT.parent,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if not line.startswith("Loading weights"):
            print(line)
    if result.returncode != 0:
        print(f"VERIFY FAILED (exit {result.returncode})", file=sys.stderr)
        tail = [ln for ln in result.stderr.splitlines()
                if "Loading weights" not in ln and "HF_TOKEN" not in ln]
        print("\n".join(tail[-25:]), file=sys.stderr)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None,
                        help="verify on the first N samples only (packaging check; "
                             "omit for the full-set score)")
    parser.add_argument("--verify", action="store_true",
                        help="after building, import the bundle's Agent from a neutral "
                             "directory and score it on the full public set")
    args = parser.parse_args()

    build(args.out)
    if args.verify:
        return verify(args.out, args.limit)
    print("\nVerify with:  uv run python3 scripts/build_submission.py --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
