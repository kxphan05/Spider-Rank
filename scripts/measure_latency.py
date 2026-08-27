"""Measure the agent's runtime cost: startup, per-turn latency, memory, tokens.

`docs/submission_rules.md` requires a disclosure of latency, token usage, and
estimated model cost. This script produces those numbers rather than
estimating them, and is the source for the corresponding section of
`docs/team_report.md` -- re-run it if the pipeline changes.

It wraps the real `Agent` in a timing proxy and drives it through the real
`evaluate()` session loop, so the per-turn numbers include everything a
scored turn actually does (retrieval, fusion, boost, classifiers, question
selection) and nothing it doesn't.

Usage:
    uv run python3 scripts/measure_latency.py --limit 20
    uv run python3 scripts/measure_latency.py --limit 20 --no-lm
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.user_profile import STORE_PATH_ENV  # noqa: E402

PUBLIC_SET = Path("data/public_set.jsonl")
CATALOG = Path("data/catalog.jsonl")


class TimingAgent:
    """Delegating proxy that records wall-clock per `respond()` call."""

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        self.turn_ms: list[float] = []
        self.reset_ms: list[float] = []

    def reset(self, session_id: str, user_profile: dict) -> None:
        start = time.perf_counter()
        self._agent.reset(session_id, user_profile)
        self.reset_ms.append((time.perf_counter() - start) * 1000)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        start = time.perf_counter()
        response = self._agent.respond(session_id, user_message, turn, top_k)
        self.turn_ms.append((time.perf_counter() - start) * 1000)
        return response


def peak_rss_mb() -> float:
    # ru_maxrss is kilobytes on Linux, bytes on macOS.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024 if sys.platform != "darwin" else peak / (1024 * 1024)


def summarize(name: str, values: list[float]) -> dict:
    if not values:
        return {"name": name, "n": 0}
    ordered = sorted(values)
    return {
        "name": name,
        "n": len(values),
        "mean_ms": round(statistics.fmean(ordered), 1),
        "p50_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1] if len(ordered) >= 20 else ordered[-1], 1),
        "max_ms": round(ordered[-1], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=20, help="samples to drive (default 20)")
    parser.add_argument("--no-lm", action="store_true",
                        help="disable the masked-LM attribute inference, to price its contribution")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    # Same isolation the eval wrapper uses: a shared store across runs would
    # leak corroborated disclosures between measurements.
    store_dir = tempfile.mkdtemp(prefix="latency_profiles_")
    os.environ[STORE_PATH_ENV] = str(Path(store_dir) / "user_profiles.json")

    rss_before = peak_rss_mb()

    start = time.perf_counter()
    agent = Agent()
    init_s = time.perf_counter() - start

    lm_enabled = agent.lm_scorer is not None
    if args.no_lm:
        agent.lm_scorer = None
        lm_enabled = False

    rss_after_init = peak_rss_mb()

    samples = load_jsonl(PUBLIC_SET)[: args.limit]
    catalog_ids, categories, products = catalog_index(CATALOG)

    timed = TimingAgent(agent)
    wall_start = time.perf_counter()
    report = evaluate(timed, samples, catalog_ids, categories, products)
    wall_s = time.perf_counter() - wall_start

    result = {
        "samples": len(samples),
        "lm_inference_enabled": lm_enabled,
        "startup": {
            "agent_init_s": round(init_s, 2),
            "note": "cold process; includes BM25 build, dense index mmap, encoder + LM load",
        },
        "per_turn": summarize("respond", timed.turn_ms),
        "per_reset": summarize("reset", timed.reset_ms),
        "per_session": {
            "turns_mean": round(len(timed.turn_ms) / max(len(samples), 1), 2),
            "latency_mean_ms": round(sum(timed.turn_ms) / max(len(samples), 1), 1),
        },
        "memory": {
            "peak_rss_mb_after_init": round(rss_after_init, 1),
            "peak_rss_mb_total": round(peak_rss_mb(), 1),
            "baseline_rss_mb": round(rss_before, 1),
        },
        "wall_clock_s": round(wall_s, 1),
        "token_usage": report.get("reported_token_usage"),
        "score": {
            "hit_rate_at_10": report.get("hit_rate_at_10"),
            "mrr": report.get("mrr"),
            "mttc": report.get("mttc"),
            "recommended_technical_score": report.get("recommended_technical_score"),
        },
    }

    print(json.dumps(result, indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
