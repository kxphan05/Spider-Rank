"""Convenience wrapper around evaluator.local_evaluator.evaluate() for local dev iteration.

`python3 -m evaluator.local_evaluator` always scores the full 200-sample public
set. This script calls the same `evaluate()` function (read-only use, no
changes to scoring logic) but adds the knobs you actually want while
iterating on `starter/agent.py`:

  --limit N          only run the first N samples (fast smoke test)
  --scenario TYPE     restrict to one scenario_type (repeatable): buying,
                      browsing, intent_override, boundary
  --seed N            shuffle samples before applying --limit, for a random
                      subset instead of always the same first N
  --quiet             suppress the per-scenario table and baseline diff

It also prints a per-scenario breakdown table and, when
docs/baseline_results.json is present, a diff against the weak-BM25 baseline
-- neither of which the bare evaluator CLI prints.

Usage:
    uv run python3 scripts/run_eval.py
    uv run python3 scripts/run_eval.py --limit 40
    uv run python3 scripts/run_eval.py --scenario buying --scenario intent_override
    uv run python3 scripts/run_eval.py --seed 7 --limit 50 --output /tmp/quick.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--baseline", default="docs/baseline_results.json")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N samples (after --scenario/--seed)")
    parser.add_argument("--scenario", action="append", default=None,
                         help="restrict to this scenario_type (repeatable): buying, browsing, intent_override, boundary")
    parser.add_argument("--seed", type=int, default=None, help="shuffle samples before --limit, for a random subset")
    parser.add_argument("--quiet", action="store_true", help="skip the per-scenario table and baseline diff")
    return parser.parse_args()


def print_scenario_table(scenario_metrics: dict[str, dict]) -> None:
    print("\n--- by scenario ---")
    for name, metrics in scenario_metrics.items():
        mttc = metrics["mttc"]
        mttc_str = f"{mttc:.2f}" if mttc is not None else "n/a"
        print(f"  {name:16s} n={metrics['sample_count']:3d}  "
              f"hit={metrics['hit_rate_at_10']:.3f}  mrr={metrics['mrr']:.3f}  mttc={mttc_str}")


def print_baseline_diff(result: dict, baseline_path: Path) -> None:
    if not baseline_path.exists():
        return
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    print(f"\n--- vs baseline ({baseline.get('baseline', baseline_path.name)}) ---")
    for key in ("hit_rate_at_10", "mrr", "mttc"):
        cur, base = result.get(key), baseline.get(key)
        if cur is None or base is None:
            continue
        delta = cur - base
        print(f"  {key:20s} {cur:.4f}  (baseline {base:.4f}, {delta:+.4f})")
    cur_score, base_score = result.get("recommended_technical_score"), baseline.get("technical_score")
    if cur_score is not None and base_score is not None:
        delta = cur_score - base_score
        print(f"  {'technical_score':20s} {cur_score:.4f}  (baseline {base_score:.4f}, {delta:+.4f})")


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.dataset)

    if args.scenario:
        allowed = set(args.scenario)
        samples = [sample for sample in samples if sample["scenario_type"] in allowed]

    if args.seed is not None:
        random.Random(args.seed).shuffle(samples)

    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        raise SystemExit("no samples left after filtering -- check --dataset/--scenario")

    catalog_ids, categories, products = catalog_index(args.catalog)

    print(f"loading agent + running {len(samples)} sample(s)...", file=sys.stderr)
    start = time.monotonic()
    result = evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    elapsed = time.monotonic() - start

    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "sessions"}
    print(json.dumps(summary, indent=2))
    print(f"\n{len(samples)} sample(s) in {elapsed:.1f}s -> {args.output}", file=sys.stderr)

    if not args.quiet:
        print_scenario_table(result["scenario_metrics"])
        print_baseline_diff(result, Path(args.baseline))


if __name__ == "__main__":
    main()
