"""Dev wrapper around evaluator.local_evaluator.evaluate().

Same evaluate() call as `python3 -m evaluator.local_evaluator` (scoring logic
untouched), plus the knobs you want while iterating:

    --limit N        first N samples only
    --scenario TYPE  restrict to buying / browsing / intent_override / boundary
    --seed N         shuffle before --limit, for a random subset
    --quiet          suppress the per-scenario table and baseline diff
    --no-progress    suppress the tqdm bar on stderr

Also prints a per-scenario breakdown and, when docs/baseline_results.json exists,
a diff against the weak-BM25 baseline. Each run gets a fresh profile store so
repeated runs are independent.

    uv run python3 scripts/run_eval.py [--limit 40] [--scenario buying]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402
from starter.user_profile import STORE_PATH_ENV  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--baseline", default="docs/baseline_results.json")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N samples (after --scenario/--seed)")
    parser.add_argument("--scenario", action="append", default=None,
                         help="restrict to this scenario_type (repeatable): buying, browsing, intent_override, boundary")
    parser.add_argument("--seed", type=int, default=None, help="shuffle samples before --limit, for a random subset")
    parser.add_argument("--quiet", action="store_true", help="skip the per-scenario table and baseline diff")
    parser.add_argument("--no-progress", action="store_true", help="suppress the tqdm progress bar")
    parser.add_argument(
        "--profile-store", default=None,
        help="long-term user-profile store path. Default: a fresh per-run temp file, so repeated "
             "runs are independent (see below). Pass 'data/user_profiles.json' to reuse the "
             "persistent store, or any fixed path to accumulate history deliberately.",
    )
    return parser.parse_args()


def with_progress(samples: list[dict], enabled: bool):
    """Wrap `samples` in a tqdm bar for evaluate() to consume.

    evaluate() only iterates its `samples` argument, so wrapping it here keeps
    the evaluator itself untouched (this script's read-only-evaluator rule).
    tqdm arrives transitively with sentence-transformers rather than as a
    declared dependency, so a missing install degrades to no bar instead of
    breaking the run.
    """
    if not enabled:
        return samples
    try:
        from tqdm import tqdm
    except ImportError:
        print("tqdm not installed; running without a progress bar", file=sys.stderr)
        return samples
    return tqdm(samples, desc="samples", unit="sample", file=sys.stderr)


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

    # Isolate the long-term profile store per run unless asked otherwise. The
    # store is write-through and outlives the process, so without this each
    # run reads back history written by the previous one: measured over the
    # 200-sample public set, the number of sessions whose reset() received a
    # carried cross-session hint went 45 -> 105 -> 200 (of 200) across three
    # consecutive runs of the same samples. That makes any A/B of an agent
    # that *consumes* those hints depend on how many times you had run the
    # eval before, which is not a property you want in a benchmark. Real
    # cross-session persistence still happens in the graded single pass, and
    # can be reproduced here on purpose with --profile-store.
    store_path = args.profile_store
    if store_path is None:
        store_path = str(Path(tempfile.mkdtemp(prefix="techjam-profiles-")) / "user_profiles.json")
        print(f"isolating profile store at {store_path} (override with --profile-store)", file=sys.stderr)
    os.environ[STORE_PATH_ENV] = store_path

    print(f"loading agent + running {len(samples)} sample(s)...", file=sys.stderr)
    agent = Agent(args.catalog)
    start = time.monotonic()
    result = evaluate(
        agent, with_progress(samples, not args.no_progress), catalog_ids, categories, products
    )
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
