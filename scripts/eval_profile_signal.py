"""Does `user_profile` carry any signal worth acting on?

`starter/user_profile.py` persists a long-term, cross-session profile store
(TODO.md III, "continuously updating ... long-term user profiles"), but its
carried value (`SessionState.profile_hint`) is deliberately inert: every way
of consuming it that has been tried regressed the full public set (CLAUDE.md
"Known open problems" #5). This script asks the prior question those
experiments skipped -- *is there signal in the profile at all?* -- so the
answer is a measurement rather than a sequence of failed A/Bs.

Four independent checks, all read-only (no Agent, no retrieval, so it runs in
seconds rather than the minutes a full eval takes):

  tags      Does a session's own `preference_tags` predict which attribute
            buckets its customer can actually answer? This is the one profile
            use CLAUDE.md #5 left open, precisely because it needs no
            cross-session identity assumption -- it only uses the profile
            handed to reset() this session. Compares each (tag, bucket) count
            against Binomial(tag support, bucket marginal) -- the null in
            which the tag tells you nothing -- and flags any cell more than
            SIGMA_THRESHOLD standard errors away.
            "Answerable" is computed from the evaluator's own reply policy:
            customer_reply() only ever discloses a constraint whose
            classify_constraint() bucket equals the asked attribute, so the
            buckets present in a sample's undisclosed constraints are exactly
            the questions that can pay off.

  scenario  Does any profile field predict scenario_type (buying / browsing /
            intent_override / boundary)? If it did, it would be a cheaper and
            far more robust intent router than the nearest-centroid embedding
            classifier, whose margins are documented as sitting inside the
            noise floor (CLAUDE.md #7).

  collision Do two sessions sharing a profile_key actually want similar
            products? This is the load-bearing assumption behind the whole
            cross-session store: with no customer id in the API contract, the
            key is a content hash of the anonymized profile, so it is only
            useful if a shared key implies a shared shopper. Compares the rate
            at which within-key target pairs share a coarse_category against a
            random-pair baseline.

  store     How fast does re-running the eval contaminate the store? The store
            is write-through and outlives the process, so run N+1 reads
            history written by run N.

Usage:
    uv run python3 scripts/eval_profile_signal.py
    uv run python3 scripts/eval_profile_signal.py --check tags --check collision
"""
from __future__ import annotations

import argparse
import collections
import itertools
import math
import random
import statistics
import tempfile
from pathlib import Path

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, REPO_ROOT  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.

from evaluator.local_evaluator import (  # noqa: E402
    catalog_index, classify_constraint, coarse_category, initial_message,
    load_jsonl, materialize_hidden_fields,
)
from starter.user_profile import UserProfileStore, profile_key  # noqa: E402

# A tag/field value seen fewer times than this is reported but never flagged
# as signal -- at n<30 a lift of 2.0 is one or two samples moving.
MIN_SUPPORT = 30
# Flag a (tag, bucket) cell only when its count departs from the null by more
# than this many standard errors. Deliberately NOT a threshold on lift: the
# rare buckets have tiny marginals (use_case is answerable in 4 of 200
# samples), so a single sample lands a lift of 2.0+ there while meaning
# nothing. Under the null "the tag tells you nothing", a cell is
# Binomial(n=tag support, p=bucket marginal), which prices that in directly.
SIGMA_THRESHOLD = 2.0


def answerable_buckets(sample: dict, categories: dict[str, list[str]], products: dict[str, dict]) -> set[str]:
    """Attribute buckets this sample's customer can still disclose after turn 1.

    Mirrors the evaluator: initial_message() marks hard_constraints[0] as
    already disclosed for `buying`, and customer_reply() matches the remaining
    constraints by classify_constraint() bucket.
    """
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    disclosed: set[str] = set()
    initial_message(effective, coarse_category(categories.get(target, [])), disclosed)
    constraints = [
        *[str(value) for value in card.get("hard_constraints", [])],
        *[str(value) for value in card.get("soft_preferences", [])],
    ]
    return {classify_constraint(value) for value in constraints if value not in disclosed}


def check_tags(samples: list[dict], categories: dict, products: dict) -> None:
    print("=== preference_tags -> answerable attribute buckets ===")
    per_sample = [(sample, answerable_buckets(sample, categories, products)) for sample in samples]
    total = len(per_sample)
    marginal = collections.Counter(bucket for _, buckets in per_sample for bucket in buckets)

    print(f"\n  marginal P(bucket answerable), n={total}:")
    for bucket, count in marginal.most_common():
        print(f"    {bucket:10s} {count / total:.3f}  ({count})")

    tag_total: collections.Counter = collections.Counter()
    tag_bucket: collections.Counter = collections.Counter()
    for sample, buckets in per_sample:
        for tag in sample["user_profile"].get("preference_tags") or []:
            tag_total[tag] += 1
            for bucket in buckets:
                tag_bucket[(tag, bucket)] += 1

    print(f"\n  P(bucket | tag) vs marginal, flagged beyond {SIGMA_THRESHOLD:g} sigma:")
    flagged = 0
    peak_sigma = 0.0
    for tag, support in tag_total.most_common():
        note = "" if support >= MIN_SUPPORT else "  (low support, not flagged)"
        print(f"    tag={tag!r} n={support}{note}")
        for bucket, count in marginal.most_common():
            observed = tag_bucket[(tag, bucket)]
            base = count / total
            expected = support * base
            spread = math.sqrt(support * base * (1 - base))
            sigma = abs(observed - expected) / spread if spread else 0.0
            lift = (observed / support) / base if base else 0.0
            interesting = support >= MIN_SUPPORT and sigma > SIGMA_THRESHOLD
            flagged += interesting
            if support >= MIN_SUPPORT:
                peak_sigma = max(peak_sigma, sigma)
            print(f"      {bucket:10s} P={observed / support:.3f}  marginal={base:.3f}  "
                  f"lift={lift:.2f}  {sigma:.1f}sigma{'   <-- SIGNAL' if interesting else ''}")
    verdict = (f"{flagged} tag/bucket pair(s) beyond {SIGMA_THRESHOLD:g} sigma"
               if flagged else
               f"nothing beyond {SIGMA_THRESHOLD:g} sigma (largest departure {peak_sigma:.1f} sigma) -- "
               "a session's own preference_tags do not predict what it can answer")
    print(f"\n  verdict: {verdict}")


def check_scenario(samples: list[dict]) -> None:
    print("\n=== profile fields -> scenario_type ===")
    total = len(samples)
    marginal = collections.Counter(sample["scenario_type"] for sample in samples)
    scenarios = [name for name, _ in marginal.most_common()]
    print(f"  marginal, n={total}: " + "  ".join(f"{name}={marginal[name] / total:.2f}" for name in scenarios))

    def report(label: str, grouped: dict[object, collections.Counter]) -> None:
        print(f"\n  field={label}")
        if len(grouped) == 1:
            print("    (constant across the whole set -- zero information by construction)")
        for value, counts in sorted(grouped.items(), key=lambda item: -sum(item[1].values())):
            support = sum(counts.values())
            note = "" if support >= MIN_SUPPORT else "  (low support)"
            distribution = "  ".join(f"{name}={counts[name] / support:.2f}" for name in scenarios)
            print(f"    {str(value):24s} n={support:3d}  {distribution}{note}")

    for field in ("purchase_frequency", "rating_style", "average_prior_rating"):
        grouped: dict[object, collections.Counter] = collections.defaultdict(collections.Counter)
        for sample in samples:
            grouped[sample["user_profile"].get(field)][sample["scenario_type"]] += 1
        report(field, grouped)

    by_tag: dict[object, collections.Counter] = collections.defaultdict(collections.Counter)
    for sample in samples:
        for tag in sample["user_profile"].get("preference_tags") or []:
            by_tag[tag][sample["scenario_type"]] += 1
    report("preference_tags (per tag)", by_tag)


def check_collision(samples: list[dict], categories: dict, trials: int = 200) -> None:
    print("\n=== profile_key collision -> target similarity ===")
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    for sample in samples:
        grouped[profile_key(sample["user_profile"])].append(str(sample["ground_truth"]["parent_asin"]))
    repeated = {key: targets for key, targets in grouped.items() if len(targets) > 1}
    print(f"  distinct keys={len(grouped)}  keys seen >1x={len(repeated)}  "
          f"sessions under them={sum(len(v) for v in repeated.values())}")
    if not repeated:
        print("  no repeated keys -- nothing to compare")
        return

    def category_of(parent_asin: str) -> str:
        return coarse_category(categories.get(parent_asin, []))

    pairs = [pair for targets in repeated.values() for pair in itertools.combinations(targets, 2)]
    agreement = sum(category_of(a) == category_of(b) for a, b in pairs) / len(pairs)

    everything = [str(sample["ground_truth"]["parent_asin"]) for sample in samples]
    rng = random.Random(0)
    rates = []
    for _ in range(trials):
        sampled = [(rng.choice(everything), rng.choice(everything)) for _ in range(len(pairs))]
        rates.append(sum(category_of(a) == category_of(b) for a, b in sampled) / len(sampled))
    baseline, spread = statistics.fmean(rates), statistics.stdev(rates)

    print(f"  within-key pairs={len(pairs)}  share a coarse_category: {agreement:.3f}")
    print(f"  random-pair baseline over {trials} trials:            {baseline:.3f} +- {spread:.3f}")
    if agreement > baseline + 2 * spread:
        print("  verdict: same-key sessions ARE more alike than chance -- carry-over has a basis")
    else:
        print("  verdict: same-key sessions are no more alike than two random sessions.\n"
              "           A shared profile_key is a template collision, not a returning shopper,\n"
              "           so cross-session carry-over has nothing to carry.")


def check_store(samples: list[dict], runs: int = 3) -> None:
    print("\n=== store contamination across repeated eval runs ===")
    store = UserProfileStore(Path(tempfile.mkdtemp(prefix="techjam-signal-")) / "user_profiles.json")
    print("  (each run starts a session per sample and records one disclosure, as a real run would)")
    for run in range(1, runs + 1):
        seen = 0
        for sample in samples:
            key, index, carried = store.start_session(sample["user_profile"])
            seen += bool(carried)
            store.record_disclosure(key, index, "material", "cotton")
        print(f"    run {run}: {seen}/{len(samples)} sessions received a carried cross-session hint at reset()")
    print("  verdict: history accumulates across runs, so an agent that consumes profile_hint\n"
          "           scores differently depending on how many times the eval was run before.\n"
          "           scripts/run_eval.py isolates each run by default (--profile-store).")


CHECKS = ("tags", "scenario", "collision", "store")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--check", action="append", choices=CHECKS, default=None,
                        help="run only this check (repeatable); default runs all four")
    args = parser.parse_args()

    selected = args.check or list(CHECKS)
    samples = load_jsonl(args.dataset)
    _, categories, products = catalog_index(args.catalog)

    if "tags" in selected:
        check_tags(samples, categories, products)
    if "scenario" in selected:
        check_scenario(samples)
    if "collision" in selected:
        check_collision(samples, categories)
    if "store" in selected:
        check_store(samples)


if __name__ == "__main__":
    main()
