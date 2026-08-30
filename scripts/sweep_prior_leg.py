"""Sweep a third RRF leg built from catalog priors, and from the user profile.

Never run. rating_number is the only 100%-covered catalog field and targets come
overwhelmingly from the popular tail: the top 1% of the catalog holds 63% of all
200 targets, median target rank 275 of 50,000.

    popularity      rank the pool by rating_number desc      (profile-independent)
    rating          rank the pool by average_rating desc     (profile-independent)
    profile_rating  rank by closeness to a rating the profile predicts

**The controls are the point**: a gain from profile_rating means nothing unless
it beats `rating`, since it could be entirely the global quality prior underneath.
A weighted RRF leg avoids the pathology that killed the three earlier profile
attempts (retrieval-experiments SKILL.md #5) -- each leg only ever adds weight/(k+rank), so an unranked
item scores zero rather than sinking below every neutral candidate.

The leg re-ranks the union of the BM25 and dense candidates, never the catalog: a
leg ordering all 50k by popularity injects the same constant bias into every
session and cannot discriminate.

Run --weights 0 first and confirm it reproduces the shipped score.

    uv run python3 scripts/sweep_prior_leg.py --variant popularity [--weights 0,0.5,1]
"""
from __future__ import annotations

import argparse
import json
import time

from _common import DEFAULT_CATALOG, DEFAULT_DATASET, isolate_profile_store  # noqa: F401

# _common puts the repo root on sys.path as an import side effect, so the
# starter/evaluator imports below resolve when this script is run directly.
import starter.agent as agent_mod  # noqa: E402
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent, routing_params  # noqa: E402
from starter.retrieval import reciprocal_rank_fusion  # noqa: E402

DEFAULT_WEIGHTS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

# Catalog mean, used as the origin the profile's prior is read against. Taken
# from the catalog at load time rather than hardcoded.
_STATS: dict[str, float] = {}


def load_priors(catalog_path) -> dict[str, tuple[float, float]]:
    """parent_asin -> (average_rating, rating_number)."""
    priors: dict[str, tuple[float, float]] = {}
    ratings = []
    with open(catalog_path) as fh:
        for line in fh:
            p = json.loads(line)
            asin = str(p.get("parent_asin"))
            rating = p.get("average_rating")
            count = p.get("rating_number")
            rating = float(rating) if isinstance(rating, (int, float)) else 0.0
            count = float(count) if isinstance(count, (int, float)) else 0.0
            priors[asin] = (rating, count)
            if rating:
                ratings.append(rating)
    _STATS["mean_rating"] = sum(ratings) / max(len(ratings), 1)
    return priors


class PriorLegAgent(Agent):
    """Agent with a third, prior-derived RRF leg spliced into `_retrieve`."""

    variant = "popularity"
    leg_weight = 0.0
    priors: dict[str, tuple[float, float]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        super().reset(session_id, user_profile)
        # The shipped SessionState keeps only a profile *key*; the profile-
        # conditioned variant needs the raw dict, so stash it beside the state.
        self._sessions[session_id].raw_profile = user_profile if isinstance(user_profile, dict) else {}

    def _prior_rank(self, candidate_ids: list[str], profile: dict) -> list[str]:
        if self.variant == "popularity":
            key = lambda a: -self.priors.get(a, (0.0, 0.0))[1]  # noqa: E731
        elif self.variant == "rating":
            key = lambda a: -self.priors.get(a, (0.0, 0.0))[0]  # noqa: E731
        else:
            # Profile-conditioned. average_prior_rating correlates with the
            # target's own average_rating at r = +0.18 on the public set, so
            # predict a target rating that leans the same direction as the
            # customer's prior, and rank by closeness to it. Deliberately a
            # fixed, unfitted map -- fitting the coefficient on the public set
            # would leak the labels this is meant to be tested against.
            prior = profile.get("average_prior_rating")
            prior = float(prior) if isinstance(prior, (int, float)) else _STATS["mean_rating"]
            predicted = _STATS["mean_rating"] + 0.25 * (prior - _STATS["mean_rating"])
            key = lambda a: abs(self.priors.get(a, (0.0, 0.0))[0] - predicted)  # noqa: E731
        return sorted(candidate_ids, key=key)

    def _retrieve(self, query, top_k, pool_size, intent_label, disclosed=None, inferred=None):
        if not query.strip():
            return []
        try:
            bm25_ranked = self.bm25.search(query, agent_mod.CANDIDATE_N)
        except Exception:
            bm25_ranked = []
        dense_ranked = []
        if self.dense is not None:
            try:
                dense_ranked = self.dense.search(query, agent_mod.CANDIDATE_N)
            except Exception:
                dense_ranked = []

        routing = routing_params(intent_label)
        legs = [(bm25_ranked, routing.bm25_weight), (dense_ranked, routing.dense_weight)]

        if self.leg_weight > 0:
            union = list(dict.fromkeys([*bm25_ranked, *dense_ranked]))
            if union:
                legs.append((self._prior_rank(union, getattr(self, "_current_profile", {})), self.leg_weight))

        rank_lists = [ranked for ranked, _ in legs if ranked]
        weights = [weight for ranked, weight in legs if ranked]
        if not rank_lists:
            return []
        fused = reciprocal_rank_fusion(rank_lists, top_n=pool_size, weights=weights)
        candidates = self._boost_by_disclosed(fused, disclosed, inferred) if (disclosed or inferred) else fused
        return self._diversify(candidates, top_k) if routing.diversify else candidates

    def respond(self, session_id, user_message, turn, top_k):
        state = self._sessions.get(session_id)
        self._current_profile = getattr(state, "raw_profile", {}) if state else {}
        return super().respond(session_id, user_message, turn, top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["popularity", "rating", "profile_rating"], default="popularity")
    parser.add_argument("--weights", default=None, help="comma-separated leg weights (default 0,0.25,0.5,1,1.5,2)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")] if args.weights else DEFAULT_WEIGHTS

    isolate_profile_store()
    samples = load_jsonl(DEFAULT_DATASET)
    if args.limit:
        samples = samples[: args.limit]
    catalog = catalog_index(DEFAULT_CATALOG)
    priors = load_priors(DEFAULT_CATALOG)

    print(f"variant={args.variant}  weights={weights}  samples={len(samples)}")
    print("(weight 0 is the shipped two-leg baseline)\n")
    header = f"{'weight':>7} {'HitRate':>9} {'MRR':>9} {'MTTC':>8} {'Technical':>11} {'vs w=0':>9}  {'mins':>5}"
    print(header)
    print("-" * len(header))

    rows, base = [], None
    for weight in weights:
        PriorLegAgent.variant = args.variant
        PriorLegAgent.leg_weight = weight
        PriorLegAgent.priors = priors
        start = time.perf_counter()
        agent = PriorLegAgent()
        report = evaluate(agent, samples, *catalog)
        mins = (time.perf_counter() - start) / 60
        score = report["recommended_technical_score"]
        if weight == 0.0:
            base = score
        rows.append({"weight": weight, "hit_rate_at_10": report["hit_rate_at_10"],
                     "mrr": report["mrr"], "mttc": report["mttc"], "technical_score": score})
        delta = "" if base is None else f"{score - base:+.4f}"
        print(f"{weight:7.2f} {report['hit_rate_at_10']:9.4f} {report['mrr']:9.4f} "
              f"{report['mttc']:8.3f} {score:11.4f} {delta:>9}  {mins:5.1f}", flush=True)

    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"variant": args.variant, "curve": rows}, fh, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
