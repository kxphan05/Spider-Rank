"""Persistent, cross-session long-term user-profile store (TODO.md III,
"Runtime Adaptation: ... continuously updating short-term session states and
long-term user profiles").

Identity problem: `docs/agent_api_contract.json`'s `reset_request` gives us
`session_id` (fresh per session -- `evaluator/local_evaluator.py` mints a new
`uuid4` per sample) and an anonymized `user_profile` dict (purchase_frequency,
average_prior_rating, rating_style, preference_tags, summary). There is no
customer/user id anywhere in the contract, so nothing marks two sessions as
"the same shopper" except the content of that anonymized profile itself.
This module keys the persistent store off a stable hash of that content:
two sessions presenting an identical anonymized profile are treated as the
same returning shopper profile. This is deliberately coarser than tracking a
literal individual -- on the public set, 125 of 200 samples' profiles are
unique, i.e. profile collisions across ostensibly-different sessions do
happen (a small template set generates the profiles).

That collision rate is not the whole story, and the rest of it is why the
carried value is currently unused. Measured by
`scripts/eval_profile_signal.py --check collision`: across the 409 pairs of
sessions that share a profile key, only 0.5% want a target in the same
coarse category, against a 1.2% +- 0.5% random-pair baseline -- a shared key
implies no more shared taste than picking two sessions at random. So
`start_session()` still returns a `carried` dict and `Agent` still threads it
into `SessionState.profile_hint`, but nothing reads it: three ways of
consuming it were each measured to regress the full public set, and the
collision number above explains why they had to (see CLAUDE.md "Known open
problems" #5). Recording still happens unconditionally -- it is what TODO.md
III asks for, it is cheap, and re-running the diagnostic is how you would
find out whether the hidden set behaves differently.

Storage is a single write-through JSON file (no server, no external DB --
"heavy external industrial vector DB clusters" are explicitly out of scope
per TODO.md, and at this project's scale a flat file is plenty). Write-through
because the API contract has no "session ended" hook to flush on -- a session
can stop at any turn up to MAX_TURNS=10 with no explicit close() call.

Per-attribute disclosure history is stored as an append-only list keyed by a
store-local `session_index` counter (not wall-clock time -- an offline batch
eval runs many sessions within milliseconds of each other, so real
timestamps wouldn't actually separate "old" from "new" the way session count
does). Recency-weighting that history (slot decay) and bounding its growth
(dynamic truncation) are later, additive layers on this same schema -- see
the module docstring update once they land.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

# Overridable so a *local benchmark* run can be isolated from the persistent
# store. This matters more than it looks: the store is write-through and
# survives process exit, so repeated local eval runs accumulate history and
# feed it back into subsequent runs. Measured on the 200-sample public set,
# counting sessions whose reset() received a non-empty carried hint:
# run 1 -> 45/200, run 2 -> 105/200, run 3 -> 200/200. Any A/B of a
# hint-consuming agent is therefore confounded by how many times the eval had
# been run before, and drifts toward "every session is influenced" as you
# iterate. Cross-session persistence is a real, intended feature for the
# graded run (one pass, genuine session history); it is purely an artifact
# when re-scoring the same 200 samples over and over. scripts/run_eval.py
# sets this to a per-run temp path by default; set it explicitly to isolate
# `python3 -m evaluator.local_evaluator` the same way:
#     TECHJAM_PROFILE_STORE=/tmp/store.json uv run python3 -m evaluator.local_evaluator
STORE_PATH_ENV = "TECHJAM_PROFILE_STORE"
DEFAULT_STORE_PATH = Path("data/user_profiles.json")


def default_store_path() -> Path:
    """Store location: `$TECHJAM_PROFILE_STORE` if set, else the repo default."""
    override = os.environ.get(STORE_PATH_ENV)
    return Path(override) if override else DEFAULT_STORE_PATH


# A profile key is a content hash with no customer id behind it (see module
# docstring) -- most repeat-key sessions on this catalog turn out to be a
# coincidental template collision, not a genuine returning shopper. Measured
# directly: carrying forward *any* single historical disclosure as a hint
# regressed the full 200-sample public set (HitRate 0.755->0.745, MRR
# 0.384->0.355, TechnicalScore 0.601->0.585) because a single wrong guess
# sinks the true target below every neutral (unknown-attribute) candidate,
# regardless of how small its boost weight is -- lowering the weight alone
# can't fix that (the experiment's PROFILE_HINT_WEIGHT constant is long gone
# from agent.py; the finding is written up in CLAUDE.md #5). Requiring the
# *same* value to recur at least twice in history before it's trusted as a
# hint filters out one-off coincidental collisions while still catching a
# shopper who has genuinely stated the same preference more than once.
#
# That last sentence is the hypothesis, and it was tested and did not hold:
# gating on corroboration still scored below baseline, because a value
# recurring twice under one key does not make it likelier to be right for a
# third, unrelated session sharing that key. Kept as the gate on `carried`
# anyway -- it is the conservative choice for whatever reads it next, and it
# costs nothing while nothing does.
MIN_CORROBORATION = 2


def profile_key(user_profile: dict) -> str:
    """Stable identity key for an anonymized `user_profile` dict.

    Canonicalized via sorted-key JSON so field order never changes the key;
    truncated to 16 hex chars (64 bits) -- collision risk at this catalog's
    scale (a few hundred distinct profile shapes) is negligible, and a short
    key keeps the on-disk file and debug logs readable.
    """
    if not isinstance(user_profile, dict):
        user_profile = {}
    canonical = json.dumps(user_profile, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class UserProfileStore:
    """Write-through JSON-backed store of long-term per-profile-key state."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_store_path()
        self._lock = Lock()
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.exception("failed to load user profile store at %s; starting empty", self.path)
                self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(self._data), encoding="utf-8")
            tmp_path.replace(self.path)
        except OSError:
            logger.exception("failed to persist user profile store to %s", self.path)

    def start_session(self, user_profile: dict) -> tuple[str, int, dict[str, str]]:
        """Register a new session under this profile's key.

        Returns `(key, session_index, carried)`: `session_index` is this
        profile's 1-based session counter (used later to timestamp
        disclosures from this session), and `carried` is the most-recent
        historical value seen per attribute, for the caller to seed as a
        *boost-only* prior -- never treated as if the customer said it this
        session.
        """
        if not isinstance(user_profile, dict):
            user_profile = {}
        key = profile_key(user_profile)
        with self._lock:
            record = self._data.setdefault(key, {"raw_profile": {}, "session_count": 0, "disclosures": {}})
            record["raw_profile"] = user_profile
            record["session_count"] += 1
            session_index = record["session_count"]
            carried: dict[str, str] = {}
            for attribute, events in record["disclosures"].items():
                if not events:
                    continue
                value, count = Counter(event["value"] for event in events).most_common(1)[0]
                if count >= MIN_CORROBORATION:
                    carried[attribute] = value
            self._save()
        return key, session_index, carried

    def record_disclosure(self, key: str, session_index: int, attribute: str, value: str) -> None:
        """Append a genuinely-this-session disclosure to `key`'s history."""
        with self._lock:
            record = self._data.setdefault(key, {"raw_profile": {}, "session_count": 0, "disclosures": {}})
            record["disclosures"].setdefault(attribute, []).append(
                {"value": value, "session_index": session_index}
            )
            self._save()
