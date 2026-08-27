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
happen (a small template set generates the profiles) -- see CLAUDE.md for
the measured consequence this has on carry-over behavior and why
`Agent`/`SessionState` only use carried values to *boost* retrieval, never to
suppress a clarifying question.

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
from collections import Counter
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("data/user_profiles.json")

# A profile key is a content hash with no customer id behind it (see module
# docstring) -- most repeat-key sessions on this catalog turn out to be a
# coincidental template collision, not a genuine returning shopper. Measured
# directly: carrying forward *any* single historical disclosure as a hint
# regressed the full 200-sample public set (HitRate 0.755->0.745, MRR
# 0.384->0.355, TechnicalScore 0.601->0.585) because a single wrong guess
# sinks the true target below every neutral (unknown-attribute) candidate,
# regardless of how small its boost weight is -- lowering the weight alone
# can't fix that (see agent.py's PROFILE_HINT_WEIGHT comment). Requiring the
# *same* value to recur at least twice in history before it's trusted as a
# hint filters out one-off coincidental collisions while still catching a
# shopper who has genuinely stated the same preference more than once.
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

    def __init__(self, path: str | Path = DEFAULT_STORE_PATH) -> None:
        self.path = Path(path)
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
