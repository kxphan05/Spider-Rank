"""Persistent, cross-session long-term user-profile store.

No customer/user id exists anywhere in the API contract, so this module keys
the store off a stable hash of the anonymized `user_profile` dict itself --
two sessions with an identical profile are treated as the same returning
shopper. Measured to carry no usable signal though (shared-key sessions want
the same category only 0.5% of the time, vs. a 1.2% random baseline -- see
the retrieval-experiments skill #5), so the `carried` value this returns is
threaded into `SessionState.profile_hint` but deliberately unread.

Storage is a single write-through JSON file (no server/DB needed at this
scale); write-through because the API has no "session ended" hook to flush
on. Disclosure history is keyed by a store-local session_index counter
rather than wall-clock time, since batch eval runs sessions milliseconds apart.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from threading import Lock
from .config import (DEFAULT_STORE_PATH, MIN_CORROBORATION, STORE_PATH_ENV)

logger = logging.getLogger(__name__)



def default_store_path() -> Path:
    """Store location: `$TECHJAM_PROFILE_STORE` if set, else the repo default."""
    override = os.environ.get(STORE_PATH_ENV)
    return Path(override) if override else DEFAULT_STORE_PATH




def profile_key(user_profile: dict) -> str:
    """Stable identity key for an anonymized `user_profile` dict.

    Canonicalized via sorted-key JSON so field order never changes the key;
    truncated to 16 hex chars, plenty at this scale and easier to read in logs.
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

        Returns `(key, session_index, carried)`: `carried` is the most-recent
        historical value per attribute, for the caller to seed as a
        boost-only prior -- never as if this customer said it this session.
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
