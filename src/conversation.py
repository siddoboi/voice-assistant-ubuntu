"""Conversation manager for the voice assistant.

Provides ``ConversationManager``: a per-call session object that holds a
rolling in-memory history of user/assistant turns, prepends a configurable
system prompt when building LLM messages, and persists every turn to a
SQLite database for offline review.

The class is config-driven — all tunables (system prompt, rolling-window
size, database path) come from ``configs/dev_config.yaml`` (or whichever
file ``VOICE_ASSISTANT_CONFIG`` points at, or an explicit ``config_path``
argument). No hardcoded values inside method bodies.

Typical usage during a call::

    with ConversationManager() as cm:
        cm.add_user_turn("what's the time")
        messages = cm.build_messages()
        # ... feed messages to llm_client.stream_generate ...
        cm.add_assistant_turn(reply_text)

Resuming a prior session by id::

    cm = ConversationManager(session_id="abc-123")
    history = cm.get_history()

The rolling window only controls what is sent to the LLM; SQLite retains
every turn ever recorded.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# Module-level config cache (mirrors the pattern in src/audio_io.py)
# --------------------------------------------------------------------------- #

_CONFIG_CACHE: dict[str, Any] = {}
_CONFIG_CACHE_PATH: str | None = None

# Default fallback path, used only when neither an explicit path nor the
# VOICE_ASSISTANT_CONFIG env var is set. Resolved relative to project root
# (parent of the ``src`` directory containing this file).
_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "dev_config.yaml"
)

# Valid message roles (matches Ollama / OpenAI chat schema).
_VALID_ROLES = ("user", "assistant", "system")


def _get_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load and cache the YAML config.

    Resolution priority:
        1. Explicit ``config_path`` argument.
        2. ``VOICE_ASSISTANT_CONFIG`` environment variable.
        3. ``configs/dev_config.yaml`` next to the project root.

    The cache is keyed on the resolved path string; switching paths
    triggers a reload.

    Args:
        config_path: Optional explicit path override.

    Returns:
        Parsed YAML as a dict.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    global _CONFIG_CACHE, _CONFIG_CACHE_PATH

    if config_path is not None:
        resolved = str(Path(config_path).resolve())
    elif os.environ.get("VOICE_ASSISTANT_CONFIG"):
        resolved = str(Path(os.environ["VOICE_ASSISTANT_CONFIG"]).resolve())
    else:
        resolved = str(_DEFAULT_CONFIG_PATH)

    if _CONFIG_CACHE_PATH == resolved and _CONFIG_CACHE:
        return _CONFIG_CACHE

    path = Path(resolved)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise yaml.YAMLError(
            f"Config root must be a mapping, got {type(data).__name__}"
        )

    _CONFIG_CACHE = data
    _CONFIG_CACHE_PATH = resolved
    return _CONFIG_CACHE


def _conversation_cfg(key: str, default: Any) -> Any:
    """Fetch a single key from the ``conversation:`` config subsection.

    Args:
        key: Subkey under ``conversation:``.
        default: Returned when the subsection or key is absent.

    Returns:
        The configured value, or ``default``.
    """
    cfg = _CONFIG_CACHE.get("conversation", {}) if _CONFIG_CACHE else {}
    if not isinstance(cfg, dict):
        return default
    return cfg.get(key, default)


# --------------------------------------------------------------------------- #
# SQLite schema
# --------------------------------------------------------------------------- #

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id            TEXT PRIMARY KEY,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    system_prompt TEXT NOT NULL
)
"""

_SCHEMA_TURNS = """
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    timestamp   REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
)
"""

_SCHEMA_INDEX = """
CREATE INDEX IF NOT EXISTS idx_turns_session
    ON turns(session_id, id)
"""


# --------------------------------------------------------------------------- #
# ConversationManager
# --------------------------------------------------------------------------- #


class ConversationManager:
    """Multi-turn conversation state with rolling history and SQLite persistence.

    Each instance corresponds to one call session. The in-memory ``history``
    holds at most ``max_history_turns`` user/assistant pairs (i.e. ``2 *
    max_history_turns`` messages) and is what ``build_messages()`` returns
    to the LLM, prepended with the system prompt. Every turn — including
    those later evicted from the rolling window — is appended to SQLite
    immediately on ``add_user_turn`` / ``add_assistant_turn``.

    The class is safe to use as a context manager; ``__exit__`` calls
    :py:meth:`end_session` which stamps ``ended_at`` and closes the DB
    connection.

    Attributes:
        session_id: UUID4 string identifying this session.
        system_prompt: Resolved system prompt for this session.
        max_history_turns: Maximum (user, assistant) pairs retained in memory.
        db_path: Path to the SQLite database file.
        history: Current rolling window of message dicts.
    """

    def __init__(
        self,
        config_path: str | os.PathLike[str] | None = None,
        session_id: str | None = None,
        system_prompt: str | None = None,
        max_history_turns: int | None = None,
        db_path: str | os.PathLike[str] | None = None,
    ) -> None:
        """Initialise a conversation session.

        Args:
            config_path: Optional path to a YAML config file. Falls back to
                ``VOICE_ASSISTANT_CONFIG`` env var, then the project default.
            session_id: If provided, resume the existing session with this
                id (loads its history from SQLite). If None, a fresh session
                is created with a generated UUID4.
            system_prompt: Override for the configured system prompt. Ignored
                when resuming an existing session (the stored prompt wins).
            max_history_turns: Override for the configured rolling-window
                size. Must be > 0.
            db_path: Override for the configured SQLite path.

        Raises:
            FileNotFoundError: If ``config_path`` is given and missing.
            ValueError: If ``max_history_turns`` resolves to a non-positive
                value, or if ``session_id`` is supplied but not found in the
                database.
        """
        _get_config(config_path)

        # Resolve effective settings: explicit arg > config > sensible default.
        self.system_prompt: str = (
            system_prompt
            if system_prompt is not None
            else _conversation_cfg("system_prompt", "")
        )
        if not isinstance(self.system_prompt, str):
            raise ValueError(
                f"system_prompt must be str, got {type(self.system_prompt).__name__}"
            )

        resolved_max = (
            max_history_turns
            if max_history_turns is not None
            else _conversation_cfg("max_history_turns", 6)
        )
        if not isinstance(resolved_max, int) or resolved_max <= 0:
            raise ValueError(
                f"max_history_turns must be a positive int, got {resolved_max!r}"
            )
        self.max_history_turns: int = resolved_max

        resolved_db = (
            db_path
            if db_path is not None
            else _conversation_cfg("db_path", "recordings/conversations.db")
        )
        self.db_path: Path = Path(resolved_db).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection = sqlite3.connect(str(self.db_path))
        # WAL mode keeps reads non-blocking while a call is writing turns.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

        self.history: list[dict[str, Any]] = []

        if session_id is None:
            self.session_id: str = str(uuid.uuid4())
            self._create_session_row()
        else:
            self.session_id = session_id
            self._resume_session()

    # ------------------------------------------------------------------ #
    # Schema / row management
    # ------------------------------------------------------------------ #

    def _init_schema(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._conn:
            self._conn.execute(_SCHEMA_SESSIONS)
            self._conn.execute(_SCHEMA_TURNS)
            self._conn.execute(_SCHEMA_INDEX)

    def _create_session_row(self) -> None:
        """Insert the row for a brand-new session."""
        with self._conn:
            self._conn.execute(
                "INSERT INTO sessions (id, started_at, ended_at, system_prompt) "
                "VALUES (?, ?, NULL, ?)",
                (self.session_id, time.time(), self.system_prompt),
            )

    def _resume_session(self) -> None:
        """Load an existing session's prompt and rolling-window history.

        Only the most recent ``2 * max_history_turns`` turns are restored
        into memory; older turns remain in SQLite.

        Raises:
            ValueError: If the session id is not in the database.
        """
        row = self._conn.execute(
            "SELECT system_prompt FROM sessions WHERE id = ?",
            (self.session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown session_id: {self.session_id}")

        # The stored prompt wins on resume — keeps conversations coherent.
        self.system_prompt = row[0]

        window = 2 * self.max_history_turns
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM turns "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (self.session_id, window),
        ).fetchall()
        # rows came back newest-first; reverse to chronological order.
        self.history = [
            {"role": r[0], "content": r[1], "timestamp": r[2]} for r in reversed(rows)
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_user_turn(self, text: str) -> None:
        """Append a user message to history and persist it.

        Args:
            text: User utterance (post-ASR transcript). Must be a non-empty
                string after stripping whitespace.

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
            TypeError: If ``text`` is not a string.
        """
        self._add_turn("user", text)

    def add_assistant_turn(self, text: str) -> None:
        """Append an assistant message to history and persist it.

        Args:
            text: Full assistant reply (after streaming completes).

        Raises:
            ValueError: If ``text`` is empty or whitespace-only.
            TypeError: If ``text`` is not a string.
        """
        self._add_turn("assistant", text)

    def _add_turn(self, role: str, text: str) -> None:
        """Shared implementation for adding a turn of either role."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only")
        if role not in _VALID_ROLES:
            raise ValueError(f"role must be one of {_VALID_ROLES}, got {role!r}")

        ts = time.time()
        with self._conn:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, role, text, ts),
            )
        self.history.append({"role": role, "content": text, "timestamp": ts})
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop oldest user/assistant pairs until within the rolling window.

        Eviction is done in pairs (one user + one assistant) to preserve
        conversational coherence — never leave a dangling user message with
        no reply, or vice versa.
        """
        max_msgs = 2 * self.max_history_turns
        # Trim pairs at a time. If history is odd-length, the unpaired tail
        # is the most recent message and is preserved.
        while len(self.history) > max_msgs:
            del self.history[0:2]

    def build_messages(self) -> list[dict[str, str]]:
        """Build the message list to send to the LLM.

        The system prompt is prepended (when non-empty), followed by the
        current rolling-window history with ``role`` and ``content`` keys
        only — ``timestamp`` is stripped because Ollama / OpenAI chat
        endpoints do not accept it.

        Returns:
            List of ``{role, content}`` dicts suitable for
            ``ollama.chat(messages=...)``.
        """
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        for turn in self.history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        return messages

    def get_history(self) -> list[dict[str, Any]]:
        """Return a shallow copy of the in-memory rolling window.

        Returns:
            List of turn dicts, each ``{role, content, timestamp}``.
        """
        return [dict(turn) for turn in self.history]

    def get_full_history(self) -> list[dict[str, Any]]:
        """Return EVERY turn ever recorded for this session, from SQLite.

        Unlike :py:meth:`get_history`, this is not limited by the rolling
        window — useful for post-call review and WER evaluation.

        Returns:
            Chronologically ordered list of turn dicts.
        """
        rows = self._conn.execute(
            "SELECT role, content, timestamp FROM turns "
            "WHERE session_id = ? ORDER BY id ASC",
            (self.session_id,),
        ).fetchall()
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows
        ]

    def end_session(self) -> None:
        """Stamp ``ended_at`` on the session row and close the DB connection.

        Idempotent: safe to call multiple times. Subsequent calls on a
        closed instance will raise ``sqlite3.ProgrammingError`` if they
        touch the DB.
        """
        if self._conn is None:
            return
        try:
            with self._conn:
                self._conn.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                    (time.time(), self.session_id),
                )
        finally:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "ConversationManager":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.end_session()


# --------------------------------------------------------------------------- #
# CLI smoke test
# --------------------------------------------------------------------------- #


def _main() -> None:  # pragma: no cover - manual smoke test
    """Tiny smoke test: create a session, log a couple of turns, print state."""
    with ConversationManager() as cm:
        print(f"Session id:     {cm.session_id}")
        print(f"System prompt:  {cm.system_prompt[:80]}...")
        print(f"Max turn pairs: {cm.max_history_turns}")
        print(f"DB path:        {cm.db_path}")
        cm.add_user_turn("Hello, who is this?")
        cm.add_assistant_turn("Hi, this is your voice assistant. How can I help?")
        cm.add_user_turn("What time is it?")
        cm.add_assistant_turn("I'm not connected to a clock, sorry.")
        print("\nbuild_messages() output:")
        for m in cm.build_messages():
            print(f"  [{m['role']:9s}] {m['content']}")


if __name__ == "__main__":  # pragma: no cover
    _main()