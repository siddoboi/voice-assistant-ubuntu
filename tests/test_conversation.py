"""Unit tests for src/conversation.py — ConversationManager.

Follows the existing test architecture (Section 10 of the master context):
class-grouped tests, ``tmp_path`` for isolation, no real network/disk
dependencies beyond a temp SQLite file. SQLite is stdlib and deterministic,
so every test here is a fast unit test — no integration marker needed.

Each test creates its own tmp config + DB so module-level config caching
in ``src/conversation`` cannot leak state between tests.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from src import conversation
from src.conversation import ConversationManager


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_config_cache() -> None:
    """Clear the module-level config cache before every test."""
    conversation._CONFIG_CACHE = {}
    conversation._CONFIG_CACHE_PATH = None


def _write_config(
    tmp_path: Path,
    *,
    system_prompt: str = "You are a test assistant. Reply briefly.",
    max_history_turns: int = 3,
    db_filename: str = "conversations.db",
    include_conversation: bool = True,
) -> Path:
    """Write a minimal config file under tmp_path and return its path."""
    db_path = tmp_path / db_filename
    cfg: dict[str, Any] = {
        "audio": {"sample_rate": 16000, "channels": 1, "dtype": "int16"},
        "paths": {"recordings_dir": str(tmp_path)},
    }
    if include_conversation:
        cfg["conversation"] = {
            "system_prompt": system_prompt,
            "max_history_turns": max_history_turns,
            "db_path": str(db_path),
        }
    cfg_path = tmp_path / "dev_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg_path


@pytest.fixture
def cfg(tmp_path: Path) -> Path:
    """Default config: 3-turn rolling window, isolated DB under tmp_path."""
    return _write_config(tmp_path)


# --------------------------------------------------------------------------- #
# ConfigLoading
# --------------------------------------------------------------------------- #


class TestConfigLoading:
    """Resolution priority: explicit path > env var > project default."""

    def test_explicit_path_used(self, cfg: Path) -> None:
        loaded = conversation._get_config(cfg)
        assert loaded["conversation"]["max_history_turns"] == 3

    def test_env_var_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg_path = _write_config(tmp_path, max_history_turns=5)
        monkeypatch.setenv("VOICE_ASSISTANT_CONFIG", str(cfg_path))
        loaded = conversation._get_config()
        assert loaded["conversation"]["max_history_turns"] == 5

    def test_explicit_beats_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_dir = tmp_path / "env"
        env_dir.mkdir()
        explicit_dir = tmp_path / "explicit"
        explicit_dir.mkdir()
        env_cfg = _write_config(env_dir, max_history_turns=99, db_filename="env.db")
        explicit_cfg = _write_config(
            explicit_dir, max_history_turns=4, db_filename="explicit.db"
        )
        monkeypatch.setenv("VOICE_ASSISTANT_CONFIG", str(env_cfg))
        loaded = conversation._get_config(explicit_cfg)
        assert loaded["conversation"]["max_history_turns"] == 4

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            conversation._get_config(tmp_path / "does_not_exist.yaml")

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not: valid: yaml: [", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            conversation._get_config(bad)

    def test_non_mapping_root_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            conversation._get_config(bad)

    def test_cache_reused_for_same_path(self, cfg: Path) -> None:
        first = conversation._get_config(cfg)
        second = conversation._get_config(cfg)
        assert first is second  # identity — cache hit

    def test_cache_reloads_on_path_change(self, tmp_path: Path) -> None:
        d1 = tmp_path / "d1"
        d1.mkdir()
        d2 = tmp_path / "d2"
        d2.mkdir()
        a = _write_config(d1, max_history_turns=2, db_filename="a.db")
        b = _write_config(d2, max_history_turns=7, db_filename="b.db")
        loaded_a = conversation._get_config(a)
        loaded_b = conversation._get_config(b)
        assert loaded_a["conversation"]["max_history_turns"] == 2
        assert loaded_b["conversation"]["max_history_turns"] == 7


# --------------------------------------------------------------------------- #
# Init
# --------------------------------------------------------------------------- #


class TestInit:
    def test_creates_session_with_uuid(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            assert isinstance(cm.session_id, str)
            assert len(cm.session_id) == 36  # UUID4 with hyphens
            assert cm.session_id.count("-") == 4

    def test_loads_system_prompt_from_config(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            assert "test assistant" in cm.system_prompt

    def test_loads_max_history_from_config(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            assert cm.max_history_turns == 3

    def test_db_path_from_config(self, tmp_path: Path, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            assert cm.db_path.exists()
            assert cm.db_path.parent == tmp_path.resolve()

    def test_explicit_system_prompt_override(self, cfg: Path) -> None:
        with ConversationManager(
            config_path=cfg, system_prompt="Custom override prompt."
        ) as cm:
            assert cm.system_prompt == "Custom override prompt."

    def test_explicit_max_history_override(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg, max_history_turns=10) as cm:
            assert cm.max_history_turns == 10

    def test_explicit_db_path_override(self, tmp_path: Path, cfg: Path) -> None:
        custom_db = tmp_path / "custom.db"
        with ConversationManager(config_path=cfg, db_path=custom_db) as cm:
            assert cm.db_path == custom_db.resolve()
            assert custom_db.exists()

    def test_creates_db_parent_dir(self, tmp_path: Path, cfg: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "conv.db"
        with ConversationManager(config_path=cfg, db_path=nested) as cm:
            assert nested.exists()
            _ = cm  # silence unused-var

    def test_zero_max_history_rejected(self, cfg: Path) -> None:
        with pytest.raises(ValueError, match="positive int"):
            ConversationManager(config_path=cfg, max_history_turns=0)

    def test_negative_max_history_rejected(self, cfg: Path) -> None:
        with pytest.raises(ValueError, match="positive int"):
            ConversationManager(config_path=cfg, max_history_turns=-1)

    def test_non_int_max_history_rejected(self, cfg: Path) -> None:
        with pytest.raises(ValueError, match="positive int"):
            ConversationManager(config_path=cfg, max_history_turns="3")  # type: ignore[arg-type]

    def test_non_string_system_prompt_rejected(self, cfg: Path) -> None:
        with pytest.raises(ValueError, match="system_prompt must be str"):
            ConversationManager(config_path=cfg, system_prompt=123)  # type: ignore[arg-type]

    def test_missing_conversation_section_uses_defaults(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path, include_conversation=False)
        with ConversationManager(
            config_path=cfg_path, db_path=tmp_path / "default.db"
        ) as cm:
            assert cm.max_history_turns == 6  # documented default
            assert cm.system_prompt == ""  # documented default


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_tables_created(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            tables = {
                row[0]
                for row in cm._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "sessions" in tables
            assert "turns" in tables

    def test_session_row_inserted_on_create(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            row = cm._conn.execute(
                "SELECT id, started_at, ended_at, system_prompt FROM sessions WHERE id = ?",
                (cm.session_id,),
            ).fetchone()
            assert row is not None
            assert row[0] == cm.session_id
            assert isinstance(row[1], float)  # started_at
            assert row[2] is None  # ended_at not set yet
            assert row[3] == cm.system_prompt

    def test_index_on_turns_created(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            indexes = {
                row[0]
                for row in cm._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            assert "idx_turns_session" in indexes


# --------------------------------------------------------------------------- #
# AddTurn
# --------------------------------------------------------------------------- #


class TestAddTurn:
    def test_add_user_appends_to_history(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("hello")
            assert len(cm.history) == 1
            assert cm.history[0]["role"] == "user"
            assert cm.history[0]["content"] == "hello"
            assert isinstance(cm.history[0]["timestamp"], float)

    def test_add_assistant_appends_to_history(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_assistant_turn("hi there")
            assert cm.history[-1] == {
                "role": "assistant",
                "content": "hi there",
                "timestamp": cm.history[-1]["timestamp"],
            }

    def test_turn_persisted_to_db(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("ping")
            row = cm._conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ?",
                (cm.session_id,),
            ).fetchone()
            assert row == ("user", "ping")

    def test_empty_text_rejected(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            with pytest.raises(ValueError, match="empty"):
                cm.add_user_turn("")

    def test_whitespace_only_rejected(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            with pytest.raises(ValueError, match="empty"):
                cm.add_assistant_turn("   \t\n  ")

    def test_non_string_text_rejected(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            with pytest.raises(TypeError, match="text must be str"):
                cm.add_user_turn(42)  # type: ignore[arg-type]

    def test_timestamps_monotonic(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("a")
            time.sleep(0.001)
            cm.add_assistant_turn("b")
            assert cm.history[1]["timestamp"] >= cm.history[0]["timestamp"]


# --------------------------------------------------------------------------- #
# RollingWindow
# --------------------------------------------------------------------------- #


class TestRollingWindow:
    """Window holds up to ``2 * max_history_turns`` messages. Evicts in pairs."""

    def test_under_capacity_keeps_all(self, cfg: Path) -> None:
        # cfg: max_history_turns=3 → 6-message capacity
        with ConversationManager(config_path=cfg) as cm:
            for i in range(2):  # 2 pairs = 4 messages
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            assert len(cm.history) == 4

    def test_at_capacity_keeps_all(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            for i in range(3):  # exactly 3 pairs = 6 messages
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            assert len(cm.history) == 6

    def test_over_capacity_evicts_oldest_pair(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            for i in range(4):  # 4 pairs = 8 messages, exceeds 6 cap
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            assert len(cm.history) == 6
            # Oldest pair (q0, a0) evicted; q1/a1 now the head.
            assert cm.history[0]["content"] == "q1"
            assert cm.history[1]["content"] == "a1"
            assert cm.history[-1]["content"] == "a3"

    def test_far_over_capacity_trims_correctly(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            for i in range(10):
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            assert len(cm.history) == 6
            assert cm.history[0]["content"] == "q7"
            assert cm.history[-1]["content"] == "a9"

    def test_eviction_does_not_affect_sqlite(self, cfg: Path) -> None:
        """Evicted turns must remain queryable from SQLite."""
        with ConversationManager(config_path=cfg) as cm:
            for i in range(5):
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            full = cm.get_full_history()
            assert len(full) == 10  # all 5 pairs preserved on disk
            assert full[0]["content"] == "q0"
            assert full[-1]["content"] == "a4"


# --------------------------------------------------------------------------- #
# BuildMessages
# --------------------------------------------------------------------------- #


class TestBuildMessages:
    def test_system_prompt_first(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("hi")
            msgs = cm.build_messages()
            assert msgs[0]["role"] == "system"
            assert "test assistant" in msgs[0]["content"]

    def test_history_order_preserved(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("one")
            cm.add_assistant_turn("two")
            cm.add_user_turn("three")
            msgs = cm.build_messages()
            assert [m["content"] for m in msgs[1:]] == ["one", "two", "three"]
            assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user"]

    def test_only_role_and_content_keys(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("hi")
            msgs = cm.build_messages()
            # Every message must have exactly {role, content} — Ollama rejects
            # extra keys like our internal 'timestamp'.
            for m in msgs:
                assert set(m.keys()) == {"role", "content"}

    def test_empty_history_returns_only_system(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            msgs = cm.build_messages()
            assert len(msgs) == 1
            assert msgs[0]["role"] == "system"

    def test_empty_system_prompt_omitted(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg, system_prompt="") as cm:
            cm.add_user_turn("hi")
            msgs = cm.build_messages()
            assert all(m["role"] != "system" for m in msgs)
            assert len(msgs) == 1

    def test_after_eviction_system_still_first(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            for i in range(8):
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            msgs = cm.build_messages()
            assert msgs[0]["role"] == "system"
            assert len(msgs) == 1 + 6  # system + 6 rolling messages


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


class TestResume:
    def test_resume_loads_prior_turns(self, cfg: Path) -> None:
        cm1 = ConversationManager(config_path=cfg)
        cm1.add_user_turn("hello")
        cm1.add_assistant_turn("hi")
        sid = cm1.session_id
        cm1.end_session()

        cm2 = ConversationManager(config_path=cfg, session_id=sid)
        try:
            assert cm2.session_id == sid
            assert len(cm2.history) == 2
            assert cm2.history[0]["content"] == "hello"
            assert cm2.history[1]["content"] == "hi"
        finally:
            cm2.end_session()

    def test_resume_uses_stored_system_prompt(self, cfg: Path) -> None:
        """On resume the stored prompt wins, even if config changed since."""
        cm1 = ConversationManager(
            config_path=cfg, system_prompt="Original prompt for session."
        )
        sid = cm1.session_id
        cm1.add_user_turn("x")
        cm1.end_session()

        # Pass a DIFFERENT system_prompt override on resume — should be ignored.
        cm2 = ConversationManager(
            config_path=cfg,
            session_id=sid,
            system_prompt="A totally new prompt that should be ignored.",
        )
        try:
            assert cm2.system_prompt == "Original prompt for session."
        finally:
            cm2.end_session()

    def test_resume_respects_rolling_window_size(self, cfg: Path) -> None:
        """If 20 turns exist on disk and window=3, only most recent 6 loaded."""
        cm1 = ConversationManager(config_path=cfg)
        sid = cm1.session_id
        for i in range(10):
            cm1.add_user_turn(f"q{i}")
            cm1.add_assistant_turn(f"a{i}")
        cm1.end_session()

        cm2 = ConversationManager(config_path=cfg, session_id=sid)
        try:
            assert len(cm2.history) == 6  # 2 * max_history_turns
            assert cm2.history[0]["content"] == "q7"
            assert cm2.history[-1]["content"] == "a9"
        finally:
            cm2.end_session()

    def test_unknown_session_id_raises(self, cfg: Path) -> None:
        with pytest.raises(ValueError, match="Unknown session_id"):
            ConversationManager(config_path=cfg, session_id="nonexistent-uuid")


# --------------------------------------------------------------------------- #
# EndSession
# --------------------------------------------------------------------------- #


class TestEndSession:
    def test_sets_ended_at(self, cfg: Path) -> None:
        cm = ConversationManager(config_path=cfg)
        sid = cm.session_id
        db = cm.db_path
        cm.end_session()
        # Reopen via raw sqlite to inspect.
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
        assert row[0] is not None
        assert isinstance(row[0], float)

    def test_idempotent(self, cfg: Path) -> None:
        cm = ConversationManager(config_path=cfg)
        cm.end_session()
        cm.end_session()  # must not raise

    def test_context_manager_calls_end(self, cfg: Path) -> None:
        sid: str
        db: Path
        with ConversationManager(config_path=cfg) as cm:
            sid = cm.session_id
            db = cm.db_path
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
        assert row[0] is not None

    def test_context_manager_ends_on_exception(self, cfg: Path) -> None:
        sid: str
        db: Path
        with pytest.raises(RuntimeError, match="boom"):
            with ConversationManager(config_path=cfg) as cm:
                sid = cm.session_id
                db = cm.db_path
                cm.add_user_turn("before crash")
                raise RuntimeError("boom")
        # Even after exception, ended_at should be set and the turn persisted.
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            assert row[0] is not None
            turn = conn.execute(
                "SELECT content FROM turns WHERE session_id = ?", (sid,)
            ).fetchone()
            assert turn[0] == "before crash"


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


class TestPersistence:
    """Turns survive process restart — critical for post-call analysis."""

    def test_turns_survive_close_and_reopen(self, cfg: Path, tmp_path: Path) -> None:
        cm = ConversationManager(config_path=cfg)
        sid = cm.session_id
        cm.add_user_turn("persistent message")
        cm.add_assistant_turn("durable reply")
        cm.end_session()
        # Open a fresh connection — simulates process restart.
        with sqlite3.connect(str(cm.db_path)) as conn:
            rows = conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
                (sid,),
            ).fetchall()
        assert rows == [
            ("user", "persistent message"),
            ("assistant", "durable reply"),
        ]

    def test_multiple_sessions_isolated(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as a:
            with ConversationManager(config_path=cfg) as b:
                assert a.session_id != b.session_id
                a.add_user_turn("from a")
                b.add_user_turn("from b")
                assert a.get_full_history()[0]["content"] == "from a"
                assert b.get_full_history()[0]["content"] == "from b"
                assert len(a.get_full_history()) == 1
                assert len(b.get_full_history()) == 1


# --------------------------------------------------------------------------- #
# GetHistory
# --------------------------------------------------------------------------- #


class TestGetHistory:
    def test_returns_copy_not_reference(self, cfg: Path) -> None:
        """Caller-side mutation must not affect internal state."""
        with ConversationManager(config_path=cfg) as cm:
            cm.add_user_turn("hi")
            snapshot = cm.get_history()
            snapshot[0]["content"] = "MUTATED"
            assert cm.history[0]["content"] == "hi"

    def test_get_full_history_chronological(self, cfg: Path) -> None:
        with ConversationManager(config_path=cfg) as cm:
            for i in range(5):
                cm.add_user_turn(f"q{i}")
                cm.add_assistant_turn(f"a{i}")
            full = cm.get_full_history()
            contents = [t["content"] for t in full]
            assert contents == [
                "q0", "a0", "q1", "a1", "q2", "a2", "q3", "a3", "q4", "a4"
            ]