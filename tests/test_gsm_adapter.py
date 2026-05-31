"""
test_gsm_adapter.py — Unit tests for src/telephony/gsm_adapter.py.

Every test runs against a FakeSerial that scripts the bytes the SIM7600EI
would return, so the whole adapter is exercised on WSL2 with no HAT attached.
`serial.Serial` is patched to hand back the FakeSerial.

Real-hardware validation (a live SIM, real RING/answer/hangup) happens on the
Pi in Week 4 and is intentionally not attempted here.

Run:
    pytest tests/test_gsm_adapter.py -v
"""

from __future__ import annotations

from collections import deque
from unittest.mock import patch

import pytest

from src.telephony import gsm_adapter
from src.telephony.gsm_adapter import (
    GSMAdapter,
    GSMCommandError,
    GSMConnectionError,
    GSMTimeout,
)


# --------------------------------------------------------------------------- #
# FakeSerial — scripts modem responses
# --------------------------------------------------------------------------- #


class FakeSerial:
    """Minimal stand-in for serial.Serial.

    `script` is a list of response lines (str, without terminators) that
    readline() will hand back one at a time as bytes. When the script is
    exhausted, readline() returns b"" (the pyserial timeout behaviour),
    which the adapter treats as "no data this tick".
    """

    def __init__(self, script: list[str] | None = None, *, raise_on_open: bool = False):
        if raise_on_open:
            raise gsm_adapter.serial.SerialException("port busy")
        self.is_open = True
        self._lines: deque[str] = deque(script or [])
        self.written: list[bytes] = []
        self.reset_count = 0

    # --- pyserial surface the adapter uses ---
    def reset_input_buffer(self) -> None:
        self.reset_count += 1

    def write(self, data: bytes) -> int:
        self.written.append(data)
        return len(data)

    def readline(self) -> bytes:
        if self._lines:
            return (self._lines.popleft() + "\r\n").encode("ascii")
        return b""

    def close(self) -> None:
        self.is_open = False

    # --- test helpers ---
    def queue(self, *lines: str) -> None:
        self._lines.extend(lines)

    def written_text(self) -> list[str]:
        return [w.decode("ascii").strip() for w in self.written]


def _patched(script: list[str] | None = None):
    """Context manager: patch serial.Serial to return a FakeSerial(script)."""
    fake = FakeSerial(script if script is not None else ["OK", "OK", "OK"])
    return patch.object(gsm_adapter.serial, "Serial", return_value=fake), fake


def _connected_adapter(extra_script: list[str] | None = None):
    """Build a connected adapter; returns (adapter, fake_serial).

    connect() consumes three OKs (AT, ATE0, AT+CLIP=1); extra_script lines are
    queued *after* those for the test's own command.
    """
    script = ["OK", "OK", "OK"] + (extra_script or [])
    fake = FakeSerial(script)
    with patch.object(gsm_adapter.serial, "Serial", return_value=fake):
        adapter = GSMAdapter(port="/dev/ttyTEST", timeout_s=0.2, ring_poll_timeout_s=0.2)
        adapter.connect()
    return adapter, fake


# --------------------------------------------------------------------------- #
# Construction / config
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_defaults(self):
        a = GSMAdapter(config_path="/nonexistent.yaml")
        assert a.port == "/dev/ttyUSB2"
        assert a.baudrate == 115200
        assert a.timeout_s == 2.0
        assert a.ring_poll_timeout_s == 30.0

    def test_explicit_overrides(self):
        a = GSMAdapter(
            port="/dev/ttyUSB9", baudrate=9600, timeout_s=1.0,
            ring_poll_timeout_s=5.0, config_path="/nonexistent.yaml",
        )
        assert a.port == "/dev/ttyUSB9"
        assert a.baudrate == 9600
        assert a.timeout_s == 1.0
        assert a.ring_poll_timeout_s == 5.0

    def test_reads_config_section(self, tmp_path):
        import yaml
        cfg = tmp_path / "dev_config.yaml"
        cfg.write_text(yaml.safe_dump({
            "telephony": {"port": "/dev/ttyUSB2", "baudrate": 57600, "timeout_s": 3.0}
        }))
        a = GSMAdapter(config_path=str(cfg))
        assert a.baudrate == 57600
        assert a.timeout_s == 3.0

    def test_not_connected_before_connect(self):
        a = GSMAdapter(config_path="/nonexistent.yaml")
        assert a.is_connected is False


# --------------------------------------------------------------------------- #
# Connect / disconnect
# --------------------------------------------------------------------------- #


class TestConnect:
    def test_connect_opens_with_config_params(self):
        fake = FakeSerial(["OK", "OK", "OK"])
        with patch.object(gsm_adapter.serial, "Serial", return_value=fake) as ctor:
            a = GSMAdapter(port="/dev/ttyTEST", baudrate=115200, timeout_s=1.5)
            a.connect()
        ctor.assert_called_once_with("/dev/ttyTEST", 115200, timeout=1.5)
        assert a.is_connected is True

    def test_connect_sends_init_sequence(self):
        a, fake = _connected_adapter()
        # AT, ATE0, AT+CLIP=1 in order.
        assert fake.written_text() == ["AT", "ATE0", "AT+CLIP=1"]

    def test_connect_raises_when_port_unavailable(self):
        def boom(*a, **k):
            raise gsm_adapter.serial.SerialException("no such device")
        with patch.object(gsm_adapter.serial, "Serial", side_effect=boom):
            a = GSMAdapter(port="/dev/ttyNOPE")
            with pytest.raises(GSMConnectionError, match="Could not open"):
                a.connect()
        assert a.is_connected is False

    def test_connect_raises_when_module_silent(self):
        # Port opens but AT never returns OK → init fails → connection error.
        fake = FakeSerial([])  # no responses
        with patch.object(gsm_adapter.serial, "Serial", return_value=fake):
            a = GSMAdapter(port="/dev/ttyTEST", timeout_s=0.1)
            with pytest.raises(GSMConnectionError, match="init failed"):
                a.connect()
        assert a.is_connected is False

    def test_disconnect_closes_port(self):
        a, fake = _connected_adapter()
        a.disconnect()
        assert a.is_connected is False
        assert fake.is_open is False

    def test_disconnect_idempotent(self):
        a, _ = _connected_adapter()
        a.disconnect()
        a.disconnect()  # must not raise


# --------------------------------------------------------------------------- #
# send_at
# --------------------------------------------------------------------------- #


class TestSendAt:
    def test_writes_command_with_terminator(self):
        a, fake = _connected_adapter(["OK"])
        a.send_at("AT+CSQ")
        assert fake.written[-1] == b"AT+CSQ\r\n"

    def test_returns_intermediate_lines(self):
        a, _ = _connected_adapter(["+CSQ: 20,99", "OK"])
        lines = a.send_at("AT+CSQ")
        assert lines == ["+CSQ: 20,99"]

    def test_ok_only_returns_empty_list(self):
        a, _ = _connected_adapter(["OK"])
        assert a.send_at("ATE0") == []

    def test_raises_on_error(self):
        a, _ = _connected_adapter(["ERROR"])
        with pytest.raises(GSMCommandError, match="failed: ERROR"):
            a.send_at("AT+BAD")

    def test_raises_on_cme_error(self):
        a, _ = _connected_adapter(["+CME ERROR: 10"])
        with pytest.raises(GSMCommandError, match="CME ERROR"):
            a.send_at("AT+CPIN?")

    def test_raises_on_busy(self):
        a, _ = _connected_adapter(["BUSY"])
        with pytest.raises(GSMCommandError, match="BUSY"):
            a.send_at("ATD123;")

    def test_timeout_when_no_final_code(self):
        a, _ = _connected_adapter([])  # nothing after connect
        with pytest.raises(GSMTimeout, match="timed out"):
            a.send_at("AT+CSQ", timeout=0.1)

    def test_skips_command_echo(self):
        # Echo on: module repeats the command before the real response.
        a, _ = _connected_adapter(["AT+CSQ", "+CSQ: 15,99", "OK"])
        lines = a.send_at("AT+CSQ")
        assert lines == ["+CSQ: 15,99"]

    def test_resets_input_before_write(self):
        a, fake = _connected_adapter(["OK"])
        before = fake.reset_count
        a.send_at("AT")
        assert fake.reset_count == before + 1

    def test_raises_when_not_connected(self):
        a = GSMAdapter(config_path="/nonexistent.yaml")
        with pytest.raises(GSMConnectionError, match="not connected"):
            a.send_at("AT")


# --------------------------------------------------------------------------- #
# Status queries
# --------------------------------------------------------------------------- #


class TestStatusQueries:
    def test_check_sim_ready(self):
        a, _ = _connected_adapter(["+CPIN: READY", "OK"])
        assert a.check_sim() is True

    def test_check_sim_not_ready(self):
        a, _ = _connected_adapter(["+CPIN: SIM PIN", "OK"])
        assert a.check_sim() is False

    def test_check_signal_parses_rssi(self):
        a, _ = _connected_adapter(["+CSQ: 24,99", "OK"])
        assert a.check_signal() == 24

    def test_check_signal_unknown_when_absent(self):
        a, _ = _connected_adapter(["OK"])
        assert a.check_signal() == 99

    def test_check_registration_home(self):
        a, _ = _connected_adapter(["+CREG: 0,1", "OK"])
        assert a.check_registration() is True

    def test_check_registration_roaming(self):
        a, _ = _connected_adapter(["+CREG: 0,5", "OK"])
        assert a.check_registration() is True

    def test_check_registration_not_registered(self):
        a, _ = _connected_adapter(["+CREG: 0,0", "OK"])
        assert a.check_registration() is False

    def test_is_call_active_true(self):
        a, _ = _connected_adapter(["+CLCC: 1,1,0,0,0,\"+12025550100\",129", "OK"])
        assert a.is_call_active() is True

    def test_is_call_active_false(self):
        a, _ = _connected_adapter(["OK"])
        assert a.is_call_active() is False


# --------------------------------------------------------------------------- #
# Call control
# --------------------------------------------------------------------------- #


class TestCallControl:
    def test_answer_sends_ata(self):
        a, fake = _connected_adapter(["OK"])
        a.answer_call()
        assert fake.written[-1] == b"ATA\r\n"

    def test_hangup_sends_chup(self):
        a, fake = _connected_adapter(["OK"])
        a.hangup()
        assert fake.written[-1] == b"AT+CHUP\r\n"

    def test_dial_sends_atd_with_semicolon(self):
        a, fake = _connected_adapter(["OK"])
        a.dial("+12025550100")
        assert fake.written[-1] == b"ATD+12025550100;\r\n"

    def test_dial_strips_whitespace(self):
        a, fake = _connected_adapter(["OK"])
        a.dial("  5551234  ")
        assert fake.written[-1] == b"ATD5551234;\r\n"

    def test_dial_rejects_empty(self):
        a, _ = _connected_adapter(["OK"])
        with pytest.raises(ValueError, match="non-empty"):
            a.dial("   ")


# --------------------------------------------------------------------------- #
# Ring detection
# --------------------------------------------------------------------------- #


class TestWaitForRing:
    def test_detects_ring(self):
        a, _ = _connected_adapter(["RING"])
        event = a.wait_for_ring(timeout=0.2)
        assert event == {"event": "RING", "caller": None}

    def test_captures_caller_from_clip(self):
        a, _ = _connected_adapter(['+CLIP: "+12025550100",145,,,,0', "RING"])
        event = a.wait_for_ring(timeout=0.2)
        assert event["event"] == "RING"
        assert event["caller"] == "+12025550100"

    def test_returns_none_on_timeout(self):
        a, _ = _connected_adapter([])  # no RING ever
        assert a.wait_for_ring(timeout=0.1) is None

    def test_raises_when_not_connected(self):
        a = GSMAdapter(config_path="/nonexistent.yaml")
        with pytest.raises(GSMConnectionError, match="not connected"):
            a.wait_for_ring(timeout=0.1)


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #


class TestContextManager:
    def test_enter_connects_exit_disconnects(self):
        fake = FakeSerial(["OK", "OK", "OK"])
        with patch.object(gsm_adapter.serial, "Serial", return_value=fake):
            with GSMAdapter(port="/dev/ttyTEST", timeout_s=0.2) as a:
                assert a.is_connected is True
            assert a.is_connected is False
        assert fake.is_open is False

    def test_exit_disconnects_on_exception(self):
        fake = FakeSerial(["OK", "OK", "OK"])
        with patch.object(gsm_adapter.serial, "Serial", return_value=fake):
            with pytest.raises(RuntimeError, match="boom"):
                with GSMAdapter(port="/dev/ttyTEST", timeout_s=0.2) as a:
                    raise RuntimeError("boom")
            assert a.is_connected is False


# --------------------------------------------------------------------------- #
# Parsers (direct unit tests)
# --------------------------------------------------------------------------- #


class TestParsers:
    def test_parse_csq(self):
        assert gsm_adapter._parse_csq("+CSQ: 20,99") == 20

    def test_parse_csq_bad(self):
        assert gsm_adapter._parse_csq("+CSQ: garbage") == 99

    def test_parse_creg(self):
        assert gsm_adapter._parse_creg("+CREG: 0,1") == 1

    def test_parse_creg_bad(self):
        assert gsm_adapter._parse_creg("+CREG: nonsense") == -1

    def test_parse_clip(self):
        assert gsm_adapter._parse_clip('+CLIP: "+12025550100",145,,,,0') == "+12025550100"

    def test_parse_clip_empty_number(self):
        assert gsm_adapter._parse_clip('+CLIP: "",128') is None