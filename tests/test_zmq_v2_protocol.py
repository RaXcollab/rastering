"""Rastering ZMQ v2 protocol roundtrip via InMemoryTransport.

Exercises the REAL ``_RasteringV2Server`` (RemoteControlServerBase
subclass) in ``raster_controller.py``. No sockets bound; tests pair
two ``InMemoryTransport`` instances so the dispatcher path runs
end-to-end with real envelope encode/parse.

Pins:
  * HELLO reply: status SUCCESS, protocol_version 2,
    capabilities = {monitors, heartbeat}, NO ``connections`` key.
  * v1 hard sunset: missing ``v`` -> v1_protocol_refused.
  * id echo on every reply.
  * PROGRAM_VALUE for coord channels delegates to request_move_x/y.
  * PROGRAM_VALUE arm_raster returns SUCCESS + extra.mode.
  * PROGRAM_VALUE move_to_next end-of-iter -> SUCCESS + extra.finished.
  * CHECK_VALUE returns cached target XY when present, else motor XY.
  * timeout_sec moves into args dict (Q2 §10-resolved).

Run:
    conda activate rastering && pytest tests/test_zmq_v2_protocol.py -v
"""
from __future__ import annotations

import json
import os
import sys
import threading
from unittest import mock

import pytest

# raster_controller.py lives one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import raster_controller as rc  # noqa: E402
    _IMPORT_ERR = None
except Exception as e:  # noqa: BLE001
    rc = None  # type: ignore
    _IMPORT_ERR = e


def _require_rc():
    if _IMPORT_ERR is not None:
        pytest.skip(
            "raster_controller not importable (rastering env?): " + repr(_IMPORT_ERR))


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def zmq_v2():
    _require_rc()
    pytest.importorskip("zmq_v2")
    import zmq_v2  # noqa: PLC0415
    return zmq_v2


def _make_outer(*, target_xy=None, motor_xy=None,
                raster_active=False, raster_has_path=False,
                raster_continuous=False, move_x_ok=True, move_y_ok=True,
                step_returns=None, raster_step_calls=None):
    """Stand-in for SystemController; duck-typed to what _RasteringV2Server
    actually reads/calls."""
    _require_rc()
    outer = mock.MagicMock()
    outer._state_lock = threading.RLock()
    outer._last_target_xy = target_xy
    outer._last_motor_xy = motor_xy
    outer._raster_active = raster_active
    outer._raster_continuous = raster_continuous
    # Real controller uses an indexed point list (_raster_path_pts), not a
    # one-shot generator; a non-empty list means "raster configured".
    outer._raster_path_pts = [(0.0, 0.0)] if raster_has_path else []

    def _move_ok(value, *, source, wait, timeout_s):
        res = mock.MagicMock()
        res.ok = bool(value)  # value=0 -> fail; non-zero -> ok per test seed
        res.message = "" if res.ok else "rejected"
        return res

    def _make_move_factory(success):
        def _move(value, *, source, wait, timeout_s):
            res = mock.MagicMock()
            res.ok = success
            res.message = "" if success else "motor rejected"
            return res
        return _move

    outer.request_move_x.side_effect = _make_move_factory(move_x_ok)
    outer.request_move_y.side_effect = _make_move_factory(move_y_ok)
    outer.raster_step.side_effect = (
        step_returns if step_returns is not None
        else (lambda **kw: (mock.MagicMock(ok=True, message=""))))
    return outer


def _roundtrip(client_t, v2_server, envelope_dict):
    client_t.send(json.dumps(envelope_dict).encode("utf-8"))
    served = v2_server.serve_once(timeout_ms=100)
    assert served is True
    return json.loads(client_t.recv(timeout_ms=100).decode("utf-8"))


@pytest.fixture
def make_v2_pair(zmq_v2):
    def _factory(**kwargs):
        outer = _make_outer(**kwargs)
        client_t, server_t = zmq_v2.InMemoryTransport.pair()
        v2_server = rc._RasteringV2Server(outer, server_t)
        return outer, client_t, v2_server
    return _factory


# ---------------------------------------------------------------- tests


def test_v2_hello_single_instance_no_connections_key(zmq_v2, make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server,
                       {"v": 2, "id": 1, "action": "HELLO"})
    assert reply["status"] == "SUCCESS"
    assert reply["id"] == 1
    assert reply["server"] == "RasteringGUI"
    assert set(reply["capabilities"]) == {"monitors", "heartbeat"}
    assert "connections" not in reply


def test_v2_v1_envelope_refused(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {"action": "HELLO"})
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "v1_protocol_refused"


def test_v2_program_value_x_delegates_to_request_move_x(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(move_x_ok=True)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 7, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_x_coord", "value": 12.5,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["id"] == 7
    outer.request_move_x.assert_called_once()
    args, kwargs = outer.request_move_x.call_args
    assert args[0] == 12.5
    assert kwargs["source"] == "zmq"
    assert kwargs["wait"] is True


def test_v2_program_value_motor_failure_returns_retryable(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(move_x_ok=False)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 8, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_x_coord", "value": 12.5,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "motor_move_failed"
    assert reply["error"]["retryable"] is True


def test_v2_program_value_timeout_sec_moves_into_args(make_v2_pair):
    """Q2 §10-resolved: per-request extras live in args, not top-level."""
    outer, client_t, v2_server = make_v2_pair(move_x_ok=True)
    _roundtrip(client_t, v2_server, {
        "v": 2, "id": 9, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_x_coord", "value": 5.0,
        "args": {"timeout_sec": 30.0},
    })
    _, kwargs = outer.request_move_x.call_args
    assert kwargs["timeout_s"] == 30.0


def test_v2_program_value_arm_raster_continuous_returns_extra_mode(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=True, raster_has_path=True)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 10, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["mode"] == "continuous"
    outer._enqueue_next_raster_point.assert_called_once()


def test_v2_program_value_arm_raster_step_mode(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=True, raster_has_path=True)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 11, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["mode"] == "step"
    outer._enqueue_next_raster_point.assert_not_called()


def test_v2_program_value_arm_raster_without_config_rejected(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 12, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 1,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "no_raster_configured"


def test_v2_program_value_move_to_next_iter_end_returns_finished_extra(make_v2_pair):
    """Iterator exhaustion: v1 used non-spec status "FINISHED"; v2 maps to
    SUCCESS + extra.finished=True per spec §1.3 (5-token enum is fixed)."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        step_returns=lambda **kw: None,  # iterator end
    )
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 13, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": None,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["finished"] is True


def test_v2_program_value_move_to_next_step_success(make_v2_pair):
    res_mock = mock.MagicMock(ok=True, message="")
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        step_returns=lambda **kw: res_mock,
    )
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 14, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": None,
    })
    assert reply["status"] == "SUCCESS"
    assert "finished" not in reply


def test_v2_program_value_move_to_next_continuous_mode_rejected(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=True)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 15, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": None,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "raster_in_continuous_mode"


def test_v2_program_value_move_to_next_not_active_rejected(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=False)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 20, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": None,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "raster_not_active"


def test_v2_program_value_move_to_next_step_failed(make_v2_pair):
    res_mock = mock.MagicMock(ok=False, message="motor stalled")
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        step_returns=lambda **kw: res_mock,
    )
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 21, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": None,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "raster_step_failed"
    assert reply["error"]["message"] == "motor stalled"


def test_v2_program_value_non_numeric_coord_rejected(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 22, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_x_coord", "value": "not-a-number",
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "invalid_value"
    outer.request_move_x.assert_not_called()


def test_v2_program_value_unknown_connection(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 16, "action": "PROGRAM_VALUE",
        "connection": "frobnicate", "value": 0,
    })
    assert reply["status"] == "UNKNOWN_CONNECTION"
    assert reply["error"]["code"] == "unknown_connection"


def test_v2_check_value_returns_target_xy_when_cached(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(
        target_xy=(12.5, 7.3), motor_xy=(99.0, 99.0))
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 17, "action": "CHECK_VALUE",
        "connection": "laser_raster_x_coord_monitor",
    })
    assert reply["status"] == "SUCCESS"
    assert reply["value"] == 12.5  # target, not motor


def test_v2_check_value_falls_back_to_motor_xy_when_no_target(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(
        target_xy=None, motor_xy=(99.0, 88.0))
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 18, "action": "CHECK_VALUE",
        "connection": "laser_raster_y_coord_monitor",
    })
    assert reply["status"] == "SUCCESS"
    assert reply["value"] == 88.0


def test_v2_check_value_unknown_connection(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 19, "action": "CHECK_VALUE",
        "connection": "frobnicate",
    })
    assert reply["status"] == "UNKNOWN_CONNECTION"
