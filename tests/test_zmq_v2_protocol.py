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
  * CHECK_VALUE returns cached target XY when present, else a typed
    refusal -- never the motor-frame value.
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
                move_pair_ok=True,
                step_returns=None, raster_step_calls=None,
                raster_source="__default__"):
    """Stand-in for SystemController; duck-typed to what _RasteringV2Server
    actually reads/calls."""
    _require_rc()
    outer = mock.MagicMock()
    outer._state_lock = threading.RLock()
    outer._last_target_xy = target_xy
    outer._last_motor_xy = motor_xy
    outer._raster_active = raster_active
    outer._raster_continuous = raster_continuous
    # Ownership of the raster: None (idle) / "local" / "remote". MagicMock would
    # auto-create a truthy child and hide a missing assignment.
    # Ownership is now a persistent flag (2026-08-07 overhaul), so tests must
    # be able to say who holds it. Default keeps the historical shape: armed
    # fixtures were built by a local arm.
    if raster_source == "__default__":
        raster_source = "local" if raster_active else None
    outer._raster_source = raster_source
    # Real int, set in SystemController.__init__ since the arm-time filter
    # started reporting drop counts. A MagicMock auto-child here is not JSON
    # serializable and turns every from-scratch arm reply into an ERROR.
    outer._raster_dropped_count = 0
    # Real controller uses an indexed point list (_raster_path_pts), not a
    # one-shot generator; a non-empty list means "raster configured".
    outer._raster_path_pts = [(0.0, 0.0)] if raster_has_path else []
    # Real controller defaults to no remote-arm provider (headless). MagicMock
    # would auto-create a truthy attribute and defeat the None check.
    outer.remote_arm_provider = None
    # Shots-per-step BLACS last programmed. Explicit None: MagicMock's
    # auto-attribute would make "was it stored?" assertions vacuous.
    outer._remote_shots_per_step = None

    def _move_ok(value, *, source, wait, timeout_s):
        res = mock.MagicMock()
        res.ok = bool(value)  # value=0 -> fail; non-zero -> ok per test seed
        res.message = "" if res.ok else "rejected"
        return res

    def _make_move_factory(success):
        # *value: single-axis movers take (v), the compound pair movers (x, y).
        def _move(*value, source, wait, timeout_s):
            res = mock.MagicMock()
            res.ok = success
            res.message = "" if success else "motor rejected"
            return res
        return _move

    outer.request_move_x.side_effect = _make_move_factory(move_x_ok)
    outer.request_move_y.side_effect = _make_move_factory(move_y_ok)
    outer.request_move_target.side_effect = _make_move_factory(move_pair_ok)
    outer.request_move_motor.side_effect = _make_move_factory(move_pair_ok)
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


# ------------------------------------------------- compound (x, y) write
# BLACS knows both coords, so it programs them as ONE PROGRAM_VALUE on
# `laser_raster_xy`: one MOVE_TARGET over the true pair, no intermediate
# (x_new, y_old) excursion and no stale partner coordinate.


def test_v2_program_value_pair_delegates_to_request_move_target(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 60, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": [12.5, 7.25],
    })
    assert reply["status"] == "SUCCESS"
    assert reply["id"] == 60
    outer.request_move_target.assert_called_once()
    args, kwargs = outer.request_move_target.call_args
    assert args == (12.5, 7.25)
    assert kwargs["source"] == "zmq"
    assert kwargs["wait"] is True          # same reply semantics as per-coord
    assert kwargs["timeout_s"] == 10.0
    # Never the single-axis path -- that's what pairs a stale partner coord.
    outer.request_move_x.assert_not_called()
    outer.request_move_y.assert_not_called()


@pytest.mark.parametrize("bad", [
    [1.0], [1.0, 2.0, 3.0], [], "1,2", 5.0, None, True,
    {"x": 1.0, "y": 2.0}, [1.0, "nope"],
    [1.0, float("inf")], [float("nan"), 2.0],
])
def test_v2_program_value_pair_bad_shape_rejected(make_v2_pair, bad):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 61, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": bad,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "invalid_value"
    assert reply["error"]["retryable"] is False
    outer.request_move_target.assert_not_called()
    outer.request_move_motor.assert_not_called()


def test_v2_program_value_pair_uncalibrated_is_not_gated(make_v2_pair):
    """Uncalibrated behaves exactly like a single-coord write: the handler
    doesn't gate on calibration, it hands the pair to MOVE_TARGET, whose
    cal-is-None branch is the motor-space passthrough (bounds-checked in
    the worker; pinned by test_raster_pathmodel)."""
    outer, client_t, v2_server = make_v2_pair()
    outer.calibration = None
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 62, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": [1.0, 2.0],
    })
    assert reply["status"] == "SUCCESS"
    outer.request_move_target.assert_called_once()


def test_v2_program_value_pair_motor_frame_bypasses_calibration(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 63, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": [3.0, 4.0],
        "args": {"frame": "motor", "timeout_sec": 30.0},
    })
    assert reply["status"] == "SUCCESS"
    outer.request_move_motor.assert_called_once()
    args, kwargs = outer.request_move_motor.call_args
    assert args == (3.0, 4.0)
    assert kwargs["timeout_s"] == 30.0
    outer.request_move_target.assert_not_called()


def test_v2_program_value_pair_unknown_frame_rejected(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 64, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": [3.0, 4.0],
        "args": {"frame": "galactic"},
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "invalid_frame"
    outer.request_move_target.assert_not_called()
    outer.request_move_motor.assert_not_called()


def test_v2_program_value_pair_move_failure_returns_retryable(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(move_pair_ok=False)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 65, "action": "PROGRAM_VALUE",
        "connection": "laser_raster_xy", "value": [12.5, 7.25],
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "motor_move_failed"
    assert reply["error"]["retryable"] is True


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
    # No active raster AND no remote-arm provider (headless): typed error,
    # returned immediately (the no-provider path has no blocking wait).
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 12, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 1,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "no_raster_configured"
    assert "no GUI panel attached" in reply["error"]["message"]


def test_v2_arm_raster_remote_provider_success_then_step(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)

    def fake_provider(want_continuous, reply):
        # What ui.py's main-thread slot does on success, minus the widgets.
        outer._raster_path_pts = [(0.0, 0.0), (1.0, 1.0)]
        outer._raster_active = True
        outer._raster_continuous = bool(want_continuous)
        reply(True)

    outer.remote_arm_provider = fake_provider
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 41, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["mode"] == "step"
    # The armed state is real: move_to_next now succeeds against it.
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 42, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": 1,
    })
    assert reply["status"] == "SUCCESS"


def test_v2_arm_raster_remode_takes_control(make_v2_pair):
    """Re-moding a locally-started raster hands ownership to the remote client
    (drives the GUI's "Control: REMOTE (BLACS)" indicator)."""
    outer, client_t, v2_server = make_v2_pair(raster_active=True, raster_has_path=True)
    assert outer._raster_source == "local"          # armed at the GUI
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 46, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
    })
    assert reply["status"] == "SUCCESS"
    assert outer._raster_source == "remote"
    outer.raster_source_signal.emit.assert_called_once_with("remote")


def test_v2_arm_raster_remote_provider_arms_as_remote_source(make_v2_pair):
    """End-to-end arm-from-scratch: handler -> provider -> the REAL
    SystemController.start_raster, which must record source "remote". A
    following move_to_next keeps it there (the step-side flip itself is pinned
    in test_raster_pathmodel.test_raster_step_flips_source_to_the_stepper --
    raster_step is a mock here)."""
    import tempfile
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)

    def fake_provider(want_continuous, reply):
        # ui.py's main-thread slot, minus the widgets: _start_raster(source="remote").
        rc.SystemController.start_raster(
            outer, [(0.0, 0.0), (1.0, 1.0)], continuous=bool(want_continuous),
            log_dir=tempfile.mkdtemp(), source="remote")
        reply(True)

    outer.remote_arm_provider = fake_provider
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 47, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["mode"] == "step"
    assert outer._raster_active is True
    assert outer._raster_source == "remote"

    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 48, "action": "PROGRAM_VALUE",
        "connection": "move_to_next", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert outer.raster_step.call_args.kwargs["source"] == "zmq"
    assert outer._raster_source == "remote"


def test_v2_arm_raster_remote_continuous_from_scratch_rejected(make_v2_pair):
    """Remote arm is step-only: continuous motion must start at the GUI."""
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)
    calls = []
    outer.remote_arm_provider = lambda want_continuous, reply: calls.append(
        want_continuous)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 45, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 1,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "continuous_arm_requires_gui"
    assert calls == []                    # provider never invoked


def test_v2_arm_raster_remote_provider_failure_code_forwarded(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)
    outer.remote_arm_provider = lambda want_continuous, reply: reply(
        False, "not_calibrated", "no calibration set; calibrate in the GUI first")
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 43, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "not_calibrated"
    assert "calibrat" in reply["error"]["message"]


def test_v2_arm_raster_remote_provider_timeout(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair(raster_active=False, raster_has_path=False)
    outer.remote_arm_provider = lambda want_continuous, reply: None  # never replies
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 44, "action": "PROGRAM_VALUE",
        "connection": "arm_raster", "value": 0,
        "args": {"timeout_sec": 0.2},
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "arm_timeout"


def test_v2_program_value_move_to_next_iter_end_returns_finished_extra(make_v2_pair):
    """Iterator exhaustion: v1 used non-spec status "FINISHED"; v2 maps to
    SUCCESS + extra.finished=True per spec §1.3 (5-token enum is fixed).
    Under local hold the same None is an acknowledge, pinned separately."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_continuous=False,
        raster_source="remote",           # exhausted while BLACS drives
        step_returns=lambda **kw: None,   # iterator end
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


def test_v2_move_to_next_not_active_under_blacs_rejected(make_v2_pair):
    """Nothing armed while BLACS drives is an error -- it is what triggers
    BLACS's re-arm self-heal. The local/unset cases fire in place instead
    (pinned separately)."""
    outer, client_t, v2_server = make_v2_pair(raster_active=False,
                                              raster_source="remote")
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


# ------------------------------------------------- shots_per_step / disarm
# BLACS's Rastering tab tells the GUI how many shots it fires per point, and
# says so when the operator unchecks Raster Mode there. Both are display /
# lifecycle only -- neither moves a motor.


def test_v2_shots_per_step_stored_and_echoed(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 50, "action": "PROGRAM_VALUE",
        "connection": "shots_per_step", "value": 7,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["shots_per_step"] == 7
    assert outer._remote_shots_per_step == 7
    outer.raster_shots_per_step_signal.emit.assert_called_once_with(7)


@pytest.mark.parametrize("bad", ["abc", 0, -3, None])
def test_v2_shots_per_step_invalid_rejected(make_v2_pair, bad):
    """Unparseable or < 1 -> typed invalid_value naming the offending value;
    nothing stored, nothing emitted."""
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 51, "action": "PROGRAM_VALUE",
        "connection": "shots_per_step", "value": bad,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "invalid_value"
    assert reply["error"]["retryable"] is False
    assert repr(bad) in reply["error"]["message"]
    assert outer._remote_shots_per_step is None
    outer.raster_shots_per_step_signal.emit.assert_not_called()


def test_v2_disarm_raster_while_active_releases_to_local(make_v2_pair):
    """disarm_raster releases ownership and PRESERVES the armed path.
    BLACS unticking Raster Mode means 'I stop driving', not 'destroy the
    operator's pattern' -- only the GUI Stop button destroys (2026-08-07)."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_has_path=True, raster_continuous=False,
        raster_source="remote")
    outer._remote_shots_per_step = 4
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 52, "action": "PROGRAM_VALUE",
        "connection": "disarm_raster", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["disarmed"] is True
    outer.stop_raster.assert_not_called()
    assert outer._raster_source == "local"
    assert outer._raster_path_pts, "armed path must survive a release"
    outer.raster_source_signal.emit.assert_called_once_with("local")
    # Shots-per-step is meaningless once released -> back to "--" in the GUI.
    assert outer._remote_shots_per_step is None
    outer.raster_shots_per_step_signal.emit.assert_called_once_with(None)


def test_v2_disarm_raster_when_inactive_is_success_noop(make_v2_pair):
    """Idempotent: BLACS may disarm something already stopped at the GUI."""
    outer, client_t, v2_server = make_v2_pair(raster_active=False)
    outer._remote_shots_per_step = 4
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 53, "action": "PROGRAM_VALUE",
        "connection": "disarm_raster", "value": 1,
    })
    assert reply["status"] == "SUCCESS"
    assert reply["disarmed"] is False
    outer.stop_raster.assert_not_called()
    assert outer._remote_shots_per_step is None
    outer.raster_shots_per_step_signal.emit.assert_called_once_with(None)


def test_v2_disarm_raster_refuses_continuous_run(make_v2_pair):
    """Never kill an operator's continuous run remotely -- and leave the
    shots-per-step display alone, since nothing was disarmed."""
    outer, client_t, v2_server = make_v2_pair(
        raster_active=True, raster_has_path=True, raster_continuous=True)
    outer._remote_shots_per_step = 4
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 54, "action": "PROGRAM_VALUE",
        "connection": "disarm_raster", "value": 1,
    })
    assert reply["status"] == "ERROR"
    assert reply["error"]["code"] == "raster_in_continuous_mode"
    assert reply["error"]["retryable"] is False
    outer.stop_raster.assert_not_called()
    assert outer._remote_shots_per_step == 4
    outer.raster_shots_per_step_signal.emit.assert_not_called()


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


def test_v2_check_value_never_answers_with_motor_xy(make_v2_pair):
    """Frame hygiene: the monitors are target-frame. With no target cache the
    reply is a typed refusal -- NEVER the motor-frame mm value, which BLACS
    would store as if it were a target coordinate."""
    outer, client_t, v2_server = make_v2_pair(
        target_xy=None, motor_xy=(99.0, 88.0))
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 18, "action": "CHECK_VALUE",
        "connection": "laser_raster_y_coord_monitor",
    })
    assert reply["status"] == "UNKNOWN_CONNECTION"
    assert reply["error"]["code"] == "position_not_initialized"
    assert "value" not in reply


def test_v2_check_value_unknown_connection(make_v2_pair):
    outer, client_t, v2_server = make_v2_pair()
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 19, "action": "CHECK_VALUE",
        "connection": "frobnicate",
    })
    assert reply["status"] == "UNKNOWN_CONNECTION"


def test_v2_check_value_uninitialized_position_returns_typed_error(make_v2_pair):
    """Fresh GUI start: both position caches None. SUCCESS with an omitted
    value key would KeyError BLACS's float(reply["value"]) — must be a
    typed, retryable error instead."""
    outer, client_t, v2_server = make_v2_pair(target_xy=None, motor_xy=None)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 20, "action": "CHECK_VALUE",
        "connection": "laser_raster_x_coord_monitor",
    })
    assert reply["status"] == "UNKNOWN_CONNECTION"
    assert reply["error"]["code"] == "position_not_initialized"
    assert reply["error"]["retryable"] is True
    assert "value" not in reply

    # Once a TARGET position lands, CHECK_VALUE recovers to SUCCESS with a
    # value (a motor-only read leaves it refusing -- see the test above).
    with outer._state_lock:
        outer._last_target_xy = (3.25, 4.5)
    reply = _roundtrip(client_t, v2_server, {
        "v": 2, "id": 21, "action": "CHECK_VALUE",
        "connection": "laser_raster_x_coord_monitor",
    })
    assert reply["status"] == "SUCCESS"
    assert reply["value"] == 3.25
