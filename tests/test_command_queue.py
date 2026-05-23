"""Regression tests for the motor command queue and backlash UX.

Five independent bugs, each with its own root cause:

  Bug A -- PriorityQueue tuple-comparison crash
    `_enqueue` pushed (priority, created_ts, cmd) into a PriorityQueue.
    `_device_home_both` enqueues two HOME_HARD commands back-to-back; both
    carry the default priority=100 and (on Windows, sub-tick spacing) an
    identical time.time() created_ts. heappush then falls through to
    comparing two MotorCommand instances -> TypeError. Fix: a unique
    monotonic counter as the tiebreaker.

  Bug B -- backlash Set produces no acknowledgment
    SET_BACKLASH_X/Y completes with tag "backlash_X"/"backlash_Y", absent
    from _LOGGABLE_SUCCESS_TAGS, so _deliver_result silently dropped the
    success. Fix: whitelist + _format_success_message branch.

  Bug C -- backlash Set fires TWICE per single user edit
    Setpoint committed on QDoubleSpinBox.editingFinished (fires on BOTH
    Enter AND focus-out). Fix: explicit "Set" button (one event).

  Bug D -- backlash Set blocked the prompt during a Home
    Post-Set re-read called request_get_backlash(wait=True) on the GUI
    thread. Fix: a dedicated async backlash_reading_signal.

  Bug E -- after Bug D's async re-read, the Reading label showed the
           PRE-set value while the console showed the new one
    _on_backlash_set enqueued request_set_backlash (default priority=100)
    THEN request_get_backlash (priority=50). The PriorityQueue is a
    min-heap, so the GET (50) was dequeued BEFORE the SET (100): the
    re-read read the stale backlash. Fix (Option C): the SET command
    itself returns the motor's post-set read-back in MotorResult.value;
    _deliver_result fans that out on backlash_reading_signal. The separate
    GET is removed entirely -- one command, no ordering hazard, and the
    Reading label reflects exactly what the motor accepted (clipping
    included).

Standalone-runnable (mirrors the sibling scripts in the repo root tests/):
    conda activate rastering && python tests/test_command_queue.py
Also collectable by pytest. The queue/controller tests need no Qt event
loop or motor hardware; the one UI handler test imports ui.py (PyQt5 +
rotpy) and SKIPs cleanly if those aren't importable.
"""

from __future__ import annotations

import itertools
import os
import queue
import sys
import types
from unittest import mock

# raster_controller.py / ui.py live one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raster_controller import (  # noqa: E402
    CommandType,
    MotorCommand,
    MotorResult,
    SystemController,
)


class _Skip(Exception):
    """Standalone-runner skip signal (pytest uses pytest.skip instead)."""


def _skip(msg: str) -> None:
    """Skip cleanly under pytest if available, else raise _Skip for the
    __main__ runner to report as SKIP (not FAIL)."""
    try:
        import pytest  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        raise _Skip(msg)
    pytest.skip(msg)


def _enqueue_self() -> types.SimpleNamespace:
    """A duck-typed `self` carrying ONLY what `_enqueue` touches.

    SystemController is a PyQt5 QObject (cannot be built via object.__new__;
    full __init__ needs motor DLLs + starts the worker thread). We invoke
    the REAL SystemController._enqueue unbound so the actual body runs with
    zero hardware and no Qt event loop.
    """
    return types.SimpleNamespace(
        _q=queue.PriorityQueue(),
        _q_seq=itertools.count(),
    )


def _deliver_result_self() -> types.SimpleNamespace:
    """Duck-typed `self` for the REAL SystemController._deliver_result.

    Every pyqtSignal is a Mock so .emit() is recorded without a QObject /
    Qt event loop. _LOGGABLE_SUCCESS_TAGS and _format_success_message come
    straight off the real class so the acknowledgment path is exercised
    exactly as in production.
    """
    return types.SimpleNamespace(
        command_done_signal=mock.Mock(name="command_done_signal"),
        motor_position_signal=mock.Mock(name="motor_position_signal"),
        target_position_signal=mock.Mock(name="target_position_signal"),
        backlash_reading_signal=mock.Mock(name="backlash_reading_signal"),
        status_signal=mock.Mock(name="status_signal"),
        _LOGGABLE_SUCCESS_TAGS=SystemController._LOGGABLE_SUCCESS_TAGS,
        _format_success_message=SystemController._format_success_message,
    )


class _BacklashMotor:
    """Stub motor whose set_backlash CLIPS (readback != requested), proving
    the SET result reports what the motor accepted -- not what was asked."""

    def __init__(self, clipped_to: float) -> None:
        self._clipped = float(clipped_to)
        self.requested = None

    def set_backlash(self, value: float) -> float:
        self.requested = float(value)
        return self._clipped  # e.g. Kinesis clamps to a device limit


def test_device_home_both_enqueue_does_not_crash() -> None:
    """Two equal-priority commands with an identical created_ts must enqueue
    and dequeue in FIFO order -- never raise TypeError (reproduces
    _device_home_both: request_home X then Y, both priority=100)."""
    sc = _enqueue_self()

    c_x = MotorCommand(cmd_type=CommandType.HOME_HARD_X, tag="home_X_hard")
    c_y = MotorCommand(cmd_type=CommandType.HOME_HARD_Y, tag="home_Y_hard")
    assert c_x.priority == c_y.priority == 100, "precondition: equal priority"
    c_x.created_ts = c_y.created_ts = 1234.5  # force the proven collision

    SystemController._enqueue(sc, c_x)
    SystemController._enqueue(sc, c_y)  # pre-fix: TypeError MotorCommand < MotorCommand

    first = sc._q.get_nowait()[-1]
    second = sc._q.get_nowait()[-1]
    assert first is c_x, "FIFO within equal priority: X enqueued first"
    assert second is c_y


def test_backlash_set_is_acknowledged() -> None:
    """A successful SET_BACKLASH reply must be loggable and format to the
    hardware-confirmed message, not the generic '<tag> complete.' fallback."""
    assert "backlash_X" in SystemController._LOGGABLE_SUCCESS_TAGS
    assert "backlash_Y" in SystemController._LOGGABLE_SUCCESS_TAGS

    res = MotorResult(ok=True, tag="backlash_X", message="backlash X set to 0.05")
    msg = SystemController._format_success_message(res)
    assert msg != "backlash_X complete.", "must not fall through to generic"
    assert "0.05" in msg, "acknowledgment should surface the accepted value"


def test_get_backlash_stays_unlogged() -> None:
    """GET_BACKLASH must NOT be whitelisted -- startup populate reads both
    axes; logging gets would double/triple-log."""
    assert "get_backlash_X" not in SystemController._LOGGABLE_SUCCESS_TAGS
    assert "get_backlash_Y" not in SystemController._LOGGABLE_SUCCESS_TAGS


def test_set_backlash_result_carries_motor_readback() -> None:
    """Bug E root-cause fix (Option C): the SET_BACKLASH worker branch must
    put the motor's post-set read-back into MotorResult.value (and the
    acknowledgment message), NOT the requested value. This is what lets the
    Reading label update from the SET itself -- no second, priority-
    inverting GET command.
    """
    stub = types.SimpleNamespace(
        motor_x=_BacklashMotor(clipped_to=0.04),  # asked 0.2, motor clamps to 0.04
        motor_y=_BacklashMotor(clipped_to=0.0),
    )
    cmd = MotorCommand(
        cmd_type=CommandType.SET_BACKLASH_X,
        payload={"value": 0.2},
        tag="backlash_X",
    )

    res = SystemController._execute(stub, cmd)

    assert res.ok and res.tag == "backlash_X"
    assert stub.motor_x.requested == 0.2, "the requested value still reaches the motor"
    assert res.value == 0.04, "result.value must be the motor read-back, not 0.2"
    assert "0.04" in res.message and "0.2" not in res.message, (
        f"ack must show the accepted value, got {res.message!r}"
    )


def test_deliver_result_fans_out_backlash_reading() -> None:
    """_deliver_result must fan a backlash read-back onto the dedicated
    async backlash_reading_signal for BOTH the SET reply (Option C,
    value=read-back) and a standalone GET reply. A SET with no read-back
    (value None) must NOT emit it -- and either way the Bug B
    acknowledgment still reaches status_signal.
    """
    cmd = types.SimpleNamespace(reply_q=None)

    # SET reply carrying the motor read-back -> fans out AND acknowledges.
    sc = _deliver_result_self()
    set_res = MotorResult(
        ok=True, message="backlash X set to 0.04", tag="backlash_X", value=0.04
    )
    SystemController._deliver_result(sc, cmd, set_res)
    sc.backlash_reading_signal.emit.assert_called_once_with("X", 0.04)
    sc.status_signal.emit.assert_called_once()
    assert "0.04" in sc.status_signal.emit.call_args[0][0]

    # Standalone GET reply (e.g. startup populate via fire-and-forget) also fans out;
    # get_backlash_* stays out of the whitelist so it does NOT hit status.
    sc2 = _deliver_result_self()
    get_res = MotorResult(
        ok=True, message="backlash X = 0.04", tag="get_backlash_X", value=0.04
    )
    SystemController._deliver_result(sc2, cmd, get_res)
    sc2.backlash_reading_signal.emit.assert_called_once_with("X", 0.04)
    sc2.status_signal.emit.assert_not_called()

    # Defensive: a SET reply without a read-back must not raise / must not
    # fan out, but must still acknowledge.
    sc3 = _deliver_result_self()
    bare = MotorResult(ok=True, message="backlash X set to 0.04", tag="backlash_X")
    SystemController._deliver_result(sc3, cmd, bare)
    sc3.backlash_reading_signal.emit.assert_not_called()
    sc3.status_signal.emit.assert_called_once()


def test_on_backlash_set_enqueues_only_the_set() -> None:
    """Bug E at the UI handler: one Set click must enqueue EXACTLY the SET
    (fire-and-forget) and NO separate get_backlash -- the SET self-reports
    its read-back, so a second command would only re-introduce the
    priority-inversion (GET priority 50 jumping ahead of SET priority 100).

    Imports ui.py (PyQt5 + rotpy); SKIPs cleanly outside the rastering env.
    """
    try:
        from ui import RasterMainWindow  # noqa: E402,PLC0415
    except Exception as e:  # noqa: BLE001
        _skip(f"ui.py not importable (needs PyQt5+rotpy / rastering env): {e!r}")

    calls = []

    class _Ctl:
        def request_set_backlash(self, axis, value, **kw):
            calls.append(("set", axis, float(value), dict(kw)))

        def request_get_backlash(self, axis, **kw):
            calls.append(("get", axis, dict(kw)))

    class _Spin:
        def value(self):
            return 0.05

    stub = types.SimpleNamespace(
        controller=_Ctl(), x_backlash=_Spin(), y_backlash=_Spin()
    )

    RasterMainWindow._on_backlash_set(stub, "X")

    sets = [c for c in calls if c[0] == "set"]
    gets = [c for c in calls if c[0] == "get"]
    assert len(sets) == 1, f"exactly one Set per click (Bugs C/3), got {sets}"
    assert sets[0][3].get("wait", False) is False, "Set must be fire-and-forget"
    assert gets == [], (
        f"NO separate get_backlash -- it would priority-invert past the Set "
        f"(Bug E); the Set self-reports its read-back. got {gets}"
    )


def test_apply_loaded_backlash_widgets_avoids_priority_inversion() -> None:
    """Bug E follow-up (review Issue 1): on a calibration load,
    load_calibration_from_path has already enqueued a priority-100
    request_set_backlash for each bundled axis. The UI must NOT then issue
    a priority-50 GET re-read (request_get_backlash via
    _refresh_backlash_reading) -- it would dequeue ahead of that SET and
    seed the Setpoint spinbox with the stale pre-load value. For bundled
    axes it must seed widgets directly from the loaded value; only
    legacy/absent axes fall back to the idle-FIFO re-read (no SET pending).

    Imports ui.py (PyQt5 + rotpy); SKIPs cleanly outside the rastering env.
    """
    try:
        from ui import RasterMainWindow  # noqa: E402,PLC0415
    except Exception as e:  # noqa: BLE001
        _skip(f"ui.py not importable (needs PyQt5+rotpy / rastering env): {e!r}")

    seeded, reread = [], []
    stub = types.SimpleNamespace(
        _set_backlash_widgets=lambda axis, v: seeded.append((axis, float(v))),
        _refresh_backlash_reading=lambda axis, **kw: reread.append((axis, kw)),
    )

    # Bundle has X backlash but not Y -> X seeded (no GET), Y re-reads.
    RasterMainWindow._apply_loaded_backlash_widgets(stub, {"backlash": {"x": 0.05}})
    assert seeded == [("X", 0.05)], f"X must be seeded from the bundle, got {seeded}"
    assert [a for a, _ in reread] == ["Y"], (
        f"only the absent axis (Y) may re-read; got {reread}"
    )
    assert reread[0][1].get("also_setpoint") is True

    # Legacy bundle (no 'backlash' key) -> no SET enqueued; both re-read.
    seeded.clear()
    reread.clear()
    RasterMainWindow._apply_loaded_backlash_widgets(stub, {})
    assert seeded == [], "legacy bundle: nothing to seed"
    assert [a for a, _ in reread] == ["X", "Y"], (
        f"legacy: both axes re-read (idle FIFO, safe); got {reread}"
    )


def test_single_axis_move_is_acknowledged() -> None:
    """User Home 'Go X/Y' (request_move_motor_axis, tag move_motor_{x,y}_only)
    is a user-initiated blocking move; it MUST log a success ack like its
    sibling 'move_motor'. Pre-fix the tag was in NEITHER whitelist nor any
    _format branch, so a successful Go produced ZERO log output (the motor
    moved silently) -> reported as 'soft/User Home broken'."""
    for tag in ("move_motor_x_only", "move_motor_y_only"):
        assert tag in SystemController._LOGGABLE_SUCCESS_TAGS, (
            f"{tag} missing from _LOGGABLE_SUCCESS_TAGS -> silent success"
        )
        res = MotorResult(ok=True, tag=tag, message="Move complete",
                          motor_xy=(1.5, 2.5))
        msg = SystemController._format_success_message(res)
        assert msg != f"{tag} complete.", "must not fall through to generic"
        assert "Move complete" in msg and "1.5" in msg

    # End-to-end: _deliver_result must actually reach status_signal.
    sc = _deliver_result_self()
    SystemController._deliver_result(
        sc, types.SimpleNamespace(reply_q=None),
        MotorResult(ok=True, tag="move_motor_x_only", message="Move complete",
                    motor_xy=(1.5, 2.5)),
    )
    sc.status_signal.emit.assert_called_once()


def test_single_axis_move_logs_start() -> None:
    """A single-axis move blocks the motor thread for seconds (KCube.move_to);
    like move_motor it MUST emit a 'starting...' line for progress feedback."""
    for tag, key, val, ctype in (
        ("move_motor_x_only", "x", 4.0, CommandType.MOVE_MOTOR_X_ONLY),
        ("move_motor_y_only", "y", 7.0, CommandType.MOVE_MOTOR_Y_ONLY),
    ):
        assert tag in SystemController._LOGGABLE_START_TAGS, (
            f"{tag} missing from _LOGGABLE_START_TAGS -> no progress feedback"
        )
        cmd = MotorCommand(cmd_type=ctype, payload={key: val}, tag=tag)
        msg = SystemController._format_start_message(cmd)
        assert msg != f"{tag} starting...", "must not fall through to generic"
        assert "starting" in msg.lower() and str(int(val)) in msg


def test_stop_is_acknowledged() -> None:
    """Audit finding (secondary): a successful STOP is a user-initiated action
    but tag 'stop' was absent from _LOGGABLE_SUCCESS_TAGS, so Stop confirmed
    nothing in the log. STOP is instantaneous -> no START line needed."""
    assert "stop" in SystemController._LOGGABLE_SUCCESS_TAGS
    res = MotorResult(ok=True, tag="stop", message="Stop executed (user)")
    msg = SystemController._format_success_message(res)
    assert msg != "stop complete.", "must not fall through to generic"
    assert "stop" in msg.lower()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException as e:  # noqa: BLE001
                if type(e).__name__ in ("_Skip", "Skipped"):
                    print(f"SKIP {name}: {e}")
                    continue
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
