"""Regression tests for the motor command queue and backlash UX.

Four independent bugs, each with its own root cause:

  Bug A -- PriorityQueue tuple-comparison crash
    `_enqueue` pushed (priority, created_ts, cmd) into a PriorityQueue.
    `_device_home_both` enqueues two HOME_HARD commands back-to-back; both
    carry the default priority=100 and (on Windows, sub-tick spacing) an
    identical time.time() created_ts. heappush then falls through to
    comparing two MotorCommand instances -> TypeError, since MotorCommand
    defines no __lt__. Fix: a unique monotonic counter as the tiebreaker
    (the documented heapq/PriorityQueue pattern).

  Bug B -- backlash Set produces no acknowledgment
    SET_BACKLASH_X/Y completes with tag "backlash_X"/"backlash_Y", but
    those tags were absent from _LOGGABLE_SUCCESS_TAGS, so _deliver_result
    silently dropped the success and nothing reached the status log.

  Bug C -- backlash Set fires TWICE per single user edit
    The Setpoint spinbox committed on QDoubleSpinBox.editingFinished, which
    Qt emits on BOTH Enter AND focus-out with no de-dup. Enter-then-click-
    away enqueued two SET_BACKLASH commands -> duplicate "backlash X set to
    <v>" log lines. Fix: commit on an explicit "Set" button (one event);
    _on_backlash_set must call request_set_backlash exactly once per click.

  Bug D -- backlash Set blocks the prompt during a Home
    The post-Set re-read called request_get_backlash(wait=True) on the GUI
    thread; _wait_reply excludes user input for timeout_s while the motor
    FIFO is busy with a ~10s Device Home, then logs a spurious timeout.
    Fix: a dedicated async backlash_reading_signal (mirrors
    motor_position_signal). _on_backlash_set is now fully fire-and-forget:
    request_set_backlash + request_get_backlash(wait=False); the Reading
    label updates from the signal when the GET lands.

Standalone-runnable (mirrors the sibling scripts in the repo root tests/):
    conda activate rastering && python tests/test_command_queue.py
Also collectable by pytest. The queue/controller tests need no Qt event
loop or motor hardware; the one UI handler test imports ui.py (PyQt5 +
pyueye) and SKIPs cleanly if those aren't importable.
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
    """A duck-typed `self` carrying ONLY the attributes `_enqueue` touches.

    SystemController is a PyQt5 QObject (cannot be built via object.__new__,
    and full __init__ needs motor DLLs + starts the worker thread). We invoke
    the REAL SystemController._enqueue unbound, passing this stand-in as self,
    so the actual method body runs against the real MotorCommand class with
    zero hardware and no Qt event loop.

    `_q_seq` is present here; the assertion is on _enqueue's *behaviour*, so
    the test is valid whether or not the fix consumes the counter.
    """
    return types.SimpleNamespace(
        _q=queue.PriorityQueue(),
        _q_seq=itertools.count(),
    )


def _deliver_result_self() -> types.SimpleNamespace:
    """Duck-typed `self` for the REAL SystemController._deliver_result.

    Every pyqtSignal is replaced with a Mock so .emit() calls are recorded
    without a QObject / Qt event loop. _LOGGABLE_SUCCESS_TAGS and
    _format_success_message come straight off the real class so the
    acknowledgment path is exercised exactly as in production.
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


def test_device_home_both_enqueue_does_not_crash() -> None:
    """Two equal-priority commands with an identical created_ts must enqueue
    and dequeue in FIFO order -- never raise TypeError.

    This reproduces _device_home_both: request_home('X', hard=True) then
    request_home('Y', hard=True), both priority=100, same time.time() tick.
    The collision is forced deterministically (rather than relying on OS
    clock coarseness) so the test is stable on any platform.
    """
    sc = _enqueue_self()

    c_x = MotorCommand(cmd_type=CommandType.HOME_HARD_X, tag="home_X_hard")
    c_y = MotorCommand(cmd_type=CommandType.HOME_HARD_Y, tag="home_Y_hard")
    assert c_x.priority == c_y.priority == 100, "precondition: equal priority"
    # Force the real-world collision the live traceback proved occurs.
    c_x.created_ts = c_y.created_ts = 1234.5

    SystemController._enqueue(sc, c_x)
    SystemController._enqueue(sc, c_y)  # pre-fix: TypeError MotorCommand < MotorCommand

    first = sc._q.get_nowait()[-1]
    second = sc._q.get_nowait()[-1]
    assert first is c_x, "FIFO within equal priority: X enqueued first"
    assert second is c_y


def test_backlash_set_is_acknowledged() -> None:
    """A successful SET_BACKLASH reply must be loggable and format to the
    hardware-confirmed message, not the generic '<tag> complete.' fallback.
    """
    assert "backlash_X" in SystemController._LOGGABLE_SUCCESS_TAGS
    assert "backlash_Y" in SystemController._LOGGABLE_SUCCESS_TAGS

    res = MotorResult(ok=True, tag="backlash_X", message="backlash X set to 0.05")
    msg = SystemController._format_success_message(res)
    assert msg != "backlash_X complete.", "must not fall through to generic"
    assert "0.05" in msg, "acknowledgment should surface the accepted value"


def test_get_backlash_stays_unlogged() -> None:
    """GET_BACKLASH must NOT be whitelisted -- every Set triggers a re-read
    and startup populates both axes; logging gets would double/triple-log.
    """
    assert "get_backlash_X" not in SystemController._LOGGABLE_SUCCESS_TAGS
    assert "get_backlash_Y" not in SystemController._LOGGABLE_SUCCESS_TAGS


def test_deliver_result_emits_backlash_reading_for_get_only() -> None:
    """Bug D wiring: a successful GET_BACKLASH result must fan out on the
    dedicated async backlash_reading_signal (axis, value) so the UI can
    refresh the Reading label WITHOUT a blocking GUI-thread wait. A SET
    result must NOT touch that signal, but MUST still reach status_signal
    (Bug B acknowledgment stays intact).
    """
    cmd = types.SimpleNamespace(reply_q=None)

    # GET reply: fans out on backlash_reading_signal, NOT on status
    # (get_backlash_* is deliberately not whitelisted -- Bug B test).
    sc = _deliver_result_self()
    get_res = MotorResult(
        ok=True, message="backlash X = 0.05", tag="get_backlash_X", value=0.05
    )
    SystemController._deliver_result(sc, cmd, get_res)
    sc.backlash_reading_signal.emit.assert_called_once_with("X", 0.05)
    sc.status_signal.emit.assert_not_called()

    # SET reply: NO backlash_reading_signal (value is None / not a get tag),
    # but the acknowledgment still reaches status_signal.
    sc2 = _deliver_result_self()
    set_res = MotorResult(
        ok=True, message="backlash X set to 0.05", tag="backlash_X"
    )
    SystemController._deliver_result(sc2, cmd, set_res)
    sc2.backlash_reading_signal.emit.assert_not_called()
    sc2.status_signal.emit.assert_called_once()
    assert "0.05" in sc2.status_signal.emit.call_args[0][0]


def test_on_backlash_set_single_and_nonblocking() -> None:
    """Bugs C + D at the UI handler: one button click must enqueue the Set
    EXACTLY once (no editingFinished double-fire) and the follow-up re-read
    must be NON-blocking (wait=False), so the prompt never freezes behind a
    Device Home.

    Imports ui.py (PyQt5 + pyueye); SKIPs cleanly outside the rastering env.
    """
    try:
        from ui import RasterMainWindow  # noqa: E402,PLC0415
    except Exception as e:  # noqa: BLE001
        _skip(f"ui.py not importable (needs PyQt5+pyueye / rastering env): {e!r}")

    calls = []

    class _Ctl:
        def request_set_backlash(self, axis, value, **kw):
            calls.append(("set", axis, float(value), dict(kw)))

        def request_get_backlash(self, axis, **kw):
            calls.append(("get", axis, dict(kw)))

    class _Spin:
        def value(self):
            return 0.05

    # _refresh_backlash_reading is the OLD blocking path. Provide a no-op so
    # the pre-fix code fails on a clean assertion (gets == 0), not AttributeError.
    stub = types.SimpleNamespace(
        controller=_Ctl(),
        x_backlash=_Spin(),
        y_backlash=_Spin(),
        _refresh_backlash_reading=lambda *a, **k: None,
    )

    RasterMainWindow._on_backlash_set(stub, "X")

    sets = [c for c in calls if c[0] == "set"]
    gets = [c for c in calls if c[0] == "get"]
    assert len(sets) == 1, f"exactly one Set per click (Bugs C/3), got {sets}"
    assert sets[0][3].get("wait", False) is False, "Set must be fire-and-forget"
    assert len(gets) == 1, f"exactly one async re-read, got {gets}"
    assert gets[0][2].get("wait", True) is False, "re-read must be NON-blocking (Bug D)"


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
