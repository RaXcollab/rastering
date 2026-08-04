"""Stage 1 (indexed path model) + Stage 2 (F2 selection/goto) tests.

The raster path moves from a one-shot generator (self._raster_iter) to an
indexed list (self._raster_path_pts) + cursor (self._raster_index), with one
advance primitive _next_raster_point_locked(). These tests pin that the indexed
walk reproduces the generator byte-for-byte and that the Continuous + Step paths
are preserved, then that selection/goto (the "go to an arbitrary site" feature)
behave per spec (select = no motion; goto = motion via move_target tag; rejected
mid-continuous-run).

Pattern mirrors test_command_queue.py: invoke real SystemController methods
UNBOUND on a duck-typed SimpleNamespace `self`, with a real RLock / PriorityQueue
and Mock signals. No Qt event loop, no motor DLLs.

    conda activate rastering && python tests/test_raster_pathmodel.py   # standalone
    conda activate rastering && pytest tests/test_raster_pathmodel.py   # pytest
"""

from __future__ import annotations

import itertools
import os
import queue
import sys
import threading
import types
from unittest import mock

# raster_controller.py / raster_paths.py live one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from raster_controller import (  # noqa: E402
    AffineCalibration,
    CommandType,
    MotorCommand,
    MotorResult,
    SystemController,
    clamp_to_bounds,
)
from raster_paths import (  # noqa: E402
    RasterSpec,
    collect_points,
    iter_path_from_spec,
)


SPECS = {
    "square_x": RasterSpec(kind="square_x", bounds=(0.0, 0.05, 0.0, 0.05), xstep=0.01, ystep=0.01),
    "square_y": RasterSpec(kind="square_y", bounds=(0.0, 0.05, 0.0, 0.05), xstep=0.01, ystep=0.01),
    "spiral": RasterSpec(kind="spiral", origin=(0.0, 0.0), radius=0.05, step=0.01,
                         angle_step=0.4, bounds=(-0.05, 0.05, -0.05, 0.05)),
    "hull": RasterSpec(kind="hull", hull_points=[(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05)],
                       xstep=0.01, ystep=0.01),
}


def _expected(spec: RasterSpec):
    return collect_points(iter_path_from_spec(spec), max_points=50000)


# ----------------------------------------------------------------------------
# Stage 1 -- indexed path model
# ----------------------------------------------------------------------------

def _cursor_self(pts):
    """Minimal `self` for the bare cursor primitive (caller holds the lock)."""
    return types.SimpleNamespace(_raster_path_pts=list(pts), _raster_index=0)


def test_next_point_locked_walks_full_path():
    """The indexed cursor yields exactly the generator's points, in order, for
    every pattern -- proving the refactor preserves the path byte-for-byte."""
    for name, spec in SPECS.items():
        exp = _expected(spec)
        assert exp, f"{name}: precondition non-empty"
        sc = _cursor_self(exp)
        walked = []
        while True:
            pt = SystemController._next_raster_point_locked(sc)
            if pt is None:
                break
            walked.append(pt)
        assert walked == exp, f"{name}: indexed walk != generator"
        assert sc._raster_index == len(exp), f"{name}: cursor not at end"


def test_next_point_locked_exhaustion_returns_none():
    """Index >= len is the exact equivalent of StopIteration; idempotent at end."""
    sc = _cursor_self([(1.0, 2.0)])
    assert SystemController._next_raster_point_locked(sc) == (1.0, 2.0)
    assert SystemController._next_raster_point_locked(sc) is None
    assert SystemController._next_raster_point_locked(sc) is None
    assert sc._raster_index == 1


def _runloop_self(pts, *, active=True):
    sc = types.SimpleNamespace(
        _state_lock=threading.RLock(),
        _raster_path_pts=list(pts),
        _raster_index=0,
        _raster_active=active,
        _q=queue.PriorityQueue(),
        _q_seq=itertools.count(),
        _finished=False,
    )
    sc._next_raster_point_locked = lambda **kw: SystemController._next_raster_point_locked(sc, **kw)
    sc._enqueue = lambda cmd: SystemController._enqueue(sc, cmd)
    sc._finish_raster = lambda: setattr(sc, "_finished", True)
    return sc


def test_enqueue_next_raster_point_matches_generator():
    """Continuous driver enqueues MOVE_TARGET commands whose targets are exactly
    the generator's points, then calls _finish_raster() once, on exhaustion."""
    spec = SPECS["square_x"]
    exp = _expected(spec)
    sc = _runloop_self(exp)
    targets = []
    for _ in range(len(exp) + 1):  # one extra call to hit exhaustion
        SystemController._enqueue_next_raster_point(sc)
        while not sc._q.empty():
            _, _, cmd = sc._q.get_nowait()
            assert cmd.cmd_type == CommandType.MOVE_TARGET
            assert cmd.tag == "raster_step" and cmd.priority == 100
            assert cmd.source == "raster"
            targets.append(cmd.payload["target_xy"])
    assert targets == exp
    assert sc._finished is True


def _step_self(pts, *, index=0, active=True, continuous=False):
    sc = types.SimpleNamespace(
        _state_lock=threading.RLock(),
        _raster_path_pts=list(pts),
        _raster_index=index,
        _raster_active=active,
        _raster_continuous=continuous,
        _q=queue.PriorityQueue(),
        _q_seq=itertools.count(),
        error_signal=mock.Mock(),
        raster_source_signal=mock.Mock(),
        _raster_source=None,
        _finished=False,
        # goto shares this `self` so one namespace covers step<->goto handoff
        _raster_selected_index=-1,
        selection_changed_signal=mock.Mock(),
        _moves=[],
    )
    sc._next_raster_point_locked = lambda **kw: SystemController._next_raster_point_locked(sc, **kw)
    sc._enqueue = lambda cmd: SystemController._enqueue(sc, cmd)
    sc._finish_raster = lambda: setattr(sc, "_finished", True)
    sc.request_move_target = lambda x, y, **kw: sc._moves.append((float(x), float(y), kw))
    return sc


def test_raster_step_advances_one_and_enqueues():
    """One Step -> one MOVE_TARGET (tag raster_step, prio 100, source preserved),
    cursor advanced by exactly one."""
    sc = _step_self([(1.0, 2.0), (3.0, 4.0)])
    res = SystemController.raster_step(sc, source="ui", wait=False)
    assert res is None
    assert sc._raster_index == 1
    assert sc._q.qsize() == 1
    _, _, cmd = sc._q.get_nowait()
    assert cmd.cmd_type == CommandType.MOVE_TARGET
    assert cmd.tag == "raster_step" and cmd.priority == 100
    assert cmd.payload["target_xy"] == (1.0, 2.0)
    assert cmd.source == "ui"


def test_step_wraps_to_path_start_for_any_source():
    """Step-mode stepping wraps for BLACS and operator alike: an exhausted
    cursor rewinds to point 0 and the raster never finishes. Stop / disarm
    are the only teardown paths (2026-08-04, ends the zmq-vs-ui split)."""
    pts = [(1.0, 2.0), (3.0, 4.0)]
    for source in ("zmq", "ui"):
        sc = _step_self(pts, index=len(pts))
        SystemController.raster_step(sc, source=source, wait=False)
        assert sc._finished is False, source
        assert sc._raster_index == 1, source  # consumed pts[0] -> point_index 0
        _, _, cmd = sc._q.get_nowait()
        assert cmd.payload["target_xy"] == (1.0, 2.0)


def test_zmq_step_on_empty_path_still_finishes():
    """wrap=True must never spin on an empty path: no point, no move,
    finish as before."""
    sc = _step_self([], index=0)
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._finished is True
    assert sc._q.empty()


def test_raster_step_rejected_in_continuous():
    """Step is disabled in continuous mode -- no enqueue, no cursor move, error."""
    sc = _step_self([(1.0, 2.0)], continuous=True)
    res = SystemController.raster_step(sc, source="ui", wait=False)
    assert res is None
    assert sc._raster_index == 0
    assert sc._q.qsize() == 0
    sc.error_signal.emit.assert_called_once()


def test_raster_step_empty_path_finishes():
    """An empty armed path finishes immediately -- there is nothing to wrap
    to, for any source (the only remaining step-driven teardown)."""
    sc = _step_self([], active=True)
    res = SystemController.raster_step(sc, source="ui", wait=False)
    assert res is None
    assert sc._finished is True
    assert sc._q.qsize() == 0


def _start_self(*, calibration=True):
    return types.SimpleNamespace(
        _state_lock=threading.RLock(),
        calibration=(object() if calibration else None),
        status_signal=mock.Mock(),
        raster_state_signal=mock.Mock(),
        raster_log_path_signal=mock.Mock(),
        raster_source_signal=mock.Mock(),
        _raster_active=False,
        _raster_source=None,
        _raster_path_pts=[],
        _raster_index=0,
    )


def test_start_raster_refuses_empty_path():
    """An empty path must NOT arm -- it emits a 'no points' status and returns,
    rather than arming then immediately finishing."""
    sc = _start_self()
    SystemController.start_raster(sc, [], continuous=False)
    assert sc._raster_active is False
    assert sc._raster_path_pts == []
    sc.status_signal.emit.assert_called()


def test_start_raster_materializes_indexed_total():
    """start_raster materializes the path into the indexed list, resets the
    cursor, and populates _raster_total_steps (progress display now works)."""
    import tempfile
    sc = _start_self()
    spec = SPECS["square_x"]
    exp = _expected(spec)
    SystemController.start_raster(sc, iter_path_from_spec(spec),
                                  continuous=False, log_dir=tempfile.mkdtemp())
    assert sc._raster_active is True
    assert sc._raster_path_pts == exp
    assert sc._raster_index == 0
    assert sc._raster_total_steps == len(exp)


# ----------------------------------------------------------------------------
# Raster ownership (_raster_source) -- drives the GUI's "Control:" indicator
# ----------------------------------------------------------------------------

def _teardown_self():
    """`self` for the two paths that end a raster (stop_raster, _finish_raster),
    armed and remotely owned."""
    return types.SimpleNamespace(
        _state_lock=threading.RLock(),
        _raster_path_pts=[(0.0, 0.0)],
        _raster_index=0,
        _raster_selected_index=-1,
        _raster_active=True,
        _raster_continuous=False,
        _raster_source="remote",
        status_signal=mock.Mock(),
        raster_state_signal=mock.Mock(),
        raster_source_signal=mock.Mock(),
        raster_finished_signal=mock.Mock(),
        selection_changed_signal=mock.Mock(),
        _flush_raster_log=mock.Mock(),
    )


def test_start_raster_records_arming_source():
    """Local Start arms as "local"; the remote-arm path (ZMQ provider / the
    "Arm for remote stepping" button) arms as "remote". Both notify the UI."""
    import tempfile
    for source, expected in ((None, "local"), ("local", "local"), ("remote", "remote")):
        sc = _start_self()
        kwargs = {} if source is None else {"source": source}
        SystemController.start_raster(sc, iter_path_from_spec(SPECS["square_x"]),
                                      continuous=False, log_dir=tempfile.mkdtemp(),
                                      **kwargs)
        assert sc._raster_source == expected
        sc.raster_source_signal.emit.assert_called_once_with(expected)


def test_raster_step_flips_source_to_the_stepper():
    """On an unowned raster, whoever steps claims it: move_to_next (source "zmq")
    for BLACS, the GUI's Step button (source "ui") for the operator. Both emit
    the new owner."""
    for source, expected in (("zmq", "remote"), ("ui", "local")):
        sc = _step_self([(1.0, 2.0), (3.0, 4.0)])
        SystemController.raster_step(sc, source=source, wait=False)
        assert sc._raster_source == expected
        sc.raster_source_signal.emit.assert_called_once_with(expected)


def test_raster_step_refuses_local_step_while_blacs_owns_it():
    """A local Step into a BLACS-owned raster would advance the cursor BLACS is
    counting on for its shot bookkeeping: refuse, leaving cursor and ownership
    untouched -- and do NOT mistake the refusal for an exhausted path."""
    sc = _step_self([(1.0, 2.0), (3.0, 4.0)])
    sc._raster_source = "remote"
    assert SystemController.raster_step(sc, source="ui", wait=False) is None
    assert sc._raster_index == 0
    assert sc._q.qsize() == 0
    assert sc._raster_source == "remote"
    assert sc._finished is False
    sc.raster_source_signal.emit.assert_not_called()
    sc.error_signal.emit.assert_called_once()

    # BLACS itself is never refused, and "Return to local control" is the way in.
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._raster_index == 1
    assert SystemController.take_local_control(sc) is True
    SystemController.raster_step(sc, source="ui", wait=False)
    assert sc._raster_index == 2
    assert sc._raster_source == "local"


def test_raster_step_on_inactive_raster_leaves_source_unset():
    """A step against a dead raster must not claim ownership."""
    sc = _step_self([(1.0, 2.0)], active=False)
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._raster_source is None
    sc.raster_source_signal.emit.assert_not_called()


def test_finish_raster_clears_source():
    """Path exhausted -> nobody owns the raster (indicator back to idle)."""
    sc = _teardown_self()
    SystemController._finish_raster(sc)
    assert sc._raster_source is None
    sc.raster_source_signal.emit.assert_called_once_with(None)


def test_stop_raster_clears_source():
    """Operator Stop -> nobody owns the raster, even if BLACS was driving."""
    sc = _teardown_self()
    SystemController.stop_raster(sc)
    assert sc._raster_source is None
    sc.raster_source_signal.emit.assert_called_once_with(None)


def test_take_local_control_flips_source_on_active_raster():
    """"Return to local control": the raster stays armed, only ownership moves,
    so the operator's Step button drives it until BLACS steps again."""
    sc = _teardown_self()
    assert SystemController.take_local_control(sc) is True
    assert sc._raster_source == "local"
    assert sc._raster_active is True          # never disarms
    sc.raster_source_signal.emit.assert_called_once_with("local")


def test_take_local_control_is_noop_when_no_raster_active():
    """Nothing to take back -> False, and the idle indicator is left alone."""
    sc = _teardown_self()
    sc._raster_active = False
    sc._raster_source = None
    assert SystemController.take_local_control(sc) is False
    assert sc._raster_source is None
    sc.raster_source_signal.emit.assert_not_called()


# ----------------------------------------------------------------------------
# Stage 2 -- F2 selection / goto (no motion on select; motion only on goto)
# ----------------------------------------------------------------------------

def _select_self(pts, *, active=True, continuous=False):
    sc = types.SimpleNamespace(
        _state_lock=threading.RLock(),
        _raster_path_pts=list(pts),
        _raster_index=0,
        _raster_selected_index=-1,
        _raster_active=active,
        _raster_continuous=continuous,
        _raster_source=None,
        selection_changed_signal=mock.Mock(),
        raster_source_signal=mock.Mock(),
        status_signal=mock.Mock(),
        error_signal=mock.Mock(),
        _moves=[],
    )
    # request_move_target is the existing primitive; record calls, no hardware.
    sc.request_move_target = lambda x, y, **kw: sc._moves.append((float(x), float(y), kw))
    return sc


def test_select_path_index_clamps_and_emits_no_motion():
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    sc = _select_self(pts)
    xy = SystemController.select_path_index(sc, 1)
    assert xy == (1.0, 1.0)
    assert sc._raster_selected_index == 1
    assert sc._moves == [], "select must NOT move motors"
    sc.selection_changed_signal.emit.assert_called_once_with(1, 1.0, 1.0)
    # clamp out-of-range
    SystemController.select_path_index(sc, 99)
    assert sc._raster_selected_index == len(pts) - 1
    SystemController.select_path_index(sc, -5)
    assert sc._raster_selected_index == 0


def test_select_nearest_point_argmin_ties_lowest():
    pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 0.0)]  # two equally-near to (10,1)
    sc = _select_self(pts)
    xy = SystemController.select_nearest_path_point(sc, 10.0, 1.0)
    assert xy == (10.0, 0.0)
    assert sc._raster_selected_index == 1, "ties resolve to the lowest index"
    assert sc._moves == []


def test_goto_path_index_sets_cursor_to_n_plus_1_and_moves():
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    sc = _select_self(pts)
    ok = SystemController.request_go_to_path_index(sc, 1, source="ui")
    assert ok is True
    assert sc._raster_index == 2, "next Step/Continuous resumes AFTER the site"
    assert sc._raster_selected_index == 1
    assert len(sc._moves) == 1
    x, y, _ = sc._moves[0]
    assert (x, y) == (1.0, 1.0)


def test_goto_takes_local_control_and_blacs_reclaims_on_next_step():
    """Go-to-site is the deliberate override: on a BLACS-owned raster it TAKES
    local control (where a local Step is refused outright), parks the cursor
    after the visited site, and BLACS's next move_to_next reclaims it there."""
    sc = _step_self([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    sc._raster_source = "remote"
    assert SystemController.request_go_to_path_index(sc, 1, source="ui") is True
    assert sc._raster_source == "local"
    assert sc._raster_index == 2
    assert sc._moves == [(1.0, 1.0, {"source": "ui"})]
    sc.raster_source_signal.emit.assert_called_once_with("local")

    SystemController.raster_step(sc, source="zmq", wait=False)
    assert sc._raster_source == "remote"
    _, _, cmd = sc._q.get_nowait()
    assert cmd.payload["target_xy"] == (2.0, 2.0), "resumes AFTER the visited site"


def test_raster_point_meta_describes_the_point_just_stepped_to():
    """Rides on the move_to_next reply -> BLACS writes it into the shot h5, so
    it must name the point just consumed and carry the calibration that makes
    its target coords meaningful."""
    sc = _step_self([(1.0, 2.0), (3.0, 4.0)])
    sc.calibration = None
    res = SystemController.raster_step(sc, source="zmq", wait=False)
    assert SystemController.raster_point_meta(sc, res) == {
        "point_index": 0, "path_len": 2, "target_xy": [1.0, 2.0]}

    sc.calibration = AffineCalibration([[2.0, 0.0], [0.0, 2.0]], [10.0, 20.0])
    meta = SystemController.raster_point_meta(sc)
    assert meta["calibration_matrix"] == [[2.0, 0.0], [0.0, 2.0]]
    assert meta["calibration_offset"] == [10.0, 20.0]


def test_point_meta_is_server_authoritative_across_a_goto_takeover():
    """An operator goto moves the cursor mid-run, so BLACS counting shots
    locally would mislabel every later shot -- the index has to come from here.
    The step's own MotorResult wins over the path lookup: it is the point the
    motors were actually commanded to."""
    sc = _step_self([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    sc.calibration = None
    SystemController.raster_step(sc, source="zmq", wait=False)
    assert SystemController.raster_point_meta(sc)["point_index"] == 0

    SystemController.request_go_to_path_index(sc, 2, source="ui")   # operator jumps
    SystemController.raster_step(sc, source="zmq", wait=False)      # BLACS resumes
    meta = SystemController.raster_point_meta(sc)
    assert meta["point_index"] == 3 and meta["target_xy"] == [3.0, 3.0]

    res = MotorResult(ok=True, target_xy=(9.9, 8.8))
    assert SystemController.raster_point_meta(sc, res)["target_xy"] == [9.9, 8.8]


def test_is_continuous_is_the_goto_gate():
    """Load-bearing since the UI stopped gating goto on its local Continuous
    checkbox (which drifts when BLACS re-arms an already-armed raster)."""
    fget = SystemController.is_continuous.fget
    assert fget(_select_self([], active=True, continuous=True)) is True
    assert fget(_select_self([], active=True, continuous=False)) is False
    assert fget(_select_self([], active=False, continuous=True)) is False


def test_goto_enqueues_move_target_tag_not_raster_step():
    """Invariant: goto routes through request_move_target (tag move_target), so
    _on_command_done -- which only chains on tag raster_step -- never treats a
    goto as a continuous step nor bumps _raster_step_count."""
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    sc = types.SimpleNamespace(
        _state_lock=threading.RLock(),
        _raster_path_pts=list(pts), _raster_index=0, _raster_selected_index=-1,
        _raster_active=True, _raster_continuous=False, _raster_source=None,
        selection_changed_signal=mock.Mock(), raster_source_signal=mock.Mock(),
        _q=queue.PriorityQueue(), _q_seq=itertools.count(),
    )
    sc.request_move_target = lambda x, y, **kw: SystemController.request_move_target(sc, x, y, **kw)
    sc._enqueue = lambda cmd: SystemController._enqueue(sc, cmd)
    ok = SystemController.request_go_to_path_index(sc, 1, source="ui")
    assert ok is True
    assert sc._q.qsize() == 1
    _, _, cmd = sc._q.get_nowait()
    assert cmd.cmd_type == CommandType.MOVE_TARGET
    assert cmd.tag == "move_target", "goto MUST use move_target, not raster_step"
    assert cmd.payload["target_xy"] == (1.0, 1.0)


def test_goto_rejected_in_continuous():
    pts = [(0.0, 0.0), (1.0, 1.0)]
    sc = _select_self(pts, active=True, continuous=True)
    ok = SystemController.request_go_to_path_index(sc, 1, source="ui")
    assert ok is False
    assert sc._moves == [], "goto must not fight the continuous run-loop"


def test_goto_no_path_rejected():
    sc = _select_self([])
    ok = SystemController.request_go_to_path_index(sc, 0, source="ui")
    assert ok is False
    assert sc._moves == []


# ----------------------------------------------------------------------------
# U2 -- Display Bounds enforcement (set_target_bounds setter + existing _execute
# enforcement of target_bounds on the calibrated MOVE_TARGET path)
# ----------------------------------------------------------------------------

def test_set_and_clear_target_bounds():
    sc = types.SimpleNamespace(_state_lock=threading.RLock(), target_bounds=None)
    SystemController.set_target_bounds(sc, (0.0, 1.0, 0.0, 1.0))
    assert sc.target_bounds == (0.0, 1.0, 0.0, 1.0)
    SystemController.clear_target_bounds(sc)
    assert sc.target_bounds is None


class _IdentityCal:
    def target_to_motor(self, x, y):
        return (x, y)

    def motor_to_target(self, x, y):
        return (x, y)


class _RecordMotor:
    def __init__(self):
        self.moves = []

    def get_position(self):
        return 0.0

    def move_to(self, v):
        self.moves.append(float(v))


def _execute_self(target_bounds):
    sc = types.SimpleNamespace(
        _state_lock=threading.RLock(),
        calibration=_IdentityCal(),
        target_bounds=target_bounds,
        motor_bounds=None,
        _last_target_xy=None,
        _last_motor_xy=None,
        motor_x=_RecordMotor(),
        motor_y=_RecordMotor(),
        _raster_log=[],
    )
    sc._within_bounds = lambda xy, b: SystemController._within_bounds(sc, xy, b)
    return sc


def test_calibrated_move_rejected_outside_target_bounds():
    """Once target_bounds is set (what set_target_bounds + Display Bounds do),
    a calibrated MOVE_TARGET outside the box is rejected and never moves the
    motors -- this is the Display-Bounds enforcement."""
    sc = _execute_self((0.0, 1.0, 0.0, 1.0))
    cmd = MotorCommand(cmd_type=CommandType.MOVE_TARGET,
                       payload={"target_xy": (5.0, 5.0)}, tag="move_target")
    res = SystemController._execute(sc, cmd)
    assert res.ok is False
    assert "out of bounds" in res.message
    assert sc.motor_x.moves == [] and sc.motor_y.moves == []


def test_calibrated_move_inside_target_bounds_passes():
    sc = _execute_self((0.0, 1.0, 0.0, 1.0))
    cmd = MotorCommand(cmd_type=CommandType.MOVE_TARGET,
                       payload={"target_xy": (0.5, 0.5)}, tag="move_target")
    res = SystemController._execute(sc, cmd)
    assert res.ok is True
    assert sc.motor_x.moves == [0.5]


def test_single_axis_move_uses_live_partner_not_cache():
    """MOVE_X_ONLY pairs the new x with the freshly-read y, never the cached
    last-commanded target: target->motor is cross-coupled, so a stale partner
    silently recomputes -- and physically moves -- the y motor on an "x-only"
    move (2026-08-04 first-move bug)."""
    sc = _execute_self(None)
    sc._last_target_xy = (5.0, 5.0)   # stale cache; live motors read (0, 0)
    cmd = MotorCommand(cmd_type=CommandType.MOVE_X_ONLY,
                       payload={"x": 0.7}, tag="move_x")
    res = SystemController._execute(sc, cmd)
    assert res.ok is True
    assert res.target_xy == (0.7, 0.0)   # live y, not the stale 5.0
    assert sc.motor_y.moves == [0.0]     # y stays put instead of jumping to 5


def test_edge_clamp_snaps_submicron_violation_onto_travel_floor():
    """A coordinate that maps a hair below motor zero -- e.g. replaying a
    position MEASURED at the travel floor after the partner axis moved
    (affine cross-terms, <=~5 um) -- snaps onto the floor and the move
    succeeds (2026-08-04 incident class)."""
    sc = _execute_self(None)
    sc.motor_bounds = (0.0, 12.0, 0.0, 12.0)
    cmd = MotorCommand(cmd_type=CommandType.MOVE_TARGET,
                       payload={"target_xy": (-0.0012, 5.0)}, tag="move_target")
    res = SystemController._execute(sc, cmd)
    assert res.ok is True
    assert sc.motor_x.moves == [0.0]
    assert sc.motor_y.moves == [5.0]


def test_real_bounds_violation_still_rejected():
    """Beyond the clamp tolerance, motor_bounds rejects with the typed
    message instead of letting Kinesis throw a .NET exception."""
    sc = _execute_self(None)
    sc.motor_bounds = (0.0, 12.0, 0.0, 12.0)
    cmd = MotorCommand(cmd_type=CommandType.MOVE_TARGET,
                       payload={"target_xy": (-0.5, 5.0)}, tag="move_target")
    res = SystemController._execute(sc, cmd)
    assert res.ok is False
    assert "out of bounds" in res.message
    assert sc.motor_x.moves == [] and sc.motor_y.moves == []


def test_clamp_to_bounds_passthrough_cases():
    """In-bounds, far-out, and bounds=None all pass through unchanged."""
    b = (0.0, 12.0, 0.0, 12.0)
    assert clamp_to_bounds((5.0, 5.0), b) == (5.0, 5.0)
    assert clamp_to_bounds((-0.5, 5.0), b) == (-0.5, 5.0)
    assert clamp_to_bounds((-0.001, 12.005), b) == (0.0, 12.0)
    assert clamp_to_bounds((-0.001, 5.0), None) == (-0.001, 5.0)


def test_jog_target_accumulates_from_live_position():
    """JOG_TARGET offsets from the freshly-read position, same fresh-partner
    rule as the single-axis moves."""
    sc = _execute_self(None)
    sc._last_target_xy = (5.0, 5.0)
    cmd = MotorCommand(cmd_type=CommandType.JOG_TARGET,
                       payload={"delta_xy": (0.25, -0.25)}, tag="jog")
    res = SystemController._execute(sc, cmd)
    assert res.ok is True
    assert res.target_xy == (0.25, -0.25)   # live (0,0) + delta


def test_from_json_rejects_pointer_file_with_clear_error():
    """Browsing to the breadcrumb pointer file (last_calibration_state.json,
    which has only 'last_calibration_path') must raise a descriptive ValueError,
    not a bare KeyError on 'calibration_matrix'."""
    try:
        AffineCalibration.from_json({"last_calibration_path": "C:/whatever.json"})
    except ValueError as e:
        assert "pointer file" in str(e).lower() or "not a calibration bundle" in str(e).lower()
    except KeyError:
        raise AssertionError("must raise a descriptive ValueError, not bare KeyError")
    else:
        raise AssertionError("expected ValueError")


def test_from_json_valid_roundtrips():
    cal = AffineCalibration.from_json({"calibration_matrix": [[1, 0], [0, 1]], "calibration_offset": [0, 0]})
    assert cal.target_to_motor(2.0, 3.0) == (2.0, 3.0)


def test_hull_bounds_none_fills_its_own_region():
    """A hull clicked anywhere (here far from the origin) must fill its OWN
    bounding box -- the spec-builder now passes bounds=None for hull."""
    spec = RasterSpec(kind="hull", bounds=None, xstep=2.0, ystep=2.0,
                      hull_points=[(100, 100), (300, 100), (200, 300)])
    pts = list(iter_path_from_spec(spec))
    assert len(pts) > 10  # grid spans the hull bbox, not a tiny corner


def test_hull_with_disjoint_scan_bounds_is_empty_regression():
    """Documents the OLD bug: a far-away scan-bounds box clipped the hull away."""
    spec = RasterSpec(kind="hull", bounds=(0, 6, 0, 6), xstep=2.0, ystep=2.0,
                      hull_points=[(100, 100), (300, 100), (200, 300)])
    assert list(iter_path_from_spec(spec)) == []


def test_hull_too_fine_grid_raises_fast_not_freeze():
    """A frame-spanning hull at the sub-pixel default step must raise a clear
    error IMMEDIATELY (budget guard) rather than freeze for ~minutes (the HIGH
    regression from the pre-merge review)."""
    spec = RasterSpec(kind="hull", bounds=None, xstep=0.01, ystep=0.01,
                      hull_points=[(10, 10), (490, 20), (250, 480)])
    try:
        list(iter_path_from_spec(spec))
    except ValueError as e:
        assert "too fine" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for the pathological grid")


def test_hull_degenerate_collinear_raises_value_error():
    """A collinear/degenerate hull raises a clean ValueError (caught by the
    preview/start-raster try/except) instead of an opaque scipy QhullError."""
    spec = RasterSpec(kind="hull", bounds=None, xstep=1.0, ystep=1.0,
                      hull_points=[(0, 0), (1, 1), (2, 2)])
    try:
        list(iter_path_from_spec(spec))
    except ValueError as e:
        assert "degenerate" in str(e).lower()
    else:
        raise AssertionError("expected ValueError for the collinear hull")


def test_user_defaults_roundtrip():
    """save_user_defaults -> load_user_defaults round-trips; absent file -> None.
    Uses a throwaway path so the operator's real settings_defaults.json is safe."""
    import raster_controller as rc
    orig = rc.USER_DEFAULTS_FILE
    rc.USER_DEFAULTS_FILE = os.path.join(os.path.dirname(orig), "_test_settings_defaults.json")
    try:
        assert rc.load_user_defaults() is None
        rc.save_user_defaults({"jog_step": {"x": 0.5, "y": 0.7}})
        got = rc.load_user_defaults()
        assert got["jog_step"]["x"] == 0.5 and got["jog_step"]["y"] == 0.7
    finally:
        if os.path.exists(rc.USER_DEFAULTS_FILE):
            os.remove(rc.USER_DEFAULTS_FILE)
        rc.USER_DEFAULTS_FILE = orig


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except BaseException as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failures else 0)
