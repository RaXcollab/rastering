"""
controller.py

Core coordinator for the kinematic mirror / beam steering app.

Design goals:
- UI thread only renders and emits user intent.
- Motor DLL access is SINGLE-OWNED by a dedicated motor I/O thread.
- Network (ZMQ) is "stupid": parse -> controller.request_* -> reply.
- Calibration is explicit: affine transform between target-space (plot coords) and motor-space.
- Rastering is a state machine in the controller, driven by UI or ZMQ commands.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import numpy as np
import zmq
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer


# -----------------------------
# Types / dataclasses
# -----------------------------

TargetXY = Tuple[float, float]   # "target space": the coordinates the user clicks in the plot (same units as plot axes)
MotorXY = Tuple[float, float]    # motor device units (whatever Motor.get_position / move_to uses)


class CommandType(Enum):
    MOVE_TARGET = auto()     # payload: {"target_xy": (x, y)}
    MOVE_X_ONLY = auto()     # payload: {"x": float}   (y taken from cached target pos)
    MOVE_Y_ONLY = auto()     # payload: {"y": float}   (x taken from cached target pos)
    JOG_TARGET = auto()      # payload: {"delta_xy": (dx, dy)} (adds to cached target pos)

    READ_POS = auto()        # no payload
    STOP = auto()            # payload: {"reason": str}

    HOME_SOFT_X = auto()
    HOME_SOFT_Y = auto()
    HOME_HARD_X = auto()
    HOME_HARD_Y = auto()

    SET_BACKLASH_X = auto()  # payload: {"value": float}
    SET_BACKLASH_Y = auto()  # payload: {"value": float}

    NOOP = auto()


@dataclass
class MotorResult:
    ok: bool
    message: str = ""
    cmd_id: str = ""
    source: str = "ui"
    tag: str = ""

    target_xy: Optional[TargetXY] = None
    motor_xy: Optional[MotorXY] = None
    ts: float = field(default_factory=time.time)


@dataclass
class MotorCommand:
    cmd_type: CommandType
    payload: Dict[str, Any] = field(default_factory=dict)

    # metadata
    cmd_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_ts: float = field(default_factory=time.time)
    source: str = "ui"       # "ui" | "zmq" | "raster" | "internal"
    tag: str = ""            # e.g., "raster_step" for chaining logic
    priority: int = 100      # lower = higher priority (STOP should be 0)

    # synchronous reply channel (optional)
    reply_q: Optional["queue.Queue[MotorResult]"] = None


@dataclass
class AffineCalibration:
    """
    motor_xy = M @ target_xy + b
    target_xy = inv(M) @ (motor_xy - b)
    """
    M: np.ndarray  # shape (2,2)
    b: np.ndarray  # shape (2,)

    def __post_init__(self) -> None:
        self.M = np.array(self.M, dtype=float).reshape(2, 2)
        self.b = np.array(self.b, dtype=float).reshape(2,)
        # cache inverse (may raise if singular)
        self._Minv = np.linalg.inv(self.M)

    def target_to_motor(self, x: float, y: float) -> MotorXY:
        v = self.M @ np.array([x, y], dtype=float) + self.b
        return float(v[0]), float(v[1])

    def motor_to_target(self, mx: float, my: float) -> TargetXY:
        v = self._Minv @ (np.array([mx, my], dtype=float) - self.b)
        return float(v[0]), float(v[1])

    def to_json(self) -> Dict[str, Any]:
        return {"calibration_matrix": self.M.tolist(), "calibration_offset": self.b.tolist()}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "AffineCalibration":
        return AffineCalibration(M=np.array(data["calibration_matrix"]), b=np.array(data["calibration_offset"]))


@dataclass
class CalibrationSession:
    required_points: int = 3
    target_points: list[TargetXY] = field(default_factory=list)
    motor_points: list[MotorXY] = field(default_factory=list)

    def add_pair(self, target_xy: TargetXY, motor_xy: MotorXY) -> None:
        self.target_points.append((float(target_xy[0]), float(target_xy[1])))
        self.motor_points.append((float(motor_xy[0]), float(motor_xy[1])))

    @property
    def n(self) -> int:
        return len(self.target_points)

    @property
    def is_ready(self) -> bool:
        return self.n >= self.required_points

    def fit_affine(self) -> Tuple[AffineCalibration, Dict[str, Any]]:
        """
        Fits affine mapping:
            [mx]   [a b][x] + [tx]
            [my] = [c d][y]   [ty]
        using least squares with N>=3.

        Returns (AffineCalibration, diagnostics)
        """
        if self.n < 3:
            raise ValueError("Need at least 3 calibration pairs")

        # Build A matrix as in your original code:
        # for each point (x,y) -> rows [x y 1 0 0 0] and [0 0 0 x y 1]
        A = np.zeros((2 * self.n, 6), dtype=float)
        b = np.zeros((2 * self.n,), dtype=float)

        for i, ((x, y), (mx, my)) in enumerate(zip(self.target_points, self.motor_points)):
            A[2 * i, 0:3] = [x, y, 1.0]
            A[2 * i + 1, 3:6] = [x, y, 1.0]
            b[2 * i] = mx
            b[2 * i + 1] = my

        params, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)

        # Extract affine components
        M = np.array([[params[0], params[1]],
                      [params[3], params[4]]], dtype=float)
        off = np.array([params[2], params[5]], dtype=float)

        # Diagnostics
        diag: Dict[str, Any] = {}
        diag["rank"] = int(rank)
        diag["singular_values"] = [float(v) for v in svals]
        diag["residuals"] = [float(r) for r in (residuals if np.ndim(residuals) else [])]

        # Condition number-ish for A (helps warn about degeneracy)
        if len(svals) > 0 and np.min(svals) > 0:
            diag["cond_A"] = float(np.max(svals) / np.min(svals))
        else:
            diag["cond_A"] = float("inf")

        # Additional geometric degeneracy check: area of triangle for first 3 points
        # (useful when required_points==3)
        if self.n >= 3:
            (x1, y1), (x2, y2), (x3, y3) = self.target_points[0], self.target_points[1], self.target_points[2]
            area2 = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            diag["triangle_area2_first3"] = float(area2)

        cal = AffineCalibration(M=M, b=off)  # may raise if singular
        return cal, diag


# -----------------------------
# Controller
# -----------------------------

class SystemController(QObject):
    """
    Coordinates motors, calibration, raster sessions, and ZMQ requests.

    UI and ZMQ should only call request_* methods. They are safe from any thread.
    """

    # --- signals to UI ---
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    # Positions (in target space and motor space)
    target_position_signal = pyqtSignal(float, float)
    motor_position_signal = pyqtSignal(float, float)

    # Command completion (useful for UI + raster chaining)
    command_done_signal = pyqtSignal(str, bool, str, str)  # cmd_id, ok, message, tag

    # Calibration lifecycle
    calibration_prompt_signal = pyqtSignal(str)
    calibration_progress_signal = pyqtSignal(int, int)  # collected, required
    calibration_ready_signal = pyqtSignal(object)       # AffineCalibration
    calibration_failed_signal = pyqtSignal(str)

    # Raster lifecycle
    raster_state_signal = pyqtSignal(bool)              # active?
    raster_finished_signal = pyqtSignal()
    raster_log_path_signal = pyqtSignal(str)

    def __init__(
        self,
        motor_x: Any,
        motor_y: Any,
        *,
        calibration_path: str = "calibration_data.json",
        target_bounds: Optional[Tuple[float, float, float, float]] = None,
        motor_bounds: Optional[Tuple[float, float, float, float]] = None,
        telemetry_period_s: float = 0.2,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)

        self.motor_x = motor_x
        self.motor_y = motor_y

        self._state_lock = threading.RLock()

        # Calibration
        self.calibration_path = calibration_path
        self.calibration: Optional[AffineCalibration] = None
        self._cal_session: Optional[CalibrationSession] = None

        # Bounds
        self.target_bounds = target_bounds   # xmin, xmax, ymin, ymax in target space
        self.motor_bounds = motor_bounds     # xmin, xmax, ymin, ymax in motor units

        # Cached positions
        self._last_motor_xy: Optional[MotorXY] = None
        self._last_target_xy: Optional[TargetXY] = None

        # Command queue + worker thread
        self._q: "queue.PriorityQueue[Tuple[int, float, MotorCommand]]" = queue.PriorityQueue()
        self._stop_evt = threading.Event()
        self._motor_thread = threading.Thread(target=self._motor_worker_loop, name="motor-io", daemon=True)
        self._motor_thread.start()

        # Raster state
        self._raster_iter: Optional[Iterator[TargetXY]] = None
        self._raster_active: bool = False
        self._raster_continuous: bool = False
        self._raster_delay_s = 0.0
        self._raster_log: list[Dict[str, Any]] = []
        self._raster_log_path: Optional[str] = None

        # Telemetry polling (via READ_POS commands, so it never touches DLL outside motor thread)
        self._telemetry_period_s = float(telemetry_period_s)
        self._telemetry_enabled = self._telemetry_period_s > 0
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, name="telemetry", daemon=True)
        if self._telemetry_enabled:
            self._telemetry_thread.start()

        # ZMQ server
        self._zmq_thread: Optional[threading.Thread] = None
        self._zmq_stop_evt = threading.Event()

        # Raster chaining: advance on completion of raster-tagged move
        self.command_done_signal.connect(self._on_command_done)

    # -------------------------
    # Public API (UI / ZMQ)
    # -------------------------

    def request_move_target(self, x: float, y: float, *, source: str = "ui", wait: bool = False, timeout_s: float = 10.0) -> Optional[MotorResult]:
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(
            cmd_type=CommandType.MOVE_TARGET,
            payload={"target_xy": (float(x), float(y))},
            source=source,
            tag="move_target",
            reply_q=reply_q,
        )
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None

    def request_move_x(self, x: float, *, source: str = "ui", wait: bool = False, timeout_s: float = 10.0) -> Optional[MotorResult]:
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(
            cmd_type=CommandType.MOVE_X_ONLY,
            payload={"x": float(x)},
            source=source,
            tag="move_x",
            reply_q=reply_q,
        )
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None

    def request_move_y(self, y: float, *, source: str = "ui", wait: bool = False, timeout_s: float = 10.0) -> Optional[MotorResult]:
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(
            cmd_type=CommandType.MOVE_Y_ONLY,
            payload={"y": float(y)},
            source=source,
            tag="move_y",
            reply_q=reply_q,
        )
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None

    def request_jog_target(self, dx: float, dy: float, *, source: str = "ui", wait: bool = False, timeout_s: float = 10.0) -> Optional[MotorResult]:
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(
            cmd_type=CommandType.JOG_TARGET,
            payload={"delta_xy": (float(dx), float(dy))},
            source=source,
            tag="jog",
            reply_q=reply_q,
        )
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None

    def request_stop(self, *, reason: str = "user") -> None:
        cmd = MotorCommand(cmd_type=CommandType.STOP, payload={"reason": reason}, source="ui", tag="stop", priority=0)
        self._enqueue(cmd)

    def request_home(self, axis: str, *, hard: bool = False, source: str = "ui", wait: bool = False, timeout_s: float = 30.0) -> Optional[MotorResult]:
        axis = axis.upper().strip()
        if axis not in ("X", "Y"):
            self.error_signal.emit("Home request rejected: axis must be 'X' or 'Y'")
            return None
        cmd_type = {
            ("X", False): CommandType.HOME_SOFT_X,
            ("Y", False): CommandType.HOME_SOFT_Y,
            ("X", True): CommandType.HOME_HARD_X,
            ("Y", True): CommandType.HOME_HARD_Y,
        }[(axis, hard)]
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(cmd_type=cmd_type, payload={}, source=source, tag=f"home_{axis}_{'hard' if hard else 'soft'}", reply_q=reply_q)
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None

    def request_set_backlash(self, axis: str, value: float, *, source: str = "ui", wait: bool = False, timeout_s: float = 5.0) -> Optional[MotorResult]:
        axis = axis.upper().strip()
        if axis not in ("X", "Y"):
            self.error_signal.emit("Backlash request rejected: axis must be 'X' or 'Y'")
            return None
        cmd_type = CommandType.SET_BACKLASH_X if axis == "X" else CommandType.SET_BACKLASH_Y
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(cmd_type=cmd_type, payload={"value": float(value)}, source=source, tag=f"backlash_{axis}", reply_q=reply_q)
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None
    
    def request_pos(self, *, source: str = "ui", wait: bool = False, timeout_s: float = 2.0) -> Optional[MotorResult]:
        """
        Request a fresh motor position read.

        If wait=True, blocks until the motor thread returns the readback (safe for calibration).
        """
        reply_q = queue.Queue(maxsize=1) if wait else None
        cmd = MotorCommand(
            cmd_type=CommandType.READ_POS,
            payload={},
            source=source,
            tag="read_pos",
            priority=50,   # higher priority than telemetry (but below STOP)
            reply_q=reply_q,
        )
        self._enqueue(cmd)
        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None


    # --- calibration control ---

    def start_calibration(self, required_points: int = 3) -> None:
        with self._state_lock:
            self._cal_session = CalibrationSession(required_points=int(required_points))
        self.calibration_prompt_signal.emit(
            f"Calibration: jog the beam to {required_points} distinct spots; click the laser spot each time."
        )
        self.calibration_progress_signal.emit(0, int(required_points))

    def cancel_calibration(self) -> None:
        with self._state_lock:
            self._cal_session = None
        self.calibration_prompt_signal.emit("Calibration cancelled.")

    def add_calibration_click(self, x: float, y: float) -> None:
        """
        Called by UI when the user clicks the laser spot during calibration.

        We MUST pair this click with a fresh motor readback to avoid telemetry race conditions.
        """
        with self._state_lock:
            sess = self._cal_session

        if sess is None:
            return

        # Optional guard: if your motor class exposes is_moving(), reject mid-motion clicks
        try:
            if hasattr(self.motor_x, "is_moving") and self.motor_x.is_moving():
                self.calibration_failed_signal.emit("Wait for motion to stop, then click the laser spot.")
                return
            if hasattr(self.motor_y, "is_moving") and self.motor_y.is_moving():
                self.calibration_failed_signal.emit("Wait for motion to stop, then click the laser spot.")
                return
        except Exception:
            pass

        # Force a fresh read from motor thread
        res = self.request_pos(source="internal", wait=True, timeout_s=2.0)
        if not res or (not res.ok) or (res.motor_xy is None):
            self.calibration_failed_signal.emit("Could not read motor position for calibration point.")
            return

        motor_xy = res.motor_xy
        sess.add_pair((float(x), float(y)), motor_xy)

        self.calibration_progress_signal.emit(sess.n, sess.required_points)

        if sess.is_ready:
            self._finish_calibration()


    def _finish_calibration(self) -> None:
        with self._state_lock:
            sess = self._cal_session
        if sess is None:
            return

        try:
            cal, diag = sess.fit_affine()
        except Exception as e:
            self.calibration_failed_signal.emit(f"Calibration failed: {e}")
            return

        # Basic degeneracy guard (first 3 points collinear -> tiny area)
        area2 = diag.get("triangle_area2_first3", None)
        if area2 is not None and area2 < 1e-6:
            self.calibration_failed_signal.emit("Calibration failed: first 3 points nearly collinear. Choose a triangle.")
            return

        self.set_calibration(cal)
        self.save_calibration()

        with self._state_lock:
            self._cal_session = None

        self.calibration_ready_signal.emit(cal)
        self.status_signal.emit(f"Calibration complete. cond(A)~{diag.get('cond_A', 'n/a')}")

    def set_calibration(self, cal: AffineCalibration) -> None:
        with self._state_lock:
            self.calibration = cal
        self.status_signal.emit("Calibration set.")

        # Update cached target position if possible
        with self._state_lock:
            lm = self._last_motor_xy
            c = self.calibration
        if lm is not None and c is not None:
            tx, ty = c.motor_to_target(lm[0], lm[1])
            with self._state_lock:
                self._last_target_xy = (tx, ty)
            self.target_position_signal.emit(tx, ty)

    def clear_calibration(self) -> None:
        with self._state_lock:
            self.calibration = None
        self.status_signal.emit("Calibration cleared.")

    def save_calibration(self) -> None:
        with self._state_lock:
            cal = self.calibration
        if cal is None:
            self.error_signal.emit("No calibration to save.")
            return
        try:
            with open(self.calibration_path, "w") as f:
                json.dump(cal.to_json(), f, indent=2)
            self.status_signal.emit(f"Calibration saved to {self.calibration_path}")
        except Exception as e:
            self.error_signal.emit(f"Failed to save calibration: {e}")

    def load_calibration(self) -> None:
        if not os.path.exists(self.calibration_path):
            self.error_signal.emit(f"No calibration file found: {self.calibration_path}")
            return
        try:
            with open(self.calibration_path, "r") as f:
                data = json.load(f)
            cal = AffineCalibration.from_json(data)
            self.set_calibration(cal)
            self.status_signal.emit(f"Loaded calibration from {self.calibration_path}")
        except Exception as e:
            self.error_signal.emit(f"Failed to load calibration: {e}")

    # --- raster control ---

    def start_raster(self, path_iter, *, continuous: bool = True, log_dir: str | None = None, delay_s: float = 0.0) -> None:
        """
        Start rastering along a target-space path (iterable of (x,y) points).
        If continuous=False, raster is "armed" and only advances when raster_step() is called.
        """
        with self._state_lock:
            self._raster_iter = iter(path_iter)
            self._raster_active = True
            self._raster_continuous = bool(continuous)
            self._raster_delay_s = float(delay_s) if continuous else 0.0
            self._raster_log = []
            self._raster_log_path = None

        if log_dir is None:
            log_dir = os.getcwd()
        os.makedirs(log_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._raster_log_path = os.path.join(log_dir, f"raster_log_{ts}.json")
        self.raster_log_path_signal.emit(self._raster_log_path)

        self.raster_state_signal.emit(True)
        self.status_signal.emit("Raster started." if continuous else "Raster armed (step mode).")

        if continuous:
            self._enqueue_next_raster_point()


    def raster_step(self, *, source: str = "ui", wait: bool = False, timeout_s: float = 10.0) -> Optional[MotorResult]:
        """
        Advance exactly one raster step.

        - In step mode (continuous=False): this is the only way the raster advances.
        - In continuous mode: we *reject* explicit stepping to avoid double-driving.
        """
        with self._state_lock:
            active = self._raster_active
            continuous = self._raster_continuous
            it = self._raster_iter

        if not active or it is None:
            self.error_signal.emit("Raster step requested but raster is not active.")
            return None

        if continuous:
            self.error_signal.emit("Raster is running in continuous mode; step is disabled.")
            return None

        try:
            x, y = next(it)
        except StopIteration:
            self._finish_raster()
            return None

        reply_q = queue.Queue(maxsize=1) if wait else None

        cmd = MotorCommand(
            cmd_type=CommandType.MOVE_TARGET,
            payload={"target_xy": (float(x), float(y))},
            source=source,            # keep initiator ("ui" or "zmq")
            tag="raster_step",        # critical
            priority=100,
            reply_q=reply_q,
        )
        self._enqueue(cmd)

        if wait:
            return self._wait_reply(reply_q, cmd.cmd_id, timeout_s)
        return None



    def stop_raster(self) -> None:
        with self._state_lock:
            was_active = self._raster_active
            self._raster_active = False
            self._raster_iter = None
            self._raster_continuous = False

        if was_active:
            self.status_signal.emit("Raster stopped.")
            self.raster_state_signal.emit(False)
            self._flush_raster_log()

    # --- ZMQ server (compatibility with existing commands) ---

    def start_zmq_server(self, bind: str = "tcp://*:55535") -> None:
        if self._zmq_thread and self._zmq_thread.is_alive():
            self.status_signal.emit("ZMQ server already running.")
            return
        self._zmq_stop_evt.clear()
        self._zmq_thread = threading.Thread(target=self._zmq_loop, args=(bind,), name="zmq-server", daemon=True)
        self._zmq_thread.start()
        self.status_signal.emit(f"ZMQ server bound on {bind}")

    def stop_zmq_server(self) -> None:
        self._zmq_stop_evt.set()
        self.status_signal.emit("ZMQ server stopping...")

    # -------------------------
    # Internal helpers
    # -------------------------

    def _wait_reply(self, reply_q: "queue.Queue[MotorResult]", cmd_id: str, timeout_s: float) -> MotorResult:
        try:
            return reply_q.get(timeout=float(timeout_s))
        except queue.Empty:
            return MotorResult(ok=False, message="timeout waiting for result", cmd_id=cmd_id)

    def _enqueue(self, cmd: MotorCommand) -> None:
        self._q.put((cmd.priority, cmd.created_ts, cmd))

    def _within_bounds(self, xy: Tuple[float, float], bounds: Optional[Tuple[float, float, float, float]]) -> bool:
        if bounds is None:
            return True
        x, y = xy
        xmin, xmax, ymin, ymax = bounds
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

    def _telemetry_loop(self) -> None:
        """
        Periodically request READ_POS. This keeps UI labels current without
        any thread other than motor-io calling into the DLL.
        """
        while not self._stop_evt.is_set():
            time.sleep(self._telemetry_period_s)
            cmd = MotorCommand(cmd_type=CommandType.READ_POS, source="internal", tag="telemetry", priority=200)
            self._enqueue(cmd)

    def _motor_worker_loop(self) -> None:
        """
        Single-owner motor I/O loop. This is the ONLY place motor DLL calls occur.
        """
        while not self._stop_evt.is_set():
            try:
                _, _, cmd = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            if cmd.cmd_type == CommandType.NOOP:
                continue

            if cmd.cmd_type == CommandType.STOP:
                res = self._execute_stop(cmd)
                self._deliver_result(cmd, res)
                continue

            try:
                res = self._execute(cmd)
            except Exception as e:
                res = MotorResult(ok=False, message=f"Unhandled motor error: {e}", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)

            self._deliver_result(cmd, res)

    def _execute_stop(self, cmd: MotorCommand) -> MotorResult:
        reason = str(cmd.payload.get("reason", ""))
        # Drain queue quickly so STOP takes effect
        self._drain_queue()
        # Attempt to stop motors if supported
        try:
            if hasattr(self.motor_x, "stop"):
                self.motor_x.stop()
            if hasattr(self.motor_y, "stop"):
                self.motor_y.stop()
            return MotorResult(ok=True, message=f"Stop executed ({reason})", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
        except Exception as e:
            return MotorResult(ok=False, message=f"Stop error: {e}", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)

    def _drain_queue(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            return

    def _execute(self, cmd: MotorCommand) -> MotorResult:
        # Helper: read motor positions (motor space)
        def read_motor_xy() -> MotorXY:
            mx = float(self.motor_x.get_position())
            my = float(self.motor_y.get_position())
            return mx, my

        if cmd.cmd_type == CommandType.READ_POS:
            mx, my = read_motor_xy()
            with self._state_lock:
                self._last_motor_xy = (mx, my)
                cal = self.calibration
            # Update target cache if possible
            if cal is not None:
                tx, ty = cal.motor_to_target(mx, my)
            else:
                # Uncalibrated passthrough: target space == motor space
                tx, ty = mx,my
            with self._state_lock:
                self._last_target_xy = (tx, ty)
            return MotorResult(
                ok=True,
                message="read_pos",
                cmd_id=cmd.cmd_id,
                source=cmd.source,
                tag=cmd.tag,
                motor_xy=(mx, my),
                target_xy=(tx, ty),
            )

        # Home / backlash do not require calibration
        if cmd.cmd_type in (CommandType.HOME_SOFT_X, CommandType.HOME_HARD_X):
            fn = getattr(self.motor_x, "hard_home" if cmd.cmd_type == CommandType.HOME_HARD_X else "soft_home", None)
            if fn is None:
                return MotorResult(ok=False, message="Motor X does not support home", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            fn()
            mx, my = read_motor_xy()
            with self._state_lock:
                self._last_motor_xy = (mx, my)
                cal = self.calibration
            if cal is not None:
                tx, ty = cal.motor_to_target(mx, my)
            else: # target space == motor space
                tx,ty = mx, my
            with self._state_lock:
                self._last_target_xy = (tx, ty)
            txy = (tx, ty)
            return MotorResult(ok=True, message="home_x complete", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag, motor_xy=(mx, my), target_xy=txy)

        if cmd.cmd_type in (CommandType.HOME_SOFT_Y, CommandType.HOME_HARD_Y):
            fn = getattr(self.motor_y, "hard_home" if cmd.cmd_type == CommandType.HOME_HARD_Y else "soft_home", None)
            if fn is None:
                return MotorResult(ok=False, message="Motor Y does not support home", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            fn()
            mx, my = read_motor_xy()
            with self._state_lock:
                self._last_motor_xy = (mx, my)
                cal = self.calibration
            if cal is not None:
                tx, ty = cal.motor_to_target(mx, my)
            else: # target space == motor space
                tx,ty = mx, my
            with self._state_lock:
                self._last_target_xy = (tx, ty)
            txy = (tx, ty)
            return MotorResult(ok=True, message="home_y complete", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag, motor_xy=(mx, my), target_xy=txy)

        if cmd.cmd_type in (CommandType.SET_BACKLASH_X, CommandType.SET_BACKLASH_Y):
            axis = "X" if cmd.cmd_type == CommandType.SET_BACKLASH_X else "Y"
            value = float(cmd.payload.get("value", 0.0))
            motor = self.motor_x if axis == "X" else self.motor_y
            fn = getattr(motor, "set_backlash", None)
            if fn is None:
                return MotorResult(ok=False, message=f"Motor {axis} does not support backlash", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            fn(value)
            return MotorResult(ok=True, message=f"backlash {axis} set to {value}", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
        
        # Motion commands (calibrated OR uncalibrated passthrough)
        with self._state_lock:
            cal = self.calibration
            last_target = self._last_target_xy
            last_motor = self._last_motor_xy

        # If we haven't gotten telemetry yet, fall back to motor cache
        if last_target is None and last_motor is not None:
            last_target = last_motor  # works for both modes (passthrough uses motor==target)

        # Determine target point
        if cmd.cmd_type == CommandType.MOVE_TARGET:
            tx, ty = cmd.payload["target_xy"]

        elif cmd.cmd_type == CommandType.MOVE_X_ONLY:
            if last_target is None:
                return MotorResult(ok=False, message="No cached target position for MOVE_X_ONLY", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            tx, ty = float(cmd.payload["x"]), float(last_target[1])

        elif cmd.cmd_type == CommandType.MOVE_Y_ONLY:
            if last_target is None:
                return MotorResult(ok=False, message="No cached target position for MOVE_Y_ONLY", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            tx, ty = float(last_target[0]), float(cmd.payload["y"])

        elif cmd.cmd_type == CommandType.JOG_TARGET:
            if last_target is None:
                return MotorResult(ok=False, message="No cached target position for JOG_TARGET", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)
            dx, dy = cmd.payload["delta_xy"]
            tx, ty = float(last_target[0]) + float(dx), float(last_target[1]) + float(dy)

        else:
            return MotorResult(ok=False, message=f"Unsupported command {cmd.cmd_type}", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag)

        target_xy = (float(tx), float(ty))

        if cal is None:
            # Uncalibrated passthrough: interpret target coords as motor coords
            motor_xy = target_xy

            # Do NOT apply target_bounds here (they're image/target-space bounds)
            if not self._within_bounds(motor_xy, self.motor_bounds):
                return MotorResult(
                    ok=False,
                    message="Rejected: motor out of bounds",
                    cmd_id=cmd.cmd_id,
                    source=cmd.source,
                    tag=cmd.tag,
                    target_xy=target_xy,
                    motor_xy=motor_xy,
                )
        else:
            # Calibrated: enforce target bounds, map, then enforce motor bounds
            if not self._within_bounds(target_xy, self.target_bounds):
                return MotorResult(
                    ok=False,
                    message="Rejected: target out of bounds",
                    cmd_id=cmd.cmd_id,
                    source=cmd.source,
                    tag=cmd.tag,
                    target_xy=target_xy,
                )

            mx, my = cal.target_to_motor(target_xy[0], target_xy[1])
            motor_xy = (float(mx), float(my))

            if not self._within_bounds(motor_xy, self.motor_bounds):
                return MotorResult(
                    ok=False,
                    message="Rejected: motor out of bounds",
                    cmd_id=cmd.cmd_id,
                    source=cmd.source,
                    tag=cmd.tag,
                    target_xy=target_xy,
                    motor_xy=motor_xy,
                )

        # Execute moves (sequential for now). Driver should include its own timeout.
        # NOTE: Your original RasterManager moves A then B. We keep that behavior for now.
        self.motor_x.move_to(motor_xy[0])
        self.motor_y.move_to(motor_xy[1])

        # Read back
        mx2, my2 = read_motor_xy()
        with self._state_lock:
            self._last_motor_xy = (mx2, my2)
            self._last_target_xy = target_xy

        # Raster logging (only for raster steps, regardless of who initiated them)
        if cmd.tag == "raster_step":
            self._raster_log.append({
                "timestamp": time.time(),
                "x": target_xy[0],
                "y": target_xy[1],
                "mx": mx2,
                "my": my2,
                "initiator": cmd.source,   # "ui" | "zmq" | "raster" | etc.
            })

        return MotorResult(ok=True, message="Move complete", cmd_id=cmd.cmd_id, source=cmd.source, tag=cmd.tag, target_xy=target_xy, motor_xy=(mx2, my2))

    def _deliver_result(self, cmd: MotorCommand, res: MotorResult) -> None:
        # Sync reply (ZMQ wait mode)
        if cmd.reply_q is not None:
            try:
                cmd.reply_q.put_nowait(res)
            except Exception:
                pass

        # Emit signals to UI (Qt will queue cross-thread automatically)
        self.command_done_signal.emit(res.cmd_id, res.ok, res.message, res.tag)

        if res.motor_xy is not None:
            self.motor_position_signal.emit(res.motor_xy[0], res.motor_xy[1])
        if res.target_xy is not None:
            self.target_position_signal.emit(res.target_xy[0], res.target_xy[1])

        if not res.ok and res.message:
            self.status_signal.emit(f"[{res.source}] {res.message}")

    # Raster chaining runs in Qt thread
    @pyqtSlot(str, bool, str, str)
    def _on_command_done(self, cmd_id: str, ok: bool, message: str, tag: str) -> None:
        # Only chain raster in continuous mode, and only after a raster step finishes.
        if tag != "raster_step":
            return

        with self._state_lock:
            active = self._raster_active
            continuous = self._raster_continuous
            delay_s = float(getattr(self, "_raster_delay_s", 0.0))

        if not active or not continuous:
            return

        if not ok:
            self.status_signal.emit("Raster halted due to an error.")
            self.stop_raster()
            return

        if delay_s > 0:
            QTimer.singleShot(int(delay_s * 1000), self._enqueue_next_raster_point)
        else:
            self._enqueue_next_raster_point()



    def _enqueue_next_raster_point(self) -> None:
        with self._state_lock:
            it = self._raster_iter
            active = self._raster_active
        if not active or it is None:
            return
        try:
            x, y = next(it)
        except StopIteration:
            self._finish_raster()
            return

        # Use source="raster" so it gets logged
        cmd = MotorCommand(
            cmd_type=CommandType.MOVE_TARGET,
            payload={"target_xy": (float(x), float(y))},
            source="raster",
            tag="raster_step",
            priority=100,
        )
        self._enqueue(cmd)

    def _finish_raster(self) -> None:
        with self._state_lock:
            self._raster_active = False
            self._raster_continuous = False
            self._raster_iter = None
        self.raster_state_signal.emit(False)
        self.raster_finished_signal.emit()
        self.status_signal.emit("Raster finished.")
        self._flush_raster_log()

    def _flush_raster_log(self) -> None:
        path = self._raster_log_path
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._raster_log, f, indent=2)
            self.status_signal.emit(f"Raster log saved: {path}")
        except Exception as e:
            self.error_signal.emit(f"Could not write raster log: {e}")
        finally:
            self._raster_log = []
            self._raster_log_path = None

    # -------------------------
    # ZMQ server implementation
    # -------------------------

    def _zmq_loop(self, bind: str) -> None:
        """
        Compatibility server with your existing JSON protocol:
        - action: PROGRAM_VALUE / CHECK_VALUE
        - connection names: laser_raster_x_coord, laser_raster_y_coord, monitors, arm_raster, disarm_raster, move_to_next
        """
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(bind)
        sock.RCVTIMEO = 250  # ms, so we can exit cleanly

        def reply(obj: Dict[str, Any]) -> None:
            sock.send(json.dumps(obj).encode())

        while not self._zmq_stop_evt.is_set():
            try:
                req = sock.recv()
            except zmq.error.Again:
                continue
            except Exception as e:
                # socket error; bail
                break

            try:
                data = json.loads(req.decode())
                action = data.get("action", "")
                connection = data.get("connection", "")
                value = data.get("value", None)
            except Exception:
                reply({"status": "ERROR", "message": "bad_json"})
                continue

            # CHECK_VALUE: return cached target coords (preferred) else motor coords
            if action == "CHECK_VALUE":
                with self._state_lock:
                    txy = self._last_target_xy
                    mxy = self._last_motor_xy
                if connection in ("laser_raster_x_coord_monitor", "laser_raster_x_coord"):
                    v = txy[0] if txy is not None else (mxy[0] if mxy is not None else None)
                elif connection in ("laser_raster_y_coord_monitor", "laser_raster_y_coord"):
                    v = txy[1] if txy is not None else (mxy[1] if mxy is not None else None)
                else:
                    v = None
                reply({"status": "SUCCESS", "value": v})
                continue

            if action != "PROGRAM_VALUE":
                reply({"status": "ERROR", "message": "unknown_action"})
                continue

            # PROGRAM_VALUE
            try:
                timeout_sec = float(data.get("timeout_sec", 10.0))
            except Exception:
                timeout_sec = 10.0

            if connection == "laser_raster_x_coord":
                res = self.request_move_x(float(value), source="zmq", wait=True, timeout_s=timeout_sec)
                reply({"status": "SUCCESS" if res and res.ok else "ERROR", "message": (res.message if res else "")})
                continue

            if connection == "laser_raster_y_coord":
                res = self.request_move_y(float(value), source="zmq", wait=True, timeout_s=timeout_sec)
                reply({"status": "SUCCESS" if res and res.ok else "ERROR", "message": (res.message if res else "")})
                continue

            if connection == "arm_raster":
                # Allow caller to choose mode:
                # - value truthy => continuous
                # - value falsy/None => step mode
                with self._state_lock:
                    has_iter = self._raster_iter is not None
                    active = self._raster_active

                if not has_iter or not active:
                    reply({"status": "ERROR", "message": "no_raster_configured"})
                    continue

                want_continuous = False
                try:
                    # Accept: 1, True, "1", "true", "continuous"
                    if isinstance(value, str):
                        want_continuous = value.strip().lower() in ("1", "true", "continuous", "cont")
                    else:
                        want_continuous = bool(value)
                except Exception:
                    want_continuous = False

                with self._state_lock:
                    self._raster_continuous = bool(want_continuous)

                if want_continuous:
                    self.status_signal.emit("ZMQ: raster armed (continuous).")
                    self._enqueue_next_raster_point()
                else:
                    self.status_signal.emit("ZMQ: raster armed (step mode).")

                reply({"status": "SUCCESS", "mode": ("continuous" if want_continuous else "step")})
                continue


            if connection == "move_to_next":
                # Step-mode handshake: move exactly one step, then reply when done.
                with self._state_lock:
                    active = self._raster_active
                    continuous = self._raster_continuous

                if not active:
                    reply({"status": "ERROR", "message": "raster_not_active"})
                    continue

                if continuous:
                    reply({"status": "ERROR", "message": "raster_in_continuous_mode"})
                    continue

                res = self.raster_step(source="zmq", wait=True, timeout_s=timeout_sec)

                # If iterator ended, raster_step() returns None and _finish_raster() fires signals.
                if res is None:
                    reply({"status": "FINISHED"})
                else:
                    reply({"status": "SUCCESS" if res.ok else "ERROR", "message": res.message})
                continue



        try:
            sock.close(0)
        except Exception:
            pass

    # -------------------------
    # Shutdown
    # -------------------------

    def shutdown(self) -> None:
        """
        Stop all threads and flush logs. Call on app exit.
        """
        self.stop_zmq_server()
        self._stop_evt.set()
        self._enqueue(MotorCommand(cmd_type=CommandType.NOOP, source="internal", priority=0))
        try:
            self._motor_thread.join(timeout=2.0)
        except Exception:
            pass
        self._flush_raster_log()


# -------------------------
# Optional factory helpers
# -------------------------

def create_controller_from_config(config_obj=None) -> "SystemController":
    """
    Convenience constructor that instantiates motors from hardware.py using config.py defaults.

    This keeps SystemController itself decoupled (it still accepts motor objects),
    but gives you a one-liner in main.py.

    Usage:
        import config
        from controller import create_controller_from_config
        ctl = create_controller_from_config(config.APP_CONFIG)
    """
    # Local imports to avoid hard dependencies when unit-testing controller logic
    from hardware import KCube, KinesisOptions

    # Late import of config to avoid circular imports
    if config_obj is None:
        try:
            import config as _cfg
            config_obj = getattr(_cfg, "APP_CONFIG", None) or _cfg
        except Exception as e:
            raise RuntimeError("No config provided and could not import config.py") from e

    # Support either dataclass-style config (APP_CONFIG.hardware.serial_x) or module constants (SERIAL_X)
    def _get(path, default=None):
        cur = config_obj
        for part in path.split("."):
            if hasattr(cur, part):
                cur = getattr(cur, part)
            elif isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur

    serial_x = _get("hardware.serial_x", _get("SERIAL_X"))
    serial_y = _get("hardware.serial_y", _get("SERIAL_Y"))
    if serial_x is None or serial_y is None:
        raise RuntimeError("Missing motor serials (expected hardware.serial_x/serial_y or SERIAL_X/SERIAL_Y)")

    opts = KinesisOptions(
        kinesis_dir=_get("hardware.kinesis_dir", None),
        poll_ms=int(_get("hardware.poll_ms", 100)),
        settings_wait_ms=int(_get("hardware.settings_wait_ms", 10_000)),
        device_settings_name=str(_get("hardware.device_settings_name", "Z912")),
        verbose=bool(_get("hardware.verbose", False)),
    )

    motor_x = KCube(serial_x, "X", options=opts)
    motor_y = KCube(serial_y, "Y", options=opts)

    ctl = SystemController(
        motor_x,
        motor_y,
        calibration_path=str(_get("paths.calibration_path", _get("CALIBRATION_PATH", "calibration_data.json"))),
        target_bounds=_get("raster.target_bounds", _get("DEFAULT_TARGET_BOUNDS", None)),
        motor_bounds=_get("hardware.motor_bounds", _get("DEFAULT_MOTOR_BOUNDS", None)),
        telemetry_period_s=float(_get("telemetry.period_s", _get("TELEMETRY_PERIOD_S", 0.2))),
    )

    # Optionally start ZMQ immediately if configured
    zmq_bind = _get("network.zmq_bind", _get("ZMQ_BIND", None))
    if zmq_bind:
        ctl.start_zmq_server(str(zmq_bind))

    return ctl
