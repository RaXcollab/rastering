"""Spinnaker (rotpy) camera driver for the rastering GUI.

Replaces the previous IDS uEye driver; the uEye driver is preserved verbatim
at ``camera_ueye.py`` for reference/diff (commit ``d46ea66``).

This file is the new active driver and is grown across sub-step commits
(2.1 = skeleton; 2.2 = ``CameraConfig`` schema; 2.3 = ``SpinCamera`` I/O;
2.4 = ``CameraThread`` slots/signals; 2.5 = ``CameraThread.run()`` loop;
2.6 = ini load/save). See ``~/.claude/plans/i-have-gotten-new-wondrous-shore.md``
"Step 2 - Execution Sequence" for the full plan.

Audit-mandated invariants (preserved across all sub-steps):
    B1  bounded ``get_next_image`` timeout; loop polls ``isInterruptionRequested()``
        >= 2/sec via the timeout-as-None-sentinel pattern.
    B2  ``SpinCamera.open()`` exception-safe (own try/except teardown across init
        steps); ``SpinCamera.close()`` idempotent across partial init.
    S1  ``grab()`` returns ``np.ascontiguousarray`` copy with ``.base is None``,
        BEFORE ``img.release()`` returns the SDK buffer to the pool.
    S2  ``_pending`` snapshot+clear is atomic under ``_params_lock``; apply is
        OUTSIDE the lock (no C++ I/O held under a mutex).
    S3  every ``set_*`` / ``request_*`` slot body is solely the mutex-guarded
        ``_pending`` dict write -- no slot touches ``self._cam`` from the GUI
        thread.
    N1  trailing-int parser on grab-exception messages; only timeout/incomplete
        codes go in ``_TRANSIENT_GRAB_ERROR_CODES``; everything else is immediate.
    N2  grab returns uint8 2-D ndarray for Mono8 (the format the pyqtgraph view
        consumes with ``autoLevels=False``).
"""

# --- Module-level OpenMP workaround (LOAD-BEARING) -------------------------
# MUST be set BEFORE the first rotpy or numpy import. The env has three
# OpenMP DLLs (rotpy's bundled libiomp5md.dll, numpy MKL's libiomp5md.dll,
# conda's libomp.dll); without this the second to initialize aborts the
# process with "OMP: Error #15". See docs/ROTPY_BUILD.md Section 3 and the
# 2026-05-22 debug session (commit 9d3e91c) for the canonical analysis.
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import logging
import time
from dataclasses import dataclass, field, replace as _dc_replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore

logger = logging.getLogger(__name__)


# --- Transient error codes (refined in sub-step 2.5) -----------------------
# Initial empty set per AUDIT:N1 safe default: no trailing-int code in
# rotpy ``SpinnakerAPIException`` messages -> not transient -> immediate emit.
# Note that timeout + incomplete-image are handled BEFORE this codepath via
# the None-sentinel pattern in ``SpinCamera.grab()`` (sub-step 2.3) -- only
# genuine SDK exceptions reach the throttle. Refine if hardware testing
# surfaces a class of recoverable exceptions worth throttling.
_TRANSIENT_GRAB_ERROR_CODES: frozenset = frozenset()
_ERROR_THROTTLE_WINDOW_S: float = 5.0


def _ensure_spinnaker_runtime() -> None:
    """No-op. rotpy's ``__init__.py`` owns DLL discovery via its bundled
    ``share/rotpy/spinnaker/{bin,cti}``. Prepending the installed Teledyne
    Spinnaker SDK (currently 4.3.0.190) onto PATH or via
    ``os.add_dll_directory`` would shadow rotpy's bundled v2.6 runtime and
    crash the bindings at first node call. The function name is retained
    so legacy callers that reference it don't ``AttributeError``."""
    return None


# --- CameraConfig (populated in sub-step 2.2) -------------------------------
@dataclass
class CameraConfig:
    """Spinnaker camera config. Field schema populated in sub-step 2.2."""
    camera_id: int = 0  # placeholder; full schema in 2.2


# --- SpinCamera (real I/O in sub-step 2.3) ----------------------------------
class SpinCamera:
    """Spinnaker camera wrapper. Real I/O via rotpy is populated in 2.3."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self._opened: bool = False
        self._init_cam_called: bool = False
        self._acquiring: bool = False
        self._cam = None  # rotpy Camera handle, populated in open()
        self._system = None  # rotpy SpinSystem handle, populated in open()

    def open(self) -> None:
        raise NotImplementedError("sub-step 2.3")

    def close(self) -> None:
        """Idempotent across partial init (AUDIT:B2). 2.3 fills the real teardown."""
        return None

    def grab(self) -> Optional[np.ndarray]:
        """Returns uint8 2-D ndarray (Mono8) per AUDIT:N2. None on timeout /
        incomplete (the interruptibility-poll sentinel per AUDIT:B1)."""
        raise NotImplementedError("sub-step 2.3")

    def reinit_aoi(self, width: int, height: int, start_x: int, start_y: int) -> None:
        raise NotImplementedError("sub-step 2.3")

    def get_camera_info(self) -> Dict[str, Any]:
        raise NotImplementedError("sub-step 2.3")


# --- CameraThread (slots in 2.4; run loop in 2.5) ---------------------------
class CameraThread(QtCore.QThread):
    """Camera worker thread.

    Signal contract preserved from the uEye driver (caller-visible API):
        new_frame(np.ndarray)        - emitted on each successful frame
        status(str)                  - lifecycle / informational
        error(str)                   - errors (throttled per AUDIT:N1)
        camera_info_signal(dict)     - ranges + current values, on open / refresh

    Slot contract (AUDIT:S2 + S3): every ``set_*`` / ``request_*`` body is
    solely the mutex-guarded ``_pending`` write -- slots never touch
    ``self._cam`` from the GUI thread. ``run()`` snapshots + clears
    ``_pending`` atomically under the lock, then applies OUTSIDE the lock."""

    new_frame = QtCore.pyqtSignal(np.ndarray)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)
    camera_info_signal = QtCore.pyqtSignal(dict)

    def __init__(self, cfg: Optional[CameraConfig] = None, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg if cfg is not None else CameraConfig()
        self._cam: Optional[SpinCamera] = None
        self._running: bool = False
        self._params_lock = QtCore.QMutex()
        self._pending: Dict[str, Any] = {}
        self._err_throttle_state: Dict[str, Tuple[int, float, float]] = {}

    def stop(self) -> None:
        """Set the loop-exit flag and request interruption. Loop
        interruptibility depends on the bounded ``get_next_image`` timeout
        in ``SpinCamera.grab()`` (AUDIT:B1) -- the timeout returns ``None``
        which causes the loop to re-check ``isInterruptionRequested()`` at
        >= 2/sec."""
        self._running = False
        self.requestInterruption()

    def run(self) -> None:
        raise NotImplementedError("sub-step 2.5")


# --- ini load/save (sub-step 2.6) -------------------------------------------
def load_ueye_config_from_ini(ini_path: str, **overrides: Any) -> CameraConfig:
    raise NotImplementedError("sub-step 2.6")


def save_settings_to_ini(ini_path: str, settings: Dict[str, Any]) -> None:
    raise NotImplementedError("sub-step 2.6")


def apply_ini_to_camera(cam: SpinCamera, ini_path: str) -> None:
    raise NotImplementedError("sub-step 2.6")


def _load_display_settings_from_ini(ini_path: str) -> Dict[str, Any]:
    raise NotImplementedError("sub-step 2.6")


# --- Bottom-of-module aliases (decision 3) ----------------------------------
# Backward-compatible names for callers that still import the uEye-era
# symbols. ui.py:28 and ui.py:470 import these; renaming would require a
# bigger same-patch edit, deferred to Step 3.
UEyeConfig = CameraConfig
UEyeCamera = SpinCamera
UEyeCameraThread = CameraThread
