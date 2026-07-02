"""Hardware-gated end-to-end smoke for the Spinnaker camera driver.

Skips cleanly if rotpy isn't importable OR if no Spinnaker camera enumerates
(CI / dev-laptop / GigE disconnected). When a camera IS present, exercises
``SpinCamera.open() -> grab() -> close()`` twice -- second cycle catches any
B2 (open() partial-init teardown) regression that would manifest as a
"camera in use" / device-in-use error on the second ``init_cam``.

Mirrors the structure of ``scripts/spin_smoke.py`` but as a pytest test so it
runs alongside the other regression tests. Frame ownership (AUDIT:S1) and
buffer shape (AUDIT:N2) are asserted on every grab.

Run locally (with camera connected):

    conda activate rastering
    python -m pytest tests/test_camera_smoke.py -v

To force-skip (e.g. on a build server with no GigE link):

    pytest -v -k "not test_camera_smoke"
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# tests/ is one level under the rastering root; conftest.py at that root sets
# up the rotpy DLL-load order BEFORE this module is imported (see
# tests/conftest.py for the mechanism).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import camera as cm  # noqa: E402
    _CAMERA_IMPORT_ERR = None
except Exception as e:  # pragma: no cover -- env-dependent
    cm = None  # type: ignore[assignment]
    _CAMERA_IMPORT_ERR = e


def _gate_or_skip():
    """SKIP if rotpy isn't usable or no camera is enumerated."""
    if _CAMERA_IMPORT_ERR is not None or cm is None:
        pytest.skip(f"camera module not importable: {_CAMERA_IMPORT_ERR}")
    try:
        from rotpy.system import SpinSystem  # noqa: PLC0415
        from rotpy.camera import CameraList  # noqa: PLC0415
    except ImportError as e:
        pytest.skip(f"rotpy not importable: {e}")
    try:
        sys_ = SpinSystem()
        cams = CameraList.create_from_system(
            sys_, update_interfaces=True, update_cams=True
        )
        n = cams.get_size()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SpinSystem enumeration failed: {e}")
    if n == 0:
        pytest.skip("no Spinnaker cameras enumerated (GigE link / IP / jumbo?)")


def _one_cycle(label: str) -> None:
    cfg = cm.CameraConfig(exposure_ms=10.0, acq_frame_rate=20.0)
    spin = cm.SpinCamera(cfg)
    spin.open()
    try:
        info = spin.get_camera_info()
        assert info, "get_camera_info returned empty dict"
        assert "sensor_width" in info and "sensor_height" in info
        assert info["sensor_width"] > 0 and info["sensor_height"] > 0

        successes = 0
        for i in range(5):
            f = spin.grab()
            if f is None:
                # Timeout / image-level transient. Don't fail; just don't
                # count it.
                continue
            assert f.dtype == np.uint8, f"[{label}] dtype {f.dtype} != uint8"
            assert f.ndim == 2, f"[{label}] ndim {f.ndim} != 2"
            assert f.flags["C_CONTIGUOUS"], f"[{label}] frame not C-contiguous"
            # AUDIT:S1 -- frame must own its memory after grab().
            assert f.base is None, f"[{label}] frame.base is not None"
            successes += 1
        assert successes >= 1, f"[{label}] zero successful grabs in 5 tries"
    finally:
        spin.close()


def test_camera_smoke_open_grab_close():
    """Single open/grab/close cycle. SKIPs if no hardware."""
    _gate_or_skip()
    _one_cycle("cycle-1")


def test_camera_smoke_b2_teardown_idempotency():
    """Two open/grab/close cycles back-to-back. A B2 leak (open() not
    exception-safe across partial init) would surface as ``camera in use``
    or ``resource_in_use`` on the second ``init_cam``. SKIPs if no
    hardware."""
    _gate_or_skip()
    _one_cycle("cycle-1")
    _one_cycle("cycle-2")


def test_ensure_acquiring_never_raises_and_self_heals():
    """No hardware needed. ensure_acquiring() must swallow a
    begin_acquisition failure (returning False), succeed once the camera
    recovers, and be a no-op while already acquiring -- the run loop leans
    on all three to un-freeze the live view after a failed AOI restart."""
    if cm is None:
        pytest.skip(f"camera module not importable: {_CAMERA_IMPORT_ERR}")

    class _Cam:
        def __init__(self):
            self.calls = 0
            self.fail = True

        def begin_acquisition(self):
            self.calls += 1
            if self.fail:
                raise RuntimeError("stream dead")

    sc = cm.SpinCamera.__new__(cm.SpinCamera)  # skip __init__ (no rotpy)
    sc._cam = _Cam()
    sc._acquiring = False
    assert sc.ensure_acquiring() is False   # failure swallowed, stays down
    sc._cam.fail = False
    assert sc.ensure_acquiring() is True    # self-heals
    assert sc._acquiring is True
    assert sc.ensure_acquiring() is True    # no re-begin while acquiring
    assert sc._cam.calls == 2

    sc_closed = cm.SpinCamera.__new__(cm.SpinCamera)
    sc_closed._cam = None
    sc_closed._acquiring = False
    assert sc_closed.ensure_acquiring() is False  # closed camera: False, no raise
