"""Regression test: every slider<->spin pair commits to the camera.

Root cause (commit 56d1733): camera_settings_dock wired each slider's
display-sync slot to block the spin's signals, then relied on a SEPARATE,
hand-duplicated `slider -> cam_thread.set_*` lambda for the camera path.
fps and gamma got that second lambda; **exposure did not**. So dragging
the exposure slider moved the spinbox display but never reached the
camera. The deeper bug class is the duplicated slider->physical formula
with no single source of truth -- the next slider added repeats the trap.

These tests pin the contract: a single slider gesture commits to the
camera exactly once, for fps AND exposure AND gamma, with the spin path
unchanged and no double-fire.

Standalone-runnable (mirrors the sibling scripts in tests/):
    conda activate rastering && python tests/test_exposure_slider_camera.py
Also collectable by pytest. Skips cleanly if PyQt5 / camera_settings_dock
are not importable (no Qt display required -- offscreen platform).
"""

from __future__ import annotations

import os
import sys
import types
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# camera_settings_dock.py lives one level up from tests/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PyQt5 import QtWidgets  # noqa: E402
    from camera_settings_dock import CameraSettingsDock  # noqa: E402

    _IMPORT_ERR = None
except Exception as e:  # pragma: no cover - environment-dependent
    _IMPORT_ERR = e


def _skip(reason: str) -> None:
    raise unittest.SkipTest(reason)


_app = None


def _ensure_app():
    global _app
    if _app is None:
        _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return _app


class _FakeCam:
    """Records set_*/request_* calls; camera_info_signal.connect is a no-op."""

    camera_info_signal = types.SimpleNamespace(connect=lambda *a, **k: None)

    def __init__(self):
        self.calls: dict = {}

    def __getattr__(self, name):
        if name.startswith(("set_", "request_")):
            def rec(*args):
                self.calls.setdefault(name, []).append(
                    args[0] if len(args) == 1 else args
                )

            return rec
        raise AttributeError(name)


_CAM_INFO = {
    "sensor_width": 1280,
    "sensor_height": 1024,
    "pixel_clocks": [],
    "pixel_clock": 0,
    "fps_min": 0.1,
    "fps_max": 200.0,
    "fps": 20.0,
    "exposure_min": 0.01,
    "exposure_max": 1000.0,
    "exposure_inc": 0.01,
    "exposure": 30.0,
    "gain": 0,
    "gain_boost": False,
    "gamma_min": 0.01,
    "gamma_max": 2.2,
    "gamma": 1.0,
    "aoi_width": 1280,
    "aoi_height": 1024,
    "aoi_x": 0,
    "aoi_y": 0,
}


def _make_dock():
    _ensure_app()
    dock = CameraSettingsDock()
    fake = _FakeCam()
    dock.connect_to_camera_thread(fake)
    # Seed realistic ranges/values; this blocks all widget signals internally.
    dock.update_from_camera_info(dict(_CAM_INFO))
    fake.calls.clear()  # discard anything emitted during population
    return dock, fake


def test_exposure_slider_commits_to_camera():
    if _IMPORT_ERR is not None:
        _skip(f"PyQt5/camera_settings_dock not importable: {_IMPORT_ERR}")
    dock, fake = _make_dock()

    dock.exposure_slider.setValue(5000)  # mid travel of the 0..10000 slider

    calls = fake.calls.get("set_exposure_ms", [])
    assert len(calls) == 1, f"expected exactly one set_exposure_ms, got {calls}"
    expected = dock._exp_min + 0.5 * (dock._exp_max - dock._exp_min)
    assert abs(float(calls[0]) - expected) < 1e-6, (
        f"set_exposure_ms({calls[0]}) != expected {expected}"
    )


def test_fps_and_gamma_sliders_still_commit():
    if _IMPORT_ERR is not None:
        _skip(f"PyQt5/camera_settings_dock not importable: {_IMPORT_ERR}")
    dock, fake = _make_dock()

    dock.fps_slider.setValue(5000)
    dock.gamma_slider.setValue(500)

    assert len(fake.calls.get("set_target_fps", [])) == 1, fake.calls
    assert len(fake.calls.get("set_gamma", [])) == 1, fake.calls


def test_one_camera_call_per_gesture_no_double_fire():
    if _IMPORT_ERR is not None:
        _skip(f"PyQt5/camera_settings_dock not importable: {_IMPORT_ERR}")
    dock, fake = _make_dock()

    dock.exposure_slider.setValue(7000)   # one slider gesture
    dock.exposure_spin.setValue(123.0)    # one spinbox gesture

    calls = fake.calls.get("set_exposure_ms", [])
    assert len(calls) == 2, (
        f"expected 2 set_exposure_ms (one per gesture, no slider<->spin "
        f"echo), got {calls}"
    )


def _run() -> int:
    if _IMPORT_ERR is not None:
        print(f"SKIP (all): PyQt5/camera_settings_dock not importable: {_IMPORT_ERR}")
        return 0
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    fails = 0
    skips = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except unittest.SkipTest as s:
            skips += 1
            print(f"SKIP {fn.__name__}: {s}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}: {e!r}")
    print(f"{len(fns) - fails - skips}/{len(fns)} passed "
          f"({skips} skipped, {fails} failed)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(_run())
