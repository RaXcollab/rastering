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
    # Sensor + AOI
    "sensor_width": 1280,
    "sensor_height": 1024,
    "roi_width": 1280,
    "roi_height": 1024,
    "roi_offset_x": 0,
    "roi_offset_y": 0,
    "aoi_width": 1280,
    "aoi_height": 1024,
    "aoi_x": 0,
    "aoi_y": 0,
    # Spinnaker-native timing
    "acq_fps": 20.0,
    "acq_fps_min": 0.1,
    "acq_fps_max": 200.0,
    "acq_fps_enable": True,
    "exposure_us": 30_000.0,
    "exposure_us_min": 20.0,
    "exposure_us_max": 30_000_000.0,
    "exposure_us_inc": 1.0,
    # Legacy ms-keyed fallback (camera.py emits both during migration window)
    "exposure": 30.0,
    "exposure_min": 0.02,
    "exposure_max": 30000.0,
    "exposure_inc": 0.001,
    # Spinnaker-native gain (Parity 2: float dB)
    "gain_db": 0.0,
    "gain_db_min": 0.0,
    "gain_db_max": 47.99,
    # Legacy int gain mirror (round-tripped from gain_db) -- camera.py emits
    # both during migration window; the new dock reads gain_db.
    "gain": 0,
    "gain_min": 0,
    "gain_max": 48,
    # Gamma + enable
    "gamma": 1.0,
    "gamma_min": 0.25,
    "gamma_max": 4.0,
    "gamma_enable": False,
    # Pixel format + enum entries
    "pixel_format": "Mono8",
    "pixel_formats": ["Mono8", "Mono16", "BayerRG8"],
    # GigE transport
    "packet_size": 9000,
    "packet_delay": 1000,
    "throughput_limit": 0,
    "throughput_limit_min": 0,
    "throughput_limit_max": 380_000_000,
    # Blackfly-native (decision 8) -- nonzero range so the dock enables the
    # widgets and we can exercise their commit paths in the gain test below.
    "black_level": 0.0,
    "black_level_min": 0.0,
    "black_level_max": 100.0,
    "black_level_clamping": True,
    "defect_correction": False,
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
    """Bug class regression test for both EXPOSURE and GAIN (AUDIT:B2).

    The legacy uEye dock had two separate paths for gain: an int 1:1
    slider<->spin display sync wired in _wire_signals + a separate
    set_master_gain lambda in connect_to_camera_thread. Step 4 collapsed
    both into the shared _bind_param_controls helper -- same as
    exposure/fps/gamma -- so a slider gesture and a spin gesture each
    commit to the camera EXACTLY ONCE per gesture, with the partner
    widget mirror-updated under blockSignals (no echo, no double-fire).
    """
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

    # Step-4 extension: same contract for gain (now float dB).
    fake.calls.clear()
    dock.gain_slider.setValue(500)   # one slider gesture -- mid travel of 0..1000
    dock.gain_spin.setValue(12.5)    # one spinbox gesture

    gain_calls = fake.calls.get("set_gain_db", [])
    assert len(gain_calls) == 2, (
        f"expected 2 set_gain_db (one per gesture, no slider<->spin echo), "
        f"got {gain_calls}"
    )
    # No leakage to the retired legacy slot.
    assert "set_master_gain" not in fake.calls, (
        f"set_master_gain MUST NOT fire after Step 4 rewire; got "
        f"{fake.calls.get('set_master_gain')}"
    )


def test_gain_slider_commits_in_dB():
    """Step-4 regression: gain slider commits the live dB value, not an
    int 0-100. The slider integer endpoint is 0..1000 (per Step 4); the
    physical range is gain_db_min..gain_db_max from camera_info."""
    if _IMPORT_ERR is not None:
        _skip(f"PyQt5/camera_settings_dock not importable: {_IMPORT_ERR}")
    dock, fake = _make_dock()

    dock.gain_slider.setValue(500)  # mid travel of 0..1000

    calls = fake.calls.get("set_gain_db", [])
    assert len(calls) == 1, f"expected exactly one set_gain_db, got {calls}"
    expected = dock._gain_min + 0.5 * (dock._gain_max - dock._gain_min)
    # _bind_param_controls rounds gain to ndigits=2, so the emitted value
    # quantizes to 0.01 dB. Tolerance is one quantization step.
    assert abs(float(calls[0]) - expected) < 0.01, (
        f"set_gain_db({calls[0]}) != expected {expected} within 0.01 dB"
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
