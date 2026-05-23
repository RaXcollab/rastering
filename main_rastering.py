# main.py
from __future__ import annotations

import os
import sys

# LOAD-BEARING: rotpy must be imported BEFORE numpy / PyQt5 on Windows, or
# the Windows DLL loader deadlocks at rotpy's .pyd resolution (no error,
# silent hang). See GUIs/rastering/camera.py module-top comment for the
# full rationale -- rotpy's __init__.py discards its add_dll_directory
# handle so the search-path window is brief; if numpy/PyQt5 DLLs land in
# between, rotpy's bindings cannot find their Spinnaker dependencies.
# Symptom (hit 2026-05-22 Step 2 hardware test): `python main_rastering.py`
# launches, prints nothing, never paints a window. Same hang the Step-2.3
# gate hit until this block was added at the top of camera.py; main.py
# is a SECOND entry point that bypasses camera.py's module-top ordering
# because it imports PyQt5 directly first.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import rotpy  # noqa: F401  -- side effect: register DLL paths
    from rotpy import system as _rotpy_system  # noqa: F401  -- eager .pyd load
    from rotpy import camera as _rotpy_camera  # noqa: F401  -- eager .pyd load
except ImportError:  # pragma: no cover -- production envs always have rotpy
    pass

from PyQt5 import QtWidgets
from PyQt5 import QtGui

import config

UI_FILE = 'raster_gui.ui'

ICON_PATH = 'rastering.ico'

def build_controller():
    """
    Build the raster controller using config + hardware.

    Tries create_controller_from_config() first (if you added it),
    otherwise constructs motors directly and builds SystemController.
    """
    import raster_controller

    # Preferred: factory helper if present
    if hasattr(raster_controller, "create_controller_from_config"):
        return raster_controller.create_controller_from_config(config.APP_CONFIG)

    # Fallback: manual construction
    from hardware import KCube, KinesisOptions
    from raster_controller import SystemController

    opts = KinesisOptions(
        kinesis_dir=config.APP_CONFIG.hardware.kinesis_dir,
        poll_ms=config.APP_CONFIG.hardware.poll_ms,
        settings_wait_ms=config.APP_CONFIG.hardware.settings_wait_ms,
        device_settings_name=config.APP_CONFIG.hardware.device_settings_name,
        verbose=config.APP_CONFIG.hardware.verbose,
    )

    motor_x = KCube(config.APP_CONFIG.hardware.serial_x, "X", options=opts)
    motor_y = KCube(config.APP_CONFIG.hardware.serial_y, "Y", options=opts)

    ctl = SystemController(
        motor_x,
        motor_y,
        calibration_path=config.APP_CONFIG.paths.calibration_path,
        target_bounds=config.APP_CONFIG.raster.target_bounds,
        motor_bounds=config.APP_CONFIG.hardware.motor_bounds,
        telemetry_period_s=config.APP_CONFIG.telemetry.period_s,
    )

    # optional: auto-start ZMQ
    if config.APP_CONFIG.network.zmq_bind:
        ctl.start_zmq_server(
            config.APP_CONFIG.network.zmq_bind,
            pub_bind=getattr(config.APP_CONFIG.network, 'pub_bind', ''),
        )

    return ctl


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(ICON_PATH))

    # Build controller + UI
    controller = build_controller()

    from ui import RasterMainWindow
    win = RasterMainWindow(controller, ui_path=UI_FILE)

    # Auto-load the last-used calibration file. On first launch (no
    # last_calibration_state.json yet) this is a no-op -- the user must
    # browse-load explicitly. The bundled camera_settings (if any) are
    # handed to the UI so it can enable the "Apply camera settings from
    # cal" button without auto-applying them.
    try:
        from raster_controller import load_last_calibration_path
        last_path = load_last_calibration_path()
        if last_path:
            data = controller.load_calibration_from_path(last_path)
            win.note_loaded_cal_bundle(data, source_path=last_path)
    except Exception as e:
        print(f"[startup] Could not auto-load last calibration: {e}")

    win.show()

    rc = app.exec_()

    # Clean shutdown
    try:
        controller.shutdown()
    except Exception:
        pass

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
