# main.py
from __future__ import annotations

import sys
from PyQt5 import QtWidgets

import config

UI_FILE = 'raster_gui.ui'


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
        ctl.start_zmq_server(config.APP_CONFIG.network.zmq_bind)

    return ctl


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)

    # Build controller + UI
    controller = build_controller()

    from ui import RasterMainWindow
    win = RasterMainWindow(controller, ui_path=UI_FILE)

    # optional: load last calibration on startup (safe if file missing)
    try:
        controller.load_calibration()
    except Exception:
        pass

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
