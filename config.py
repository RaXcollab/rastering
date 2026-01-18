"""
config.py

Central configuration for the beam steering / raster application.

Philosophy:
- Keep all settings here (single source of truth).
- No environment-variable overrides.

NOTE: Target-space units here should match your plot/click coordinates (pixels or mm depending on UI scaling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

Bounds = Tuple[float, float, float, float]  # xmin, xmax, ymin, ymax


# -------------------------
# Dataclass-style config
# -------------------------

@dataclass(frozen=True)
class HardwareConfig:
    # Motor serials
    serial_x: str = "27270471"
    serial_y: str = "27270522"

    # Kinesis install directory.
    # - Set explicitly if needed, or leave None to let hardware.py search common locations.
    kinesis_dir: Optional[str] = r"C:\Program Files\Thorlabs\Kinesis"

    # Kinesis init/polling
    poll_ms: int = 100
    settings_wait_ms: int = 10_000
    device_settings_name: str = "Z912"
    verbose: bool = False

    # HARD safety bounds in motor units (xmin, xmax, ymin, ymax)
    # Set to None to disable hard bounding in the controller.
    motor_bounds: Optional[Bounds] = None


@dataclass(frozen=True)
class TelemetryConfig:
    # Position polling period (seconds); controller enqueues READ_POS at this rate
    period_s: float = 0.2


@dataclass(frozen=True)
class NetworkConfig:
    # ZMQ bind endpoint (set to "" or None to disable auto-start)
    zmq_bind: str = "tcp://*:55535"


@dataclass(frozen=True)
class CameraConfig:
    # Defaults for camera operation (uEye). ui.py reads these for UEyeConfig construction.

    # Which uEye camera to open (0 is typical for single-camera setups)
    camera_id: int = 0

    # Acquisition pixel clock (MHz): controls FPS + bandwidth
    pixel_clock_mhz: int = 10

    # Default exposure (ms)
    exposure_ms_default: float = 50.0

    # Image orientation options (applied in UI via a display transform)
    flip_x: bool = False
    flip_y: bool = True

    # Requested AOI / output size (camera.py can honor these depending on your setup)
    width: int = 420
    height: int = 420
    # Add offsets to shift the AOI from the center (pixels)
    # Swapped x,y or inverted +/- if you rotate or flip the image in the UI
    roi_offset_x: int = -100
    roi_offset_y: int = 100

    # Capture strategy + channel order
    use_freeze: bool = True   # use FreezeVideo loop
    emit_rgb: bool = True     # convert BGR->RGB for UI


@dataclass(frozen=True)
class RasterDefaults:
    # SOFT bounds in target-space units (what user sees/raster uses).
    # Keep None if you want UI to define these from hull/bounds widgets.
    target_bounds: Optional[Bounds] = None

    x_step: float = 0.2
    y_step: float = 0.2

    # Spiral defaults
    spiral_radius: float = 1.0
    spiral_step: float = 0.1
    spiral_angle_step: float = 0.2
    spiral_angle_step_change: float = 0.0


@dataclass(frozen=True)
class PathsConfig:
    calibration_path: str = "calibration_data.json"
    raster_log_dir: str = "logs"


@dataclass(frozen=True)
class AppConfig:
    hardware: HardwareConfig = HardwareConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    network: NetworkConfig = NetworkConfig()
    camera: CameraConfig = CameraConfig()
    raster: RasterDefaults = RasterDefaults()
    paths: PathsConfig = PathsConfig()


APP_CONFIG = AppConfig()

# -------------------------
# Backwards-compatible constants (optional)
# -------------------------

SERIAL_X = APP_CONFIG.hardware.serial_x
SERIAL_Y = APP_CONFIG.hardware.serial_y

CALIBRATION_PATH = APP_CONFIG.paths.calibration_path
ZMQ_BIND = APP_CONFIG.network.zmq_bind
TELEMETRY_PERIOD_S = APP_CONFIG.telemetry.period_s

DEFAULT_TARGET_BOUNDS = APP_CONFIG.raster.target_bounds
DEFAULT_MOTOR_BOUNDS = APP_CONFIG.hardware.motor_bounds
