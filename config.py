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
    serial_x: str = "27270522"
    serial_y: str = "27270471"
    

    # Kinesis install directory.
    # - Set explicitly if needed, or leave None to let hardware.py search common locations.
    kinesis_dir: Optional[str] = None

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
    # PUB-SUB bind endpoint for broadcasting status to BLACS (set to "" to disable)
    pub_bind: str = "tcp://*:55536"


@dataclass(frozen=True)
class CameraConfig:
    # Defaults for camera operation (Spinnaker / Blackfly GigE via rotpy).
    # ui.py reads these for the runtime ``camera.CameraConfig`` construction.
    # Renamed from the uEye era at Step 3 of the rotpy migration.

    # Which camera to bind. Prefer ``serial``; ``camera_id`` is index-fallback.
    serial: Optional[str] = None  # e.g. "26120532" -- single-camera rig
    camera_id: int = 0

    # Path to a Spinnaker-schema .ini file. If set + present, its settings
    # OVERRIDE the fields below. Set to "" or None to use the manual fields.
    # New schema lives in ``camera_params_spin.ini``; the legacy uEye file
    # ``camera_params.ini`` is kept (untouched) as a rollback artifact.
    camera_params_ini: Optional[str] = 'camera_params_spin.ini'

    # Default exposure (ms) -- Spinnaker accepts microseconds internally
    # (ExposureTime), camera.py converts ms -> us at the node-set layer.
    exposure_ms_default: float = 30.0

    # Acquisition frame rate (fps). Replaces uEye ``target_fps`` (Parity 5).
    # Camera-side control via Spinnaker's ``AcquisitionFrameRate`` node.
    acq_frame_rate: float = 1 / exposure_ms_default * 1000  # 1/30ms = 33.3 fps
    # When False, the camera lets exposure exceed the 1/fps frame period
    # (Parity 5 -- the Blackfly-native equivalent of uEye's
    # ``prioritize_exposure`` timing mode).
    acq_frame_rate_enable: bool = True

    # Analog controls. Spinnaker gain is float dB on most Blackfly models
    # (~0..48 dB) -- NOT the uEye int 0..100 of prior versions (Parity 2).
    gain_db: float = 0.0
    gamma: float = 1.0
    gamma_enable: bool = False

    # Pixel format. Mono8 keeps the rastering GUI's existing
    # uint8 2-D display path; non-Mono8 formats require dock-side
    # range-scaling (out of scope until Step 4).
    pixel_format: str = "Mono8"

    # GigE transport tuning. Replaces uEye pixel-clock (Parity 4 -- GigE has
    # no pixel clock; bandwidth is controlled at the link layer).
    gige_packet_size: int = 9000      # jumbo frames must be enabled on NIC
    # ``None`` = leave at camera default (Blackfly auto-selects). Set an int
    # to cap throughput (bytes/sec) for multi-camera or shared-link setups.
    device_link_throughput_limit: Optional[int] = None

    # Image orientation (applied in UI via a display transform, not on camera)
    flip_x: bool = False
    flip_y: bool = True

    # Requested AOI / output size. 0 = full sensor (camera default).
    width: int = 500
    height: int = 500
    # Absolute top-left offsets into the sensor (Parity 6 -- Spinnaker is
    # top-left absolute, NOT centered like uEye).
    roi_offset_x: int = 0
    roi_offset_y: int = -100

    # Output frame conversion (display-only flag, never reaches the camera).
    emit_rgb: bool = False  # if True, grab() returns (H,W,3); else (H,W)


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
