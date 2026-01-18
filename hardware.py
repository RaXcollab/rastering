"""
hardware.py

Hardware driver layer for the beam-steering kinematic mirror application.

Goals:
- Provide a small, stable API for the rest of the application:
    - move_to(pos)
    - get_position()
    - stop()        (optional but recommended)
    - soft_home() / hard_home()
    - set_backlash(value)
    - disconnect()
- Contain all Thorlabs Kinesis / pythonnet (clr) handling here.
- Build the Kinesis device list ONCE per process.
- Avoid PyQt imports; this module should be usable in tests / headless contexts.

Notes:
- This file is adapted from your existing toolbox.py (no wheel reinvention),
  but cleaned up for reliability (timeouts, explicit imports, better errors).
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Sequence, Union

# pythonnet / .NET interop
import clr  # type: ignore

# ---------------------------
# Kinesis assembly loading
# ---------------------------

_DEFAULT_KINESIS_DIRS: Sequence[str] = (
    r"C:\Program Files\Thorlabs\Kinesis",
    r"C:\Program Files (x86)\Thorlabs\Kinesis",
)

# You can override with env var KINESIS_DIR if needed.
_ENV_KINESIS_DIR = "KINESIS_DIR"

# Global, process-wide init guards
_KINESIS_LOADED = False
_DEVICE_LIST_BUILT = False
_INIT_LOCK = threading.Lock()


def _try_add_reference(dll_path: str) -> None:
    if os.path.exists(dll_path):
        clr.AddReference(dll_path)
    else:
        raise FileNotFoundError(dll_path)


def ensure_kinesis_loaded(kinesis_dir: Optional[str] = None) -> None:
    """
    Load required Thorlabs Kinesis .NET assemblies exactly once.
    """
    global _KINESIS_LOADED
    if _KINESIS_LOADED:
        return

    with _INIT_LOCK:
        if _KINESIS_LOADED:
            return

        # Determine Kinesis directory
        candidates = []
        if kinesis_dir:
            candidates.append(kinesis_dir)
        env_dir = os.environ.get(_ENV_KINESIS_DIR, "").strip()
        if env_dir:
            candidates.append(env_dir)
        candidates.extend(_DEFAULT_KINESIS_DIRS)

        last_err: Optional[Exception] = None
        for base in candidates:
            base = os.path.normpath(base)
            try:
                dm = os.path.join(base, "Thorlabs.MotionControl.DeviceManagerCLI.dll")
                gm = os.path.join(base, "Thorlabs.MotionControl.GenericMotorCLI.dll")

                # The KCube DCServo dll name sometimes appears with different capitalization.
                # Try the common variants.
                kc_variants = [
                    "Thorlabs.MotionControl.KCube.DCServoCLI.dll",
                    "ThorLabs.MotionControl.KCube.DCServoCLI.dll",
                ]

                _try_add_reference(dm)
                _try_add_reference(gm)

                kc_loaded = False
                for kc in kc_variants:
                    try:
                        _try_add_reference(os.path.join(base, kc))
                        kc_loaded = True
                        break
                    except FileNotFoundError as e:
                        last_err = e
                        continue

                if not kc_loaded:
                    raise last_err or FileNotFoundError("KCube DCServo CLI dll not found")

                _KINESIS_LOADED = True
                return

            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            "Failed to load Thorlabs Kinesis assemblies. "
            "Set environment variable KINESIS_DIR to your Kinesis install folder. "
            f"Last error: {last_err}"
        )


def build_device_list_once() -> None:
    """
    Build the Thorlabs device list once per process.
    """
    global _DEVICE_LIST_BUILT
    if _DEVICE_LIST_BUILT:
        return

    ensure_kinesis_loaded()
    with _INIT_LOCK:
        if _DEVICE_LIST_BUILT:
            return
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI  # type: ignore
        DeviceManagerCLI.BuildDeviceList()
        _DEVICE_LIST_BUILT = True


def list_device_serials() -> list[str]:
    """
    Returns list of connected device serial numbers as strings (best-effort).
    """
    build_device_list_once()
    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI  # type: ignore
    devices = DeviceManagerCLI.GetDeviceList()
    # pythonnet iterables are fine; normalize to python strings
    return [str(d) for d in devices]


# ---------------------------
# Motor interface + drivers
# ---------------------------

class Motor:
    """
    Minimal motor interface expected by controller.py.

    NOTE: controller.py serializes motor access in a dedicated thread,
    so move_to() is allowed to block here (but MUST have a timeout).
    """
    def __init__(self, serial_no: Union[str, int], name: str):
        self.serial_no = str(serial_no)
        self.name = name
        self.is_locked = False

    def lock(self) -> None:
        self.is_locked = True

    def unlock(self) -> None:
        self.is_locked = False

    def is_available(self) -> bool:
        raise NotImplementedError

    def move_to(self, pos: float, *, timeout_s: float = 10.0) -> float:
        raise NotImplementedError

    def get_position(self) -> float:
        raise NotImplementedError

    def stop(self) -> None:
        # Optional; not all devices support it
        return

    def soft_home(self, *, timeout_s: float = 30.0) -> float:
        return self.get_position()

    def hard_home(self, *, timeout_s: float = 30.0) -> float:
        return self.get_position()

    def set_backlash(self, value: float) -> float:
        return float(value)

    def disconnect(self) -> None:
        return


@dataclass
class KinesisOptions:
    """
    Optional knobs for KCube initialization.
    """
    kinesis_dir: Optional[str] = None
    poll_ms: int = 100
    settings_wait_ms: int = 10_000
    device_settings_name: str = "Z912"  # from your toolbox.py
    verbose: bool = False


class KCube(Motor):
    """
    Thorlabs KCube DC Servo (Kinesis) wrapper.

    Adapted from toolbox.KCube with the following improvements:
    - Loads assemblies robustly (env var / common dirs).
    - Builds device list once.
    - Uses explicit imports.
    - Has timeouts on move/home loops.
    - Raises helpful exceptions on init failure.
    """

    def __init__(self, serial_no: Union[str, int], name: str, *, options: Optional[KinesisOptions] = None):
        super().__init__(serial_no, name)
        self.options = options or KinesisOptions()
        ensure_kinesis_loaded(self.options.kinesis_dir)
        build_device_list_once()

        # .NET imports
        from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI  # type: ignore
        from Thorlabs.MotionControl.KCube.DCServoCLI import KCubeDCServo  # type: ignore
        from System import Decimal  # type: ignore
        from System import Action, UInt64  # type: ignore

        self._Decimal = Decimal
        self._Action = Action
        self._UInt64 = UInt64

        self._task_complete = True
        self._task_id = 0
        self._last_known_pos = 0.0
        self._available = False
        self._device = None

        # Determine availability
        devs = [str(d) for d in DeviceManagerCLI.GetDeviceList()]
        self._available = self.serial_no in devs

        if not self._available:
            raise RuntimeError(f"KCube {name} ({self.serial_no}) not found. Connected devices: {devs}")

        # Connect + initialize
        try:
            self._device = KCubeDCServo.CreateKCubeDCServo(self.serial_no)
            self._device.Connect(self.serial_no)

            time.sleep(0.25)
            self._device.StartPolling(int(self.options.poll_ms))
            time.sleep(0.25)
            self._device.EnableDevice()
            time.sleep(0.25)

            if not self._device.IsSettingsInitialized():
                self._device.WaitForSettingsInitialized(int(self.options.settings_wait_ms))
                if not self._device.IsSettingsInitialized():
                    raise RuntimeError("Device settings did not initialize")

            # Load motor configuration and set a known settings profile name (as in toolbox.py)
            m_config = self._device.LoadMotorConfiguration(self.serial_no)

            m_config.DeviceSettingsName = str(self.options.device_settings_name)  # e.g. "Z912"
            m_config.UpdateCurrentConfiguration()
            self._device.SetSettings(self._device.MotorDeviceSettings, True, False)


            self._last_known_pos = float(self._Decimal.ToSingle(self._device.Position))

        except Exception as e:
            # Try to disconnect cleanly if partially initialized
            try:
                self.disconnect()
            except Exception:
                pass
            raise RuntimeError(f"Failed to initialize KCube {name} ({self.serial_no}): {e}") from e

    def is_available(self) -> bool:
        return self._available

    # ---- motion helpers ----

    def _task_complete_callback(self, task_id) -> None:
        try:
            if self._task_id == task_id and task_id > 0:
                self._task_complete = True
        except Exception:
            # best-effort only
            self._task_complete = True

    def get_position(self) -> float:
        if self._device is None:
            return float(self._last_known_pos)
        try:
            # Prefer live position; this works even during motion on most Kinesis devices.
            pos = float(self._Decimal.ToSingle(self._device.Position))
            self._last_known_pos = pos
            return pos
        except Exception:
            return float(self._last_known_pos)

    def move_to(self, pos: float, *, timeout_s: float = 10.0, deadband: float = 1e-4, poll_s: float = 0.05) -> float:
        if self.is_locked:
            return self.get_position()
        if self._device is None:
            raise RuntimeError("Device not connected")

        cur = self.get_position()
        if abs(cur - float(pos)) < deadband:
            return cur

        # Guard against overlapping commands
        if not self._task_complete:
            # refuse to overlap; controller is supposed to serialize
            return self.get_position()

        self._last_known_pos = cur
        self._task_complete = False

        dpos = self._Decimal(float(pos))
        try:
            self._task_id = self._device.MoveTo(dpos, self._Action[self._UInt64](self._task_complete_callback))
        except Exception as e:
            self._task_complete = True
            raise ValueError(f"MoveTo({pos}) failed: {e}") from e

        t0 = time.time()
        while not self._task_complete:
            if time.time() - t0 > float(timeout_s):
                # best effort stop
                try:
                    self.stop()
                except Exception:
                    pass
                self._task_complete = True
                raise TimeoutError(f"Motor {self.name} timed out moving to {pos}")
            time.sleep(float(poll_s))

        return self.get_position()

    def stop(self) -> None:
        if self._device is None:
            return
        # Try common Kinesis stop methods
        if hasattr(self._device, "StopImmediate"):
            self._device.StopImmediate()
        elif hasattr(self._device, "Stop"):
            try:
                self._device.Stop()  # some APIs accept no args
            except TypeError:
                try:
                    self._device.Stop(0)  # sometimes requires a bool or enum; 0 is best-effort
                except Exception:
                    pass
        self._task_complete = True

    def hard_home(self, *, timeout_s: float = 30.0, poll_s: float = 0.2) -> float:
        """
        Hardware home (as in toolbox.py hard_home): run Home() and wait.
        """
        if self.is_locked:
            return self.get_position()
        if self._device is None:
            raise RuntimeError("Device not connected")

        self._task_complete = False
        try:
            self._task_id = self._device.Home(self._Action[self._UInt64](self._task_complete_callback))
        except Exception as e:
            self._task_complete = True
            raise RuntimeError(f"Home() failed: {e}") from e

        t0 = time.time()
        while not self._task_complete:
            if time.time() - t0 > float(timeout_s):
                try:
                    self.stop()
                except Exception:
                    pass
                self._task_complete = True
                raise TimeoutError(f"Motor {self.name} timed out during home")
            time.sleep(float(poll_s))

        return self.get_position()

    def soft_home(self, *, timeout_s: float = 45.0) -> float:
        """
        Soft home: do NOT move to a fixed position.
        For this project, treat soft home as the same as hard home.
        """
        return self.hard_home(timeout_s=timeout_s)

    def set_backlash(self, value: float) -> float:
        if self._device is None:
            raise RuntimeError("Device not connected")
        try:
            self._device.SetBacklash(float(value))
            return float(self._device.GetBacklash())
        except Exception as e:
            raise RuntimeError(f"SetBacklash({value}) failed: {e}") from e

    def disconnect(self) -> None:
        if self._device is None:
            return
        try:
            if hasattr(self._device, "StopPolling"):
                self._device.StopPolling()
        except Exception:
            pass
        try:
            self._device.Disconnect()
        except Exception:
            pass
        self._device = None


class SimulatedMotor(Motor):
    """
    Lightweight motor simulator for UI/raster testing without hardware.
    """
    def __init__(self, serial_no: Union[str, int], name: str, initial: float = 0.0):
        super().__init__(serial_no, name)
        self._pos = float(initial)
        self._available = True

    def is_available(self) -> bool:
        return self._available

    def get_position(self) -> float:
        return float(self._pos)

    def move_to(self, pos: float, *, timeout_s: float = 10.0) -> float:
        if self.is_locked:
            return self.get_position()
        self._pos = float(pos)
        return float(self._pos)

    def soft_home(self, *, timeout_s: float = 1.0) -> float:
        self._pos = 0.0
        return float(self._pos)

    def hard_home(self, *, timeout_s: float = 1.0) -> float:
        self._pos = 0.0
        return float(self._pos)

    def set_backlash(self, value: float) -> float:
        return float(value)
