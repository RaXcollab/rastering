"""Spinnaker / rotpy minimum-viable smoke test.

Gate for Step 2 of the camera migration plan. Confirms that:

1. ``rotpy`` is importable in the active env (`rastering` conda env).
2. The Spinnaker runtime DLLs and GenTL producer are discoverable.
3. At least one camera enumerates.
4. ``init_cam`` -> set ``PixelFormat=Mono8`` -> ``begin_acquisition`` ->
   ``get_next_image`` -> ``release`` -> ``end_acquisition`` -> ``deinit_cam``
   completes without raising.
5. The returned frame is a contiguous **uint8 2-D ndarray** (the format the
   GUI's pyqtgraph view consumes with ``autoLevels=False`` -- per audit N2).

Run from the activated rastering env:

    conda activate rastering
    python GUIs\\rastering\\scripts\\spin_smoke.py

Exits 0 on success ("SMOKE OK"). Non-zero on any failure with a friendly
message pointing at the troubleshooting section of ``docs/ROTPY_BUILD.md``.

This script must NOT depend on PyQt5, BLACS, or any rastering-internal module --
it stands alone, so it can be run before the new ``camera.py`` lands.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# -----------------------------------------------------------------------------
# Spinnaker runtime bootstrap: make the SDK DLLs and GenTL producer findable
# before importing rotpy. Matches camera.py's _ensure_spinnaker_runtime().
# Safe to skip silently if the SDK isn't at the expected path -- rotpy will
# raise a clear ImportError downstream.
# -----------------------------------------------------------------------------
_SDK_ROOT = Path(r"C:\Program Files\Teledyne\Spinnaker")
_BIN = _SDK_ROOT / "bin64" / "vs2015"
_GENTL = _SDK_ROOT / "cti64" / "vs2015"

if _BIN.is_dir():
    # add_dll_directory is the modern Windows DLL-search-path mechanism
    # (PATH alone is no longer honored for extension modules in py>=3.8).
    try:
        os.add_dll_directory(str(_BIN))
    except (AttributeError, OSError):
        pass  # py<3.8 or path missing; fall back to PATH
    # Belt-and-suspenders for tooling that still reads PATH.
    os.environ["PATH"] = f"{_BIN};{os.environ.get('PATH', '')}"

if _GENTL.is_dir() and "GENICAM_GENTL64_PATH" not in os.environ:
    os.environ["GENICAM_GENTL64_PATH"] = str(_GENTL)


def _fail(msg: str, *, hint: str = "", code: int = 1) -> "NoReturn":
    print(f"[smoke] FAIL: {msg}", file=sys.stderr)
    if hint:
        print(f"[smoke] hint: {hint}", file=sys.stderr)
    print(
        "[smoke] see docs/ROTPY_BUILD.md (Troubleshooting) for diagnosis steps.",
        file=sys.stderr,
    )
    sys.exit(code)


# -----------------------------------------------------------------------------
# rotpy import
# -----------------------------------------------------------------------------
try:
    from rotpy.system import SpinSystem
    from rotpy.camera import CameraList
except ImportError as e:
    _fail(
        f"rotpy not importable: {e}",
        hint=(
            "Build + install rotpy from source per docs/ROTPY_BUILD.md (step 1). "
            "Confirm you are in the rastering conda env: `conda info --envs`."
        ),
    )


try:
    import numpy as np
except ImportError as e:
    _fail(f"numpy not importable: {e}", hint="Activate the rastering env first.")


# -----------------------------------------------------------------------------
# Library info
# -----------------------------------------------------------------------------
try:
    system = SpinSystem()
    libver = system.get_library_version()
    print(f"[smoke] Spinnaker library version: {libver}")
except Exception as e:
    _fail(
        f"SpinSystem() failed: {e}",
        hint=(
            "rotpy imported but the Spinnaker runtime did not load. "
            "Check GENICAM_GENTL64_PATH and the bin64\\vs2015 directory."
        ),
    )


# -----------------------------------------------------------------------------
# Enumerate cameras
# -----------------------------------------------------------------------------
try:
    cameras = CameraList.create_from_system(system, update_cams=True, update_interfaces=True)
    n = cameras.get_size()
    print(f"[smoke] cameras detected: {n}")
except Exception as e:
    _fail(f"CameraList.create_from_system failed: {e}")

if n == 0:
    _fail(
        "no cameras enumerated",
        hint=(
            "Check the GigE link, NIC subnet, jumbo frames, and SpinView visibility. "
            "Camera must be reachable from this host before rotpy can find it."
        ),
        code=2,
    )


# -----------------------------------------------------------------------------
# Init first camera + identify
# -----------------------------------------------------------------------------
cam = cameras.create_camera_by_index(0)

# Use try/finally to guarantee deinit even on mid-init failures.
init_ok = False
acq_ok = False
try:
    cam.init_cam()
    init_ok = True

    # Best-effort identifiers; surface failures as diagnostic, not fatal.
    def _nv(name: str) -> str:
        try:
            node = getattr(cam.camera_nodes, name)
            return node.get_node_value() if hasattr(node, "get_node_value") else str(node)
        except Exception as e:
            return f"<unavailable: {e.__class__.__name__}>"

    model = _nv("DeviceModelName")
    serial = _nv("DeviceSerialNumber")
    print(f"[smoke] cam[0] DeviceModelName={model}  serial={serial}")

    # Force Mono8 so the smoke is consistent (the rastering GUI expects uint8 2-D).
    try:
        cam.camera_nodes.PixelFormat.set_node_value_from_str("Mono8")
        print("[smoke] PixelFormat set to Mono8")
    except Exception as e:
        print(f"[smoke] WARNING: could not force PixelFormat=Mono8: {e}", file=sys.stderr)

    # Acquire one frame.
    cam.begin_acquisition()
    acq_ok = True
    print("[smoke] begin_acquisition OK")

    img = cam.get_next_image(timeout=5)  # seconds in rotpy
    try:
        if img.is_incomplete():
            _fail(
                "first image returned incomplete",
                hint="Likely GigE bandwidth issue; see docs/ROTPY_BUILD.md #5.",
            )

        # Get the raw buffer + dimensions and copy into a contiguous ndarray.
        # camera.py will use np.ascontiguousarray to enforce .base is None.
        h = img.get_height()
        w = img.get_width()
        data = img.get_image_data()  # bytes-like, view into SDK buffer
        # Reshape into 2-D; copy (np.frombuffer makes a view, but assigning
        # to ndarray + .copy() detaches us from the SDK buffer ownership).
        flat = np.frombuffer(bytes(data), dtype=np.uint8)
        if flat.size != h * w:
            _fail(
                f"buffer size mismatch: got {flat.size} bytes, expected {h*w}",
                hint="PixelFormat may not be Mono8.",
            )
        frame = np.ascontiguousarray(flat.reshape(h, w))

        # Audit N2 invariant: uint8 2-D, contiguous, owns its memory.
        assert frame.dtype == np.uint8, f"dtype {frame.dtype} != uint8"
        assert frame.ndim == 2, f"ndim {frame.ndim} != 2"
        assert frame.flags["C_CONTIGUOUS"], "frame is not C-contiguous"
        assert frame.base is None, "frame still references SDK buffer (must copy before release)"

        print(f"[smoke] grab OK: dtype={frame.dtype} ndim={frame.ndim} shape={frame.shape}")
    finally:
        # Release the SDK image back to the buffer pool BEFORE end_acquisition.
        try:
            img.release()
        except Exception as e:
            print(f"[smoke] WARNING: img.release() raised: {e}", file=sys.stderr)

except SystemExit:
    raise
except Exception as e:
    traceback.print_exc()
    _fail(f"acquisition path failed: {e!r}")

finally:
    # Idempotent teardown -- matches the plan's close() contract.
    if acq_ok:
        try:
            cam.end_acquisition()
        except Exception as e:
            print(f"[smoke] WARNING: end_acquisition raised: {e}", file=sys.stderr)
    if init_ok:
        try:
            cam.deinit_cam()
        except Exception as e:
            print(f"[smoke] WARNING: deinit_cam raised: {e}", file=sys.stderr)

print("[smoke] SMOKE OK")
sys.exit(0)
