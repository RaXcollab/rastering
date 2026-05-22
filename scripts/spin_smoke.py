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

# OpenMP duplicate-runtime workaround. Three OpenMP DLLs coexist in this env:
#   - share/rotpy/spinnaker/bin/libiomp5md.dll (Intel OMP, 885KB, rotpy bundle)
#   - Library/bin/libiomp5md.dll               (Intel OMP, 158KB, numpy MKL)
#   - Library/bin/libomp.dll                   (LLVM OMP, 670KB, conda env)
# rotpy import loads its bundled libiomp5md.dll first; then numpy/scipy
# transitively load libomp.dll -> "OMP: Error #15: ... already initialized"
# and the process aborts. KMP_DUPLICATE_LIB_OK lets Intel OMP accept the
# second runtime. Documented by Intel as "unsupported but works for most
# use cases" -- standard workaround used by PyTorch, sklearn, etc.
# MUST be set BEFORE the first rotpy/numpy import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# NOTE: do NOT bootstrap the installed Teledyne Spinnaker SDK paths here.
# rotpy ships its own Spinnaker runtime libs (currently v2.6.0.157, bundled
# in the cp311 wheel under ``{sys.prefix}/share/rotpy/spinnaker/{bin,cti}``)
# and ``rotpy/__init__.py`` prepends those to PATH / add_dll_directory /
# GENICAM_GENTL64_PATH on import. Prepending the installed SDK's vs2015 bin
# (e.g. 4.3.0.190) clobbers that, causing rotpy's .pyd files (built against
# v2.6 headers) to load v4.3 DLLs at runtime -> ABI mismatch -> silent crash
# with no Python-level error. Let rotpy own its DLL search.


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
        # rotpy 0.2.1 image API: get_status() returns one of the 14 strings
        # in rotpy.names.spin.img_status_values; "no_error" (=0) means the
        # frame is OK. get_completed() exists but does NOT have the obvious
        # "not is_incomplete()" semantic -- empirically (2026-05-22 first
        # hardware run, BFS-PGE-16S2M) it can return False while
        # get_status() reports "no_error". The canonical quality check is
        # get_status(). See docs/ROTPY_API.md for the full enum + class
        # surface.
        status = img.get_status()
        if status != "no_error":
            _fail(
                f"first image returned with status={status!r}",
                hint="Likely GigE bandwidth / packet-loss; see docs/ROTPY_BUILD.md #6.2.",
            )

        # rotpy 0.2.1: get_image_data() returns a fresh bytearray that
        # OWNS its memory (it's already a copy of the SDK buffer at the C++
        # ImagePtr layer). np.frombuffer(bytearray) creates an ndarray whose
        # .base is the bytearray -- the SDK buffer can be released safely.
        # The AUDIT:S1 "frame.base is None" check demands a stricter
        # standalone-ndarray invariant; we satisfy it with an explicit
        # .copy() so the emitted frame never references any intermediate.
        h = img.get_height()
        w = img.get_width()
        data = img.get_image_data()  # bytearray, owned copy of SDK buffer
        if len(data) != h * w:
            _fail(
                f"buffer size mismatch: got {len(data)} bytes, expected {h*w}",
                hint="PixelFormat may not be Mono8.",
            )
        # One-pass copy: frombuffer (view onto bytearray) -> reshape (view)
        # -> .copy() (fresh owned ndarray, .base is None).
        frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w).copy()

        # Audit N2 invariant: uint8 2-D, contiguous, owns its memory.
        assert frame.dtype == np.uint8, f"dtype {frame.dtype} != uint8"
        assert frame.ndim == 2, f"ndim {frame.ndim} != 2"
        assert frame.flags["C_CONTIGUOUS"], "frame is not C-contiguous"
        assert frame.base is None, "frame still references intermediate buffer"

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
