# Installing `rotpy` for the rastering GUI

This document is the runbook for installing **[`rotpy`](https://github.com/matham/rotpy)**
(Cython bindings for the Teledyne FLIR Spinnaker SDK) into the `rastering` conda env on
this Windows machine. It covers the **wheel-first install** (preferred) and the
**source-build fallback** (currently blocked — see §4).

**TL;DR — what actually works on this machine (2026-05-22):**
1. `conda activate rastering && pip install rotpy` — installs the PyPI cp311-win_amd64
   wheel (rotpy 0.2.1, ships its own Spinnaker 2.6.0.157 runtime in
   `share/rotpy/spinnaker/`). No source build, no MSVC needed.
2. Set `KMP_DUPLICATE_LIB_OK=TRUE` BEFORE the first rotpy/numpy import — required
   workaround for the OpenMP duplicate-runtime conflict (see §3). Already wired into
   `scripts/spin_smoke.py` and will be wired into `camera.py`.
3. The installed Spinnaker SDK 4.3.0.190 at `C:\Program Files\Teledyne\Spinnaker\` is
   **only used by SpinView** and the camera-side drivers. rotpy uses its own bundled
   2.6 runtime — and that's fine: GenICam/GigE-Vision wire protocols are stable across
   SDK versions, so a 4.x-firmware Blackfly S talks happily to a 2.6 client lib.

Earlier guidance in this doc claimed PySpin has no cp311 wheel (true) and concluded
rotpy must therefore be built from source (false — rotpy ships its own wheels).
Corrected here.

---

## 0. Prerequisites

| Item | Why | How to verify |
|---|---|---|
| **`rastering` conda env** | Python 3.11.14, numpy 2.4.2, PyQt5; the env `rotpy` installs into | `conda env list` shows `rastering`; from the activated env, `python -c "import sys;print(sys.version)"` reports `3.11.14`. |
| **Teledyne FLIR Spinnaker SDK** (any 3.x or 4.x, already installed) | Provides the **driver/firmware** layer that lets your NIC + Windows talk to the camera, plus **SpinView** for hardware-side troubleshooting. NOT used directly by `rotpy` at runtime (rotpy uses its own bundled libs). | `C:\Program Files\Teledyne\Spinnaker\include\Spinnaker.h` exists. |
| **NIC jumbo frames / persistent IP** | Required for GigE Blackfly S — without these, enumeration finds 0 cameras. | Device Manager → network adapter → Advanced → Jumbo Frame = 9000. SpinView → camera node → set persistent or Force IP on the NIC's subnet. |

**Microsoft C++ Build Tools are NOT required for the wheel install.** They are only
required for the source-build fallback (§4).

---

## 1. Install (wheel — preferred)

```bat
:: 1a. Activate the rastering env. NOTE: the env lives at
::     C:\Users\radmo\miniconda\envs\rastering   (no '3' in 'miniconda').
call C:\Users\radmo\miniconda\Scripts\activate.bat rastering
python --version
:: -> Python 3.11.14

:: 1b. Install the precompiled wheel.
pip install rotpy
```

Expected: ~5 seconds. Final line: `Successfully installed rotpy-0.2.1`.

The wheel unpacks into:
- `Lib\site-packages\rotpy\` — bindings (`.pyd` files compiled for cp311-win_amd64)
- `share\rotpy\spinnaker\bin\` — Spinnaker 2.6 runtime DLLs (Spinnaker_v140.dll,
  GCBase, GenApi, libiomp5md.dll, etc.)
- `share\rotpy\spinnaker\cti\` — bundled GenTL producer (`FLIR_GenTL_v140.cti`)

`rotpy/__init__.py` prepends `share/rotpy/spinnaker/{bin,cti}` to `PATH`,
`os.add_dll_directory`, and `GENICAM_GENTL64_PATH` automatically on first import.

---

## 2. Sanity check (no camera required)

```bat
conda activate rastering
python -c "import os; os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE'); from rotpy.system import SpinSystem; print(SpinSystem().get_library_version())"
```

Expected: `(2, 6, 0, 157)` (the version of Spinnaker rotpy ships).

If you DON'T set `KMP_DUPLICATE_LIB_OK=TRUE` first AND you import numpy in the same
process, you'll hit `OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll
already initialized` and the process aborts. See §3.

---

## 3. OpenMP duplicate-runtime workaround (REQUIRED)

This env has **three** OpenMP DLLs:

| Path | Provider | Size | Type |
|---|---|---|---|
| `Library\bin\libiomp5md.dll` | conda env (numpy MKL) | 158KB | Intel OMP |
| `Library\bin\libomp.dll` | conda env (LLVM-built deps) | 670KB | LLVM OMP |
| `share\rotpy\spinnaker\bin\libiomp5md.dll` | rotpy bundle | 885KB | Intel OMP |

When `rotpy` is imported, its bundled `libiomp5md.dll` is loaded first. Then numpy
(via MKL) or scipy/sklearn (via LLVM) tries to load a *second* OpenMP runtime →
process aborts with the "OMP Error #15" message.

**Workaround:** set `KMP_DUPLICATE_LIB_OK=TRUE` BEFORE the first rotpy/numpy import.
Intel documents this as "unsupported but works in practice" — it's the same workaround
PyTorch, sklearn, and OpenCV use. Already wired into `scripts/spin_smoke.py`; will be
wired into the new `camera.py` as the very first line after `import os`.

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# ... all other imports follow ...
```

`setdefault` is used (not `[]=`) so an operator override (`set
KMP_DUPLICATE_LIB_OK=FALSE`) before launch still wins.

---

## 4. Source-build fallback (CURRENTLY BLOCKED — do not use)

Source builds against the installed Spinnaker SDK 4.3.0.190 fail at the C++ compile
step:

```
rotpy\_cam_defs\_cam_defs4.cpp(7158): error C2039:
  'SourceSelector_Source0': is not a member of 'Spinnaker'
```

**Root cause:** rotpy's bindings are pre-generated against a Spinnaker SDK whose
`CameraDefs.h` defines `SourceSelector_{Source0,Source1,Source2,All}` (likely the same
2.6.x bundle PyPI ships). The installed 4.3.0.190 renamed those enum values to
`SourceSelector_{Sensor1,Sensor2}`. All three rotpy PyPI versions (0.1.0, 0.2.0, 0.2.1)
were generated against the older header and hit the same compile error against modern
SDKs. Confirmed by testing each version (2026-05-22).

If you need a source build (e.g., to target a cp312+ env where wheels don't exist), the
options are:
1. Patch `rotpy/_cam_defs/_cam_defs4.pyx` to use the new `Sensor1/Sensor2` enum names.
2. Build against rotpy's bundled SDK snapshot at
   `https://github.com/matham/rotpy/releases/download/v0.1.0.dev0/spinnaker_win.7z`
   (point `ROTPY_INCLUDE` / `ROTPY_LIB` at that bundle).
3. Wait for an upstream rotpy release that targets the modern Spinnaker SDK.

Neither is required for the rastering GUI — option (1) of §1 (just `pip install rotpy`)
works today.

---

## 5. Smoke test

`scripts/spin_smoke.py` is the minimum-viable enumeration + single-grab test. It sets
`KMP_DUPLICATE_LIB_OK=TRUE` for you.

```bat
conda activate rastering
python GUIs\rastering\scripts\spin_smoke.py
```

Expected output (cameras must be physically connected, powered, and reachable):
```
[smoke] Spinnaker library version: (2, 6, 0, 157)
[smoke] cameras detected: 1
[smoke] cam[0] DeviceModelName=Blackfly S BFS-PGE-...  serial=2034XXXX
[smoke] PixelFormat set to Mono8
[smoke] begin_acquisition OK
[smoke] grab OK: dtype=uint8 ndim=2 shape=(H, W)
[smoke] SMOKE OK
```

Symptom → diagnosis:
- `ModuleNotFoundError: rotpy` → step 1 didn't install into the active env. Re-check
  `conda env list` and that you ran `conda activate rastering` (the env path is
  `C:\Users\radmo\miniconda\envs\rastering`; no `3` in `miniconda`).
- `OMP: Error #15: ...` → §3 workaround not in effect. spin_smoke.py sets it; if
  you're calling rotpy from another script, it must set the env var too.
- `Spinnaker library version: (2, 6, 0, 157)` then process exits → the rotpy stack is
  fine; any further failure is hardware.
- `cameras detected: 0` → §6 hardware troubleshooting.
- `grab OK` but `dtype` not uint8 / `ndim` not 2 → camera's PixelFormat default isn't
  Mono8. The smoke script does set Mono8 explicitly; if it can't, fix node defaults in
  SpinView or restrict the new `camera.py` to Mono8.

---

## 6. Hardware troubleshooting

### 6.1 `cameras detected: 0`
First **open SpinView** — if SpinView can't see the camera, neither can rotpy. Fix it
in SpinView, then re-run the smoke. Common causes:

- Camera not powered, PoE injector unplugged, GigE cable unplugged, NIC link-down.
- Camera on a different subnet than the host NIC. In SpinView, double-click the camera,
  click *Auto Force IP* or set a persistent IP on the NIC's subnet.
- Windows Firewall blocking the GigE discovery broadcast (UDP 3956) for the NIC's
  network profile. Allow `SpinViewWPF.exe` and your `python.exe` through the firewall
  on the relevant profile (Private / Domain), or change the NIC profile to Private.
- Jumbo frames (9000 MTU) not enabled on the host NIC. Set it in Device Manager →
  network adapter → Advanced → Jumbo Frame / Jumbo Packet = 9000 (or "9014 bytes").
- Two GigE NICs both responding to discovery — make sure each camera is on its own
  dedicated NIC with its own subnet.

### 6.2 `Image Incomplete` errors at runtime
GigE bandwidth saturation. In order of effort:
- Raise the NIC's receive-buffer size (Device Manager → Advanced → Receive Buffers →
  2048+).
- Lower `DeviceLinkThroughputLimit` to leave bandwidth headroom.
- Lower `AcquisitionFrameRate` or use a smaller AOI.
- Drop pixel-format bit-depth (Mono12 → Mono8) if applicable.

### 6.3 `ImportError: DLL load failed` after a Spinnaker SDK uninstall/upgrade
rotpy's bundled libs are self-contained — an SDK uninstall/upgrade should NOT affect
rotpy. If you still hit this, `pip install --force-reinstall rotpy` will repopulate
`share/rotpy/spinnaker/`.

---

## 7. Uninstall / rollback

```bat
conda activate rastering
pip uninstall rotpy
:: pyueye can be left installed -- the preserved camera_ueye.py needs it to import,
:: and the new camera.py never imports it. Remove only if you want to drop the
:: reference driver from being importable.
```

---

## Appendix: where everything lives

| Path | What |
|---|---|
| `C:\Users\radmo\miniconda\envs\rastering\Lib\site-packages\rotpy\` | rotpy bindings (`.pyd` for cp311-win_amd64) |
| `C:\Users\radmo\miniconda\envs\rastering\share\rotpy\spinnaker\bin\` | rotpy's **bundled** Spinnaker 2.6 runtime DLLs |
| `C:\Users\radmo\miniconda\envs\rastering\share\rotpy\spinnaker\cti\` | rotpy's **bundled** GenTL producer (`FLIR_GenTL_v140.cti`) |
| `C:\Program Files\Teledyne\Spinnaker\` | Installed Spinnaker SDK 4.3.0.190 — used by **SpinView and the kernel/NIC drivers**, NOT by rotpy at runtime |
| `C:\Program Files\Teledyne\Spinnaker\bin64\vs2015\Spinnaker_v140.dll` | Installed SDK runtime — **do NOT prepend this to PATH** when using rotpy (would shadow rotpy's bundled v2.6 lib and crash the .pyd files) |
| `GUIs\rastering\scripts\spin_smoke.py` | This repo's smoke test |
| `GUIs\rastering\scripts\build_rotpy.cmd` | Helper for the (currently-blocked) source build path |
| `GUIs\rastering\camera.py` | New Spinnaker driver (post-Step 2 of migration plan) |
| `GUIs\rastering\camera_ueye.py` | Preserved pre-migration uEye driver (reference only) |
