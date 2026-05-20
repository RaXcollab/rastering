# Building `rotpy` from source for the rastering GUI

This document is the runbook for building **[`rotpy`](https://github.com/matham/rotpy)**
(Cython bindings for the Teledyne FLIR Spinnaker SDK) from source on this Windows
machine, against the installed Spinnaker SDK, into the `rastering` conda env. It is
required because official **`PySpin` has no Python 3.11 wheel** — FLIR ships PySpin for
Python ≤3.10 only — and the `rastering` env is Python 3.11.14.

Last verified Spinnaker SDK release for this procedure: 4.x (vs2015 / vs2017 toolset
libs present at `C:\Program Files\Teledyne\Spinnaker\lib64\{vs2015,vs2017}`, GenTL at
`cti64\vs2015\Spinnaker_GenTL_v140.cti`).

---

## 0. Prerequisites

| Item | Why | How to verify |
|---|---|---|
| **Microsoft C++ Build Tools** (VS2022, "Desktop development with C++" workload — MSVC v143 + Windows SDK) | Cython extension compile (`cl.exe` + headers + linker) | Open **"x64 Native Tools Command Prompt for VS 2022"** from the Start menu; `cl` prints the MSVC banner. |
| **Teledyne FLIR Spinnaker SDK** (already installed) | Provides `Spinnaker.h`, import libs, runtime DLLs, GenTL `.cti` | `C:\Program Files\Teledyne\Spinnaker\include\Spinnaker.h` exists |
| **`rastering` conda env** | Python 3.11.14, numpy 2.4.2, PyQt5; the env `rotpy` will be installed into | `conda env list` shows `rastering`; `python -c "import sys;print(sys.version)"` from the activated env reports 3.11.14 |
| `GENICAM_GENTL64_PATH` system env var | Lets the Spinnaker runtime find the GenTL producer at import | `echo %GENICAM_GENTL64_PATH%` from any prompt — expected to contain `...\Spinnaker\cti64\vs2015` |

If Build Tools are not yet installed, download
**[`vs_BuildTools.exe`](https://visualstudio.microsoft.com/visual-cpp-build-tools/)** (~3 MB
bootstrapper), run as administrator, and check the **"Desktop development with C++"**
workload. ~6 GB on disk. Reboot is not strictly required but does not hurt.

---

## 1. Build

All commands run in an **x64 Native Tools Command Prompt for VS 2022** (start-menu
shortcut — opens cmd.exe with the MSVC environment already configured). Do NOT use a
regular PowerShell or Git Bash window: the toolchain env vars (`INCLUDE`, `LIB`, `PATH`
to `cl.exe`) only exist in this shell.

```bat
:: 1a. Activate the rastering env
call C:\Users\radmo\miniconda3\Scripts\activate.bat rastering
python --version
:: -> Python 3.11.14

:: 1b. Install build deps (Cython 3 + recent pip/setuptools/wheel)
python -m pip install --upgrade pip setuptools wheel "Cython>=3"

:: 1c. Point rotpy's setup.py at the installed Spinnaker SDK.
::     `lib64\vs2015` is the default and matches the v140 GenTL producer.
::     If the link step fails with unresolved symbols or LNK1104/LNK2019 errors,
::     re-try with `lib64\vs2017` (see Troubleshooting #2 below).
set "ROTPY_INCLUDE=C:\Program Files\Teledyne\Spinnaker\include"
set "ROTPY_LIB=C:\Program Files\Teledyne\Spinnaker\lib64\vs2015"

:: 1d. Build + install from source. --no-binary forces sdist (we want the source
::     build, not a wheel); --no-build-isolation lets us reuse the env's Cython.
python -m pip install rotpy --no-binary rotpy --no-build-isolation
```

Expected: build takes ~1–3 minutes. Final line should be
`Successfully installed rotpy-X.Y.Z`.

---

## 2. Runtime: DLL search path + GenTL producer

`rotpy` is just bindings; at runtime it loads `Spinnaker_v140.dll` and friends from the
Spinnaker `bin64\vs2015\` directory, and uses the GenTL producer at
`cti64\vs2015\Spinnaker_GenTL_v140.cti`. Two places this must be set up:

### 2a. System-level (already done on this machine)
- `GENICAM_GENTL64_PATH` → `C:\Program Files\Teledyne\Spinnaker\cti64\vs2015`
  (set by the Spinnaker installer)

### 2b. Per-process (the launcher / `conda activate.d` script must add this)

Either prepend the bin dir to `PATH` before launching Python, OR rely on the
module-level `os.add_dll_directory(...)` self-bootstrap that `camera.py` performs at
import (see `camera.py` `_ensure_spinnaker_runtime()`).

For an interactive `cmd.exe` test:
```bat
set "PATH=C:\Program Files\Teledyne\Spinnaker\bin64\vs2015;%PATH%"
python -c "from rotpy.system import SpinSystem; print(SpinSystem().get_library_version())"
```

Expected: a `(major, minor, build, type)` tuple matching the installed SDK version.

---

## 3. Smoke test

The repo ships `scripts/spin_smoke.py` — minimum-viable enumeration + single-grab test:

```bat
conda activate rastering
python GUIs\rastering\scripts\spin_smoke.py
```

Expected output (cameras must be physically connected and powered):
```
[smoke] Spinnaker library version: 4.x.y.z
[smoke] cameras detected: 1
[smoke] cam[0] DeviceModelName=Blackfly S BFS-PGE-...  serial=2034XXXX
[smoke] begin_acquisition OK
[smoke] grab OK: dtype=uint8 ndim=2 shape=(H, W)
[smoke] SMOKE OK
```

Any of these = **stop and debug before attempting the GUI**:
- `ImportError: No module named rotpy` → step 1 didn't install into the active env.
- `cameras detected: 0` → check the GigE link, NIC subnet, jumbo frames, SpinView.
- `dtype=` not `uint8` or `ndim` not `2` → `PixelFormat` is not `Mono8`; fix node
  defaults in SpinView, or restrict camera.py to Mono8 in v1.

---

## 4. Troubleshooting

### 1. `cl.exe : not recognized` / `error: Microsoft Visual C++ 14.0 or greater is required`
You're not in the x64 Native Tools shell, or the Build Tools workload is missing
the MSVC v143 component. Re-open the **"x64 Native Tools Command Prompt for VS 2022"**
from the Start menu and confirm `where cl` prints a path under
`...\VC\Tools\MSVC\<version>\bin\Hostx64\x64\cl.exe`.

### 2. `LNK1104: cannot open file 'Spinnaker_v140.lib'` or `LNK2019: unresolved external symbol`
The SDK's `vs2015` import libs may not match what the v143 compiler emits. Retry with:
```bat
set "ROTPY_LIB=C:\Program Files\Teledyne\Spinnaker\lib64\vs2017"
python -m pip install rotpy --no-binary rotpy --no-build-isolation --force-reinstall
```
If `vs2017` also fails, the installed Spinnaker SDK predates the libs we need — install
the latest Spinnaker for Windows 10/11 x64 from Teledyne FLIR.

### 3. Cython 3 vs numpy 2 build warning storm
This env runs **numpy 2.4.2** and **Cython 3** — both stable as of writing. If the
build emits hundreds of `numpy/__init__.pxd` deprecation warnings, they are
expected and not errors. If the build actually **fails** with
`AttributeError: module 'numpy' has no attribute 'X'`, pin Cython to a known-good
release (`pip install "Cython==3.0.11"`) and retry.

### 4. Import works, enumeration finds 0 cameras
- Camera not powered, GigE cable unplugged, NIC down.
- Camera on a different subnet than the host NIC. Use Spinnaker's **SpinView** to set
  a persistent / Force IP that matches the NIC subnet.
- Windows Firewall blocking the GigE discovery broadcast (UDP 3956) for the NIC's
  network. Allow `SpinViewWPF.exe` and your Python through the firewall on the
  relevant profile (Private / Domain), or set the NIC profile to Private.
- Jumbo frames (9000 MTU) not enabled on the host NIC. Set it in Device Manager →
  network adapter → Advanced → Jumbo Frame = 9000 (or "9014 bytes").

### 5. `Image Incomplete` errors at runtime
GigE bandwidth saturation. Lower one of (`AcquisitionFrameRate`,
`DeviceLinkThroughputLimit`, sensor pixel-format bit-depth) OR raise the NIC's
receive-buffer size (Device Manager → Advanced → Receive Buffers → 2048+).

### 6. After uninstall / OS update: `ImportError: DLL load failed`
The Spinnaker `bin64\vs2015\` directory is no longer on `PATH` (or the SDK was
upgraded and the v140 DLLs are gone). Re-run an `os.add_dll_directory(...)` to
the bin dir, or re-install Spinnaker. Verify with the inline check in §2b.

---

## 5. Uninstall / rollback

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
| `C:\Program Files\Teledyne\Spinnaker\include\` | SDK headers (Spinnaker.h, etc.) — set as `ROTPY_INCLUDE` |
| `C:\Program Files\Teledyne\Spinnaker\lib64\vs2015\` | Import libs (default `ROTPY_LIB`) |
| `C:\Program Files\Teledyne\Spinnaker\lib64\vs2017\` | Fallback import libs |
| `C:\Program Files\Teledyne\Spinnaker\bin64\vs2015\` | Runtime DLLs (`Spinnaker_v140.dll`, etc.) — must be reachable via `PATH` or `os.add_dll_directory` at import time |
| `C:\Program Files\Teledyne\Spinnaker\cti64\vs2015\Spinnaker_GenTL_v140.cti` | GenTL producer — found via `GENICAM_GENTL64_PATH` |
| `C:\Users\radmo\miniconda3\envs\rastering\Lib\site-packages\rotpy\` | Installed `rotpy` package (post-step 1) |
| `GUIs\rastering\scripts\spin_smoke.py` | This repo's smoke test |
| `GUIs\rastering\camera.py` | New Spinnaker driver (post-Step 2 of migration plan) |
| `GUIs\rastering\camera_ueye.py` | Preserved pre-migration uEye driver (reference only) |
