# rastering
# Laser Ablation Rastering GUI

A flexible GUI-based Python software suite for **precision laser ablation rastering** with real-time camera feedback, motor control, and custom path planning. Designed for use with **Thorlabs KCube** motors and **Teledyne FLIR Blackfly S GigE cameras** (via `rotpy`), it supports multiple rastering modes, calibration workflows, and real-time overlays.

> **Camera migration (2026-05):** the GUI was ported from IDS uEye (`pyueye`)
> to Teledyne FLIR Spinnaker (`rotpy`). The archived uEye driver is preserved
> at [`camera_ueye.py`](camera_ueye.py) for reference / diff only — it is no
> longer imported by the app. See [`docs/ROTPY_BUILD.md`](docs/ROTPY_BUILD.md)
> for installation, troubleshooting, and the bundled Spinnaker runtime
> details.

---

## Features

### Real-Time Camera Integration
- Uses Teledyne FLIR Spinnaker SDK via `rotpy` (PyPI `rotpy` wheel — bundles
  Spinnaker 2.6.0.157 runtime; no separate SDK install required).
- Adjustable exposure (ms), gain (dB), gamma, AOI, pixel format, GigE
  packet size + throughput, Black Level, Defect Correction.
- Live stream with real-time motor position and raster path overlays.

### Motor Control (Thorlabs K-Cubes)
- Control X and Y axes via GUI.
- Homing, jogging, and absolute positioning.
- Configurable backlash correction.

### Rastering Algorithms
Select from:
- **Square Raster (X-priority)**: Row-by-row.
- **Square Raster (Y-priority)**: Column-by-column.
- **Spiral Raster**: Center-outward pattern.
- **Convex Hull Raster**: Raster within any drawn arbitrary region.

Path preview is fully interactive and overlays on the live camera feed.

### Pixel-to-Motor Calibration
- Two-click calibration between camera pixels and motor coordinates.
- Supports saving and reusing calibration data.
- Manual override of scaling and offset.

### Graphical User Interface
- Built with PyQt5 and QtDesigner (`raster_gui2.ui`).
- Spin boxes and sliders for all parameters.
- Image scale and raster preview widgets.
- On-screen click tools to define paths, hulls, or calibration.

### Logging
- Logs motor positions at each raster step.
- JSON format for easy data analysis.
- Automatic timestamped filenames.

### ZMQ Server Integration
- Remote control over TCP with JSON commands.
- Actions supported: move motors, start/stop rastering, query position.
- Easily scriptable from other software.

---

## Requirements

- Python **3.11** (rotpy wheel is cp311-win_amd64)
- `pyqt5`, `pyqtgraph`, `numpy`, `scipy`, `Pillow`, `pyzmq`
- `rotpy` (PyPI install includes the bundled Spinnaker 2.6.0.157 runtime
  -- no separate Spinnaker SDK download required). Full install + DLL-load
  ordering notes in [`docs/ROTPY_BUILD.md`](docs/ROTPY_BUILD.md).
- [Thorlabs Kinesis Software](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control) (required for motor control)

The full Python dependency list is pinned in
[`requirements.txt`](requirements.txt):

```bash
conda activate rastering
pip install -r requirements.txt
```

Ensure Kinesis `.dll` references in `toolbox.py` / `config.py` match your
system's installation path. For Spinnaker GigE cameras, see the NIC
configuration notes in `docs/ROTPY_BUILD.md` (jumbo frames, persistent IP,
Spinnaker Adapter Configuration Utility).

---

## Quick Start

1. Clone the repo and install dependencies via `pip install -r requirements.txt`.
2. Connect your Blackfly GigE camera and Thorlabs KCube motors. Configure
   the camera serial in `config.py` (`APP_CONFIG.camera.serial`); confirm
   motor serial numbers in `config.py` (`hardware.serial_x` /
   `hardware.serial_y`).
3. Confirm the camera enumerates by running the standalone smoke test:

   ```bash
   conda activate rastering
   python scripts/spin_smoke.py
   ```

   Expect `[smoke] SMOKE OK` and a `(1080, 1440)` uint8 frame on a default
   Blackfly S BFS-PGE-16S2M.

4. Launch the software:

```bash
python main_rastering.py
