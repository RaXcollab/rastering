# rastering
# Laser Ablation Rastering GUI

A flexible GUI-based Python software suite for **precision laser ablation rastering** with real-time camera feedback, motor control, and custom path planning. Designed for use with **Thorlabs KCube** motors and **IDS uEye cameras**, it supports multiple rastering modes, calibration workflows, and real-time overlays.

---

## Features

### Real-Time Camera Integration
- Uses IDS `pyueye` camera driver.
- Adjustable exposure control.
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

- Python 3.x  
- `pyqt5`, `pyqtgraph`, `numpy`, `scipy`, `Pillow`  
- [IDS uEye camera SDK](https://en.ids-imaging.com/download-ueye.html)  
- [Thorlabs Kinesis Software](https://www.thorlabs.com/software_pages/ViewSoftwarePage.cfm?Code=Motion_Control) (required for motor control)  

Ensure Kinesis `.dll` references in `toolbox.py` match your system’s installation path.

---

## Quick Start

1. Clone the repo and install dependencies.
2. Connect your uEye camera and Thorlabs KCube motors (update serial numbers in `toolbox.py` if needed).
3. Launch the software:

```bash
python gui2.py
