"""
ui.py

Qt UI layer built from the Qt Designer file raster_gui2.ui.

Responsibilities:
- Load the .ui layout.
- Host the PyQtGraph display (ImageItem + overlays).
- Collect user intent (clicks, buttons) and forward to controller via its request_* API.
- Render controller state via signals (status, positions, calibration, raster).

Non-goals:
- No direct motor/DLL calls.
- No ZMQ networking.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from raster_paths import RasterSpec, iter_path_from_spec, collect_points
from camera import UEyeCameraThread, UEyeConfig
from PyQt5 import QtCore, QtGui, QtWidgets, uic

UI_FILE = "raster_gui.ui"

# Optional: read default flip settings from config.py if available
try:
    import config as _config
except Exception:
    _config = None


TargetXY = Tuple[float, float]


# =====================================================================
# Camera Settings Dock Widget
# =====================================================================

class CameraSettingsDock(QtWidgets.QDockWidget):
    """
    Dockable panel exposing live camera controls, modeled after uEye Cockpit:
      - Pixel clock (combo with valid values from camera)
      - Exposure (slider + spinbox with live range from camera)
      - Gain (slider + spinbox, 0-100)
      - Gain boost (checkbox)
      - Gamma (slider + spinbox)
      - AOI (width, height, start_x, start_y with sliders + Apply button)
      - Rotation (combo: 0/90/180/270)
      - Flip X / Flip Y (checkboxes)
      - Save Settings button

    Ranges are populated by camera_info_signal from the camera thread.
    """

    # Emitted when user changes rotation or flip (display-only transforms)
    rotation_changed = QtCore.pyqtSignal(int)    # k value for np.rot90
    flip_x_changed = QtCore.pyqtSignal(bool)
    flip_y_changed = QtCore.pyqtSignal(bool)

    # Emitted when user clicks Save Settings
    save_requested = QtCore.pyqtSignal()
    # Emitted when user clicks Load Config
    load_requested = QtCore.pyqtSignal()
    # Emitted when user clicks Revert
    revert_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Camera Settings", parent)
        self.setObjectName("CameraSettingsDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        self._building = False  # suppress signals during programmatic updates

        # Put everything in a scroll area so it fits on small screens
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Pixel Clock ---
        grp_timing = QtWidgets.QGroupBox("Timing")
        fl_timing = QtWidgets.QFormLayout(grp_timing)

        self.pclk_combo = QtWidgets.QComboBox()
        self.pclk_combo.setToolTip("Pixel clock (MHz). Changes available exposure & FPS ranges.")
        fl_timing.addRow("Pixel Clock:", self.pclk_combo)

        # --- Timing mode selector ---
        self.timing_mode_combo = QtWidgets.QComboBox()
        self.timing_mode_combo.addItem("FPS Control")
        self.timing_mode_combo.addItem("Exposure Control")
        self.timing_mode_combo.setToolTip(
            "FPS Control: set target FPS, exposure range is constrained.\n"
            "Exposure Control: set exposure freely, FPS adjusts automatically."
        )
        fl_timing.addRow("Timing Mode:", self.timing_mode_combo)

        self.fps_range_label = QtWidgets.QLabel("FPS range: --")
        self.fps_range_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_timing.addRow(self.fps_range_label)

        # --- FPS (slider + spin) ---
        fps_row = QtWidgets.QHBoxLayout()
        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setMinimum(0)
        self.fps_slider.setMaximum(10000)  # mapped to actual fps range
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setRange(0.1, 200.0)
        self.fps_spin.setValue(20.0)
        fps_row.addWidget(self.fps_slider, 3)
        fps_row.addWidget(self.fps_spin, 1)
        self.fps_label = QtWidgets.QLabel("Target FPS:")
        fl_timing.addRow(self.fps_label, fps_row)

        # --- Exposure (slider + spin) ---
        exp_row = QtWidgets.QHBoxLayout()
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exposure_slider.setMinimum(0)
        self.exposure_slider.setMaximum(10000)  # mapped to actual ms range
        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setDecimals(3)
        self.exposure_spin.setSuffix(" ms")
        exp_row.addWidget(self.exposure_slider, 3)
        exp_row.addWidget(self.exposure_spin, 1)
        fl_timing.addRow("Exposure:", exp_row)

        self.exp_range_label = QtWidgets.QLabel("Range: --")
        self.exp_range_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_timing.addRow(self.exp_range_label)

        layout.addWidget(grp_timing)

        # --- Gain / Gamma ---
        grp_analog = QtWidgets.QGroupBox("Analog")
        fl_analog = QtWidgets.QFormLayout(grp_analog)

        # Gain (slider + spin)
        gain_row = QtWidgets.QHBoxLayout()
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 100)
        self.gain_spin = QtWidgets.QSpinBox()
        self.gain_spin.setRange(0, 100)
        gain_row.addWidget(self.gain_slider, 3)
        gain_row.addWidget(self.gain_spin, 1)
        fl_analog.addRow("Gain:", gain_row)

        self.gain_boost_cb = QtWidgets.QCheckBox("Gain Boost")
        fl_analog.addRow(self.gain_boost_cb)

        # Gamma (slider + spin)
        gamma_row = QtWidgets.QHBoxLayout()
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(1, 1000)       # maps to 0.01 – 10.00
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 10.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(0.05)
        gamma_row.addWidget(self.gamma_slider, 3)
        gamma_row.addWidget(self.gamma_spin, 1)
        fl_analog.addRow("Gamma:", gamma_row)

        layout.addWidget(grp_analog)

        # --- AOI ---
        grp_aoi = QtWidgets.QGroupBox("AOI (Region of Interest)")
        fl_aoi = QtWidgets.QFormLayout(grp_aoi)

        self.aoi_width_spin = QtWidgets.QSpinBox()
        self.aoi_width_spin.setRange(4, 4096)
        self.aoi_width_spin.setSingleStep(4)
        fl_aoi.addRow("Width:", self.aoi_width_spin)

        self.aoi_height_spin = QtWidgets.QSpinBox()
        self.aoi_height_spin.setRange(4, 4096)
        self.aoi_height_spin.setSingleStep(4)
        fl_aoi.addRow("Height:", self.aoi_height_spin)

        # Start X (slider + spin)
        aoi_x_row = QtWidgets.QHBoxLayout()
        self.aoi_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aoi_x_slider.setRange(0, 4096)
        self.aoi_x_slider.setSingleStep(4)
        self.aoi_x_slider.setPageStep(40)
        self.aoi_x_spin = QtWidgets.QSpinBox()
        self.aoi_x_spin.setRange(0, 4096)
        self.aoi_x_spin.setSingleStep(4)
        self.aoi_x_spin.setToolTip("Absolute sensor Start X (same as uEye Cockpit)")
        aoi_x_row.addWidget(self.aoi_x_slider, 3)
        aoi_x_row.addWidget(self.aoi_x_spin, 1)
        fl_aoi.addRow("Start X:", aoi_x_row)

        # Start Y (slider + spin)
        aoi_y_row = QtWidgets.QHBoxLayout()
        self.aoi_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aoi_y_slider.setRange(0, 4096)
        self.aoi_y_slider.setSingleStep(4)
        self.aoi_y_slider.setPageStep(40)
        self.aoi_y_spin = QtWidgets.QSpinBox()
        self.aoi_y_spin.setRange(0, 4096)
        self.aoi_y_spin.setSingleStep(4)
        self.aoi_y_spin.setToolTip("Absolute sensor Start Y (same as uEye Cockpit)")
        aoi_y_row.addWidget(self.aoi_y_slider, 3)
        aoi_y_row.addWidget(self.aoi_y_spin, 1)
        fl_aoi.addRow("Start Y:", aoi_y_row)

        aoi_btn_row = QtWidgets.QHBoxLayout()
        self.aoi_apply_btn = QtWidgets.QPushButton("Apply AOI")
        self.aoi_apply_btn.setToolTip("Apply new AOI (requires brief camera reinit)")
        self.aoi_center_btn = QtWidgets.QPushButton("Center")
        self.aoi_center_btn.setToolTip("Center the AOI on the sensor")
        aoi_btn_row.addWidget(self.aoi_apply_btn)
        aoi_btn_row.addWidget(self.aoi_center_btn)
        fl_aoi.addRow(aoi_btn_row)

        self.sensor_label = QtWidgets.QLabel("Sensor: --")
        self.sensor_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_aoi.addRow(self.sensor_label)

        layout.addWidget(grp_aoi)

        # --- Display transforms ---
        grp_disp = QtWidgets.QGroupBox("Display")
        fl_disp = QtWidgets.QFormLayout(grp_disp)

        self.rotation_combo = QtWidgets.QComboBox()
        self.rotation_combo.addItems(["0°", "90° CW", "180°", "90° CCW"])
        fl_disp.addRow("Rotation:", self.rotation_combo)

        flip_row = QtWidgets.QHBoxLayout()
        self.flip_x_cb = QtWidgets.QCheckBox("Flip X")
        self.flip_y_cb = QtWidgets.QCheckBox("Flip Y")
        flip_row.addWidget(self.flip_x_cb)
        flip_row.addWidget(self.flip_y_cb)
        fl_disp.addRow(flip_row)

        layout.addWidget(grp_disp)

        # --- Config label ---
        self.config_label = QtWidgets.QLabel("Loaded: (none)")
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet("color: gray; font-size: 10px; padding: 2px;")
        layout.addWidget(self.config_label)

        # --- Save / Load / Revert buttons ---
        self.save_btn = QtWidgets.QPushButton("💾  Save Settings")
        self.save_btn.setToolTip("Save current camera + display settings to .ini for next launch")
        self.save_btn.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.save_btn)

        self.load_btn = QtWidgets.QPushButton("📂  Load Config")
        self.load_btn.setToolTip("Load camera settings from an .ini file")
        self.load_btn.setStyleSheet("padding: 4px;")
        layout.addWidget(self.load_btn)

        self.revert_btn = QtWidgets.QPushButton("↩  Revert")
        self.revert_btn.setToolTip("Revert to the most recently loaded/saved .ini")
        self.revert_btn.setStyleSheet("padding: 4px;")
        layout.addWidget(self.revert_btn)

        layout.addStretch()

        scroll.setWidget(container)
        self.setWidget(scroll)

        # Internal state for slider <-> spin sync
        self._exp_min = 0.01
        self._exp_max = 1000.0
        self._fps_min = 0.1
        self._fps_max = 200.0
        self._gamma_min = 0.01
        self._gamma_max = 2.2

        # Sensor dimensions (populated from camera_info)
        self._sensor_w = 1280
        self._sensor_h = 1024

        # --- Internal wiring ---
        self._wire_signals()

    def _wire_signals(self) -> None:
        # Timing mode
        self.timing_mode_combo.currentIndexChanged.connect(self._on_timing_mode_changed)
        # Apply initial enable/disable state
        self._on_timing_mode_changed(self.timing_mode_combo.currentIndex())

        # FPS: keep slider and spinbox in sync
        self.fps_slider.valueChanged.connect(self._fps_slider_to_spin)
        self.fps_spin.valueChanged.connect(self._fps_spin_to_slider)

        # Exposure: keep slider and spinbox in sync
        self.exposure_slider.valueChanged.connect(self._exp_slider_to_spin)
        self.exposure_spin.valueChanged.connect(self._exp_spin_to_slider)

        # Gain: keep slider and spinbox in sync
        self.gain_slider.valueChanged.connect(self.gain_spin.setValue)
        self.gain_spin.valueChanged.connect(self.gain_slider.setValue)

        # Gamma: keep slider and spinbox in sync
        self.gamma_slider.valueChanged.connect(self._gamma_slider_to_spin)
        self.gamma_spin.valueChanged.connect(self._gamma_spin_to_slider)

        # AOI sliders <-> spinboxes
        self.aoi_x_slider.valueChanged.connect(self._aoi_x_slider_to_spin)
        self.aoi_x_spin.valueChanged.connect(self._aoi_x_spin_to_slider)
        self.aoi_y_slider.valueChanged.connect(self._aoi_y_slider_to_spin)
        self.aoi_y_spin.valueChanged.connect(self._aoi_y_spin_to_slider)

        # AOI center button
        self.aoi_center_btn.clicked.connect(self._center_aoi)

        # Display transform signals
        self.rotation_combo.currentIndexChanged.connect(self._emit_rotation)
        self.flip_x_cb.toggled.connect(self.flip_x_changed.emit)
        self.flip_y_cb.toggled.connect(self.flip_y_changed.emit)

        # Save / Load / Revert buttons
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.load_btn.clicked.connect(self.load_requested.emit)
        self.revert_btn.clicked.connect(self.revert_requested.emit)

    # --- Timing mode ---

    def _on_timing_mode_changed(self, index: int) -> None:
        """Enable/disable FPS and Exposure controls based on timing mode."""
        fps_mode = (index == 0)
        # FPS controls: enabled in FPS mode, greyed in Exposure mode
        self.fps_slider.setEnabled(fps_mode)
        self.fps_spin.setEnabled(fps_mode)
        # Exposure controls: enabled in Exposure mode, greyed in FPS mode
        self.exposure_slider.setEnabled(not fps_mode)
        self.exposure_spin.setEnabled(not fps_mode)

    # --- FPS slider <-> spin sync (maps 0–10000 → fps_min–fps_max) ---

    def _fps_slider_to_spin(self, slider_val: int) -> None:
        if self._building:
            return
        frac = slider_val / 10000.0
        fps = self._fps_min + frac * (self._fps_max - self._fps_min)
        self._building = True
        self.fps_spin.setValue(round(fps, 2))
        self._building = False

    def _fps_spin_to_slider(self, fps: float) -> None:
        if self._building:
            return
        rng = self._fps_max - self._fps_min
        if rng > 0:
            frac = (fps - self._fps_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self._building = True
        self.fps_slider.setValue(int(round(frac * 10000)))
        self._building = False

    # --- Exposure slider <-> spin sync (maps 0–10000 → exp_min–exp_max) ---

    def _exp_slider_to_spin(self, slider_val: int) -> None:
        if self._building:
            return
        frac = slider_val / 10000.0
        ms = self._exp_min + frac * (self._exp_max - self._exp_min)
        self._building = True
        self.exposure_spin.setValue(ms)
        self._building = False

    def _exp_spin_to_slider(self, ms: float) -> None:
        if self._building:
            return
        rng = self._exp_max - self._exp_min
        if rng > 0:
            frac = (ms - self._exp_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self._building = True
        self.exposure_slider.setValue(int(round(frac * 10000)))
        self._building = False

    # --- Gamma slider <-> spin sync (slider 1–1000 → 0.01–10.00) ---

    def _gamma_slider_to_spin(self, slider_val: int) -> None:
        if self._building:
            return
        frac = (slider_val - 1) / 999.0
        gamma = self._gamma_min + frac * (self._gamma_max - self._gamma_min)
        self._building = True
        self.gamma_spin.setValue(round(gamma, 2))
        self._building = False

    def _gamma_spin_to_slider(self, gamma: float) -> None:
        if self._building:
            return
        rng = self._gamma_max - self._gamma_min
        if rng > 0:
            frac = (gamma - self._gamma_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self._building = True
        self.gamma_slider.setValue(int(round(1 + frac * 999)))
        self._building = False

    # --- AOI X slider <-> spin sync (aligned to step of 4) ---

    def _aoi_x_slider_to_spin(self, v: int) -> None:
        if self._building:
            return
        v = (v // 4) * 4
        self._building = True
        self.aoi_x_spin.setValue(v)
        self._building = False

    def _aoi_x_spin_to_slider(self, v: int) -> None:
        if self._building:
            return
        self._building = True
        self.aoi_x_slider.setValue(v)
        self._building = False

    # --- AOI Y slider <-> spin sync ---

    def _aoi_y_slider_to_spin(self, v: int) -> None:
        if self._building:
            return
        v = (v // 4) * 4
        self._building = True
        self.aoi_y_spin.setValue(v)
        self._building = False

    def _aoi_y_spin_to_slider(self, v: int) -> None:
        if self._building:
            return
        self._building = True
        self.aoi_y_slider.setValue(v)
        self._building = False

    def _center_aoi(self) -> None:
        w = self.aoi_width_spin.value()
        h = self.aoi_height_spin.value()
        cx = max(0, (self._sensor_w - w) // 2)
        cy = max(0, (self._sensor_h - h) // 2)
        # Align to 4
        cx = (cx // 4) * 4
        cy = (cy // 4) * 4
        self.aoi_x_spin.setValue(cx)
        self.aoi_y_spin.setValue(cy)

    def _emit_rotation(self, index: int) -> None:
        # Combo order: 0°, 90° CW, 180°, 90° CCW
        # np.rot90 k: 0, -1 (CW), 2 (180), 1 (CCW)
        k_map = {0: 0, 1: -1, 2: 2, 3: 1}
        self.rotation_changed.emit(k_map.get(index, 0))

    # --- Populate from camera_info dict ---

    def update_from_camera_info(self, info: dict) -> None:
        """
        Called when camera thread emits camera_info_signal.
        Updates all control ranges and current values without firing
        change signals back to the camera.
        """
        self._building = True
        try:
            # Sensor
            sw = info.get("sensor_width", 1280)
            sh = info.get("sensor_height", 1024)
            self._sensor_w = sw
            self._sensor_h = sh
            self.sensor_label.setText(f"Sensor: {sw} × {sh}")
            self.aoi_width_spin.setMaximum(sw)
            self.aoi_height_spin.setMaximum(sh)
            self.aoi_x_spin.setMaximum(max(0, sw - 4))
            self.aoi_y_spin.setMaximum(max(0, sh - 4))
            self.aoi_x_slider.setMaximum(max(0, sw - 4))
            self.aoi_y_slider.setMaximum(max(0, sh - 4))

            # Pixel clocks
            clocks = info.get("pixel_clocks", [])
            cur_pclk = info.get("pixel_clock", 0)
            self.pclk_combo.clear()
            for c in clocks:
                self.pclk_combo.addItem(f"{c} MHz", c)
            for i in range(self.pclk_combo.count()):
                if self.pclk_combo.itemData(i) == cur_pclk:
                    self.pclk_combo.setCurrentIndex(i)
                    break

            # FPS range + actual
            fps_min = info.get("fps_min", 0.1)
            fps_max = info.get("fps_max", 200.0)
            fps_cur = info.get("fps", 0.0)
            self._fps_min = fps_min
            self._fps_max = fps_max
            self.fps_range_label.setText(f"FPS range: {fps_min:.1f} – {fps_max:.1f}")
            self.fps_spin.setRange(fps_min, fps_max)
            self.fps_spin.setValue(fps_cur)
            # Sync slider
            fps_rng = fps_max - fps_min
            if fps_rng > 0:
                fps_frac = (fps_cur - fps_min) / fps_rng
            else:
                fps_frac = 0.0
            self.fps_slider.setValue(int(round(max(0, min(1, fps_frac)) * 10000)))

            # Exposure
            self._exp_min = info.get("exposure_min", 0.01)
            self._exp_max = info.get("exposure_max", 1000.0)
            exp_inc = info.get("exposure_inc", 0.01)
            exp_cur = info.get("exposure", 30.0)

            self.exposure_spin.setRange(self._exp_min, self._exp_max)
            self.exposure_spin.setSingleStep(max(exp_inc, 0.001))
            self.exposure_spin.setValue(exp_cur)
            # Sync slider
            rng = self._exp_max - self._exp_min
            if rng > 0:
                frac = (exp_cur - self._exp_min) / rng
            else:
                frac = 0.0
            self.exposure_slider.setValue(int(round(max(0, min(1, frac)) * 10000)))

            self.exp_range_label.setText(
                f"Range: {self._exp_min:.3f} – {self._exp_max:.3f} ms"
            )

            # Gain
            self.gain_spin.setValue(info.get("gain", 0))
            self.gain_slider.setValue(info.get("gain", 0))
            self.gain_boost_cb.setChecked(info.get("gain_boost", False))

            # Gamma
            self._gamma_min = info.get("gamma_min", 0.01)
            self._gamma_max = info.get("gamma_max", 2.2)
            self.gamma_spin.setRange(self._gamma_min, self._gamma_max)
            gamma_cur = info.get("gamma", 1.0)
            self.gamma_spin.setValue(gamma_cur)
            # Sync slider
            grng = self._gamma_max - self._gamma_min
            if grng > 0:
                gfrac = (gamma_cur - self._gamma_min) / grng
            else:
                gfrac = 0.0
            self.gamma_slider.setValue(int(round(1 + max(0, min(1, gfrac)) * 999)))

            # AOI
            aoi_w = info.get("aoi_width", sw)
            aoi_h = info.get("aoi_height", sh)
            aoi_x = info.get("aoi_x", 0)
            aoi_y = info.get("aoi_y", 0)
            self.aoi_width_spin.setValue(aoi_w)
            self.aoi_height_spin.setValue(aoi_h)
            self.aoi_x_spin.setValue(aoi_x)
            self.aoi_y_spin.setValue(aoi_y)
            self.aoi_x_slider.setValue(aoi_x)
            self.aoi_y_slider.setValue(aoi_y)

        finally:
            self._building = False

    def get_current_settings(self) -> dict:
        """
        Read all current dock values into a dict for saving.
        """
        # Rotation k from combo index
        k_map = {0: 0, 1: -1, 2: 2, 3: 1}
        rot_k = k_map.get(self.rotation_combo.currentIndex(), 0)

        return {
            "pixel_clock": self.pclk_combo.currentData() or 0,
            "timing_mode": "exposure" if self.timing_mode_combo.currentIndex() == 1 else "fps",
            "target_fps": self.fps_spin.value(),
            "exposure": self.exposure_spin.value(),
            "gain": self.gain_spin.value(),
            "gain_boost": self.gain_boost_cb.isChecked(),
            "gamma": self.gamma_spin.value(),
            "aoi_width": self.aoi_width_spin.value(),
            "aoi_height": self.aoi_height_spin.value(),
            "aoi_x": self.aoi_x_spin.value(),
            "aoi_y": self.aoi_y_spin.value(),
            "rotation_k": rot_k,
            "flip_x": self.flip_x_cb.isChecked(),
            "flip_y": self.flip_y_cb.isChecked(),
        }

    def set_loaded_config_label(self, path: str) -> None:
        """Update the config label to show the loaded INI filename."""
        if path:
            self.config_label.setText(f"Loaded: {os.path.basename(path)}")
            self.config_label.setToolTip(path)
        else:
            self.config_label.setText("Loaded: (none)")
            self.config_label.setToolTip("")

    def connect_to_camera_thread(self, cam_thread) -> None:
        """
        Wire dock controls → camera thread parameter slots.
        Call once after camera thread is created.
        """
        # Timing mode
        self.timing_mode_combo.currentIndexChanged.connect(
            lambda idx: None if self._building else cam_thread.set_prioritize_exposure(idx == 1)
        )

        # FPS: spin is canonical (slider syncs to it via _fps_slider_to_spin)
        # Same _building guard pattern as gamma to handle slider-driven spin updates
        self.fps_spin.valueChanged.connect(
            lambda v: None if self._building else cam_thread.set_target_fps(float(v))
        )
        self.fps_slider.valueChanged.connect(
            lambda v: (
                None if self._building else
                cam_thread.set_target_fps(
                    round(self._fps_min + (v / 10000.0)
                          * (self._fps_max - self._fps_min), 2)
                )
            )
        )

        # Exposure: spin is the canonical control (slider syncs to it)
        self.exposure_spin.valueChanged.connect(
            lambda v: None if self._building else cam_thread.set_exposure_ms(float(v))
        )

        # Gain
        self.gain_spin.valueChanged.connect(
            lambda v: None if self._building else cam_thread.set_master_gain(int(v))
        )

        # Gain boost
        self.gain_boost_cb.toggled.connect(
            lambda v: None if self._building else cam_thread.set_gain_boost(bool(v))
        )

        # Gamma
        # NOTE: gamma_spin.valueChanged alone is insufficient here because when
        # the *slider* drives the spin, _gamma_slider_to_spin sets _building=True
        # around the spin update, which causes the lambda below to skip the camera
        # call.  We therefore also connect the slider directly.  The slider→camera
        # lambda converts the integer slider value (1–1000) back to the float gamma
        # using the same formula as _gamma_slider_to_spin.
        self.gamma_spin.valueChanged.connect(
            lambda v: None if self._building else cam_thread.set_gamma(float(v))
        )
        self.gamma_slider.valueChanged.connect(
            lambda v: (
                None if self._building else
                cam_thread.set_gamma(
                    round(self._gamma_min + ((v - 1) / 999.0)
                          * (self._gamma_max - self._gamma_min), 2)
                )
            )
        )

        # Pixel clock
        self.pclk_combo.currentIndexChanged.connect(
            lambda idx: self._on_pclk_changed(cam_thread)
        )

        # AOI apply
        self.aoi_apply_btn.clicked.connect(
            lambda: cam_thread.request_aoi_change(
                self.aoi_width_spin.value(),
                self.aoi_height_spin.value(),
                self.aoi_x_spin.value(),
                self.aoi_y_spin.value(),
            )
        )

        # Camera info signal → populate ranges
        cam_thread.camera_info_signal.connect(self.update_from_camera_info)

    def _on_pclk_changed(self, cam_thread) -> None:
        if self._building:
            return
        data = self.pclk_combo.currentData()
        if data is not None:
            cam_thread.set_pixel_clock(int(data))


class RasterMainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller, *, ui_path: Optional[str] = None, parent=None):
        super().__init__(parent)

        if ui_path is None:
            # default: same directory as this file
            ui_path = os.path.join(os.path.dirname(__file__), UI_FILE)

        uic.loadUi(ui_path, self)

        self.controller = controller

        # --- add step/continuous raster controls (no .ui edit required) ---
        self._install_raster_mode_controls()


        # --- UI state ---
        self._mode = "normal"   # normal | scale | calibrate
        self._scale_clicks: List[TargetXY] = []
        self._hull_points: List[TargetXY] = []
        self._update_ui_calibration_state(False)  # initial uncalibrated

        # last position history (for jogging points display)
        self._history: List[TargetXY] = []

        # Frametime metrics
        self._last_frame_time = time.perf_counter()
        self._fps_smoothed = None

        # Image scale (units per pixel in plot coordinates)
        self._img_scale: float = float(self.scaleImage.value()) if hasattr(self, "scaleImage") else 1.0
        # Flip settings: default from config.APP_CONFIG.camera.flip_x / flip_y when present
        self._flip_x: bool = bool(getattr(getattr(getattr(_config, 'APP_CONFIG', None), 'camera', None), 'flip_x', False)) if _config else False
        self._flip_y: bool = bool(getattr(getattr(getattr(_config, 'APP_CONFIG', None), 'camera', None), 'flip_y', False)) if _config else False
        self._last_frame_shape: Optional[Tuple[int, int]] = None  # (h, w)

        # Display rotation: k for np.rot90 (0=none, -1=90CW, 2=180, 1=90CCW)
        self._rotation_k: int = -1  # default: 90° CW (matches original hardcoded value)

        # --- Build plot display into placeholder widget "plot" ---
        self._init_plot()

        # --- Wire UI -> controller ---
        self._connect_ui_actions()

        # --- Wire controller -> UI ---
        self._connect_controller_signals()
        
        # --- Install Camera Settings dock ---
        self._install_camera_settings_dock()
        
        # Camera setup
        self._start_camera()

        self._log(f"Display: rotation={self._rotation_k}, flip_x={self._flip_x}, flip_y={self._flip_y}")

    # -------------------------
    # Plot setup + overlays
    # -------------------------

    def _init_plot(self) -> None:
        # Insert PlotWidget into the designer placeholder widget named "plot"
        placeholder = getattr(self, "plot", None)
        if placeholder is None:
            raise RuntimeError("UI is missing QWidget named 'plot'")

        layout = placeholder.layout()
        if layout is None:
            layout = QtWidgets.QVBoxLayout(placeholder)
            layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        # Create a floating FPS label on top of the plot
        self.fps_label = QtWidgets.QLabel(self.plot_widget)
        self.fps_label.setStyleSheet("color: #00FF00; font-weight: bold; font-size: 14px;")
        self.fps_label.setText("FPS: 0.0")
        self.fps_label.move(30, 10) # (30,10) px from top-left of the plot
        self.fps_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents) # Let clicks pass through
        self.fps_label.show()
        # ----------------------

        self.vb = self.plot_widget.getViewBox()
        self.plot_widget.setAspectLocked(True)
        # Flip orientation (fast) using ViewBox
        # NOTE: invertX/invertY invert axis direction; this is exactly what you want for flips.
        self.vb.invertX(self._flip_x)
        self.vb.invertY(self._flip_y)

        # Image item (fast path for numpy arrays)
        # Explicitly set axisOrder to row-major, ensures (H, W) maps to (x, y)
        self.img_item = pg.ImageItem(axisOrder='row-major')
        self.plot_widget.addItem(self.img_item)

        # Overlays: hull points, raster path, manual preview
        self.hull_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#c402cf"))
        self.raster_scatter = pg.ScatterPlotItem(size=5, brush=pg.mkBrush("#2b7cff"))
        self.manual_scatter = pg.ScatterPlotItem(size=7, brush=pg.mkBrush("#ff8c00"))
        self.current_target_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ff0000"))

        self.plot_widget.addItem(self.hull_scatter)
        self.plot_widget.addItem(self.raster_scatter)
        self.plot_widget.addItem(self.manual_scatter)
        self.plot_widget.addItem(self.current_target_marker)

        # Direction lines (optional)
        self._dir_items: List[pg.PlotDataItem] = []

        # Bounds rectangle
        self._bounds_item = None

        # Mouse click
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_click)

        # Crosshair + mouse position readout (pixel_x_pos / pixel_y_pos)
        self._vline = pg.InfiniteLine(angle=90, movable=False)
        self._hline = pg.InfiniteLine(angle=0, movable=False)
        self.plot_widget.addItem(self._vline, ignoreBounds=True)
        self.plot_widget.addItem(self._hline, ignoreBounds=True)

        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)

    def set_frame(self, frame: np.ndarray) -> None:
        """
        Called by camera thread (should be invoked via Qt signal to stay in UI thread).
        Expects a 2D grayscale or 3D RGB ndarray.
        """
        if frame is None:
            return
        
        # Rotate Image (configurable via Camera Settings dock)
        if self._rotation_k != 0:
            frame = np.rot90(frame, k=self._rotation_k)
        # --------------------

        # Update FPS display
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        
        if dt > 0:
            current_fps = 1.0 / dt
            # Simple smoothing (90% history, 10% new) to stop jitter
            if self._fps_smoothed is None:
                self._fps_smoothed = current_fps
            else:
                self._fps_smoothed = (0.9 * self._fps_smoothed) + (0.1 * current_fps)
            self.fps_label.setText(f"FPS: {self._fps_smoothed:.1f}")
            self.fps_label.adjustSize() # Ensure text fits if it gets wider
        # ----------------------

        # Update image
        self.img_item.setImage(frame, autoLevels=False)

        # Track shape and apply scaling so the plot axes represent "scaled units"
        try:
            h, w = int(frame.shape[0]), int(frame.shape[1])
        except Exception:
            return

        if self._last_frame_shape != (h, w):
            self._last_frame_shape = (h, w)
            self._apply_image_scale()   # applies dist-per-pixel to ImageItem rect/transform
        
    def closeEvent(self, event):
        try:
            if hasattr(self, "camera_thread"):
                self.camera_thread.stop()
                self.camera_thread.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)


    def _on_mouse_moved(self, pos) -> None:
        if self.vb.sceneBoundingRect().contains(pos):
            mouse_point = self.vb.mapSceneToView(pos)
            x = float(mouse_point.x())
            y = float(mouse_point.y())
            self._vline.setPos(x)
            self._hline.setPos(y)
            # show in UI labels
            if hasattr(self, "pixel_x_pos"):
                self.pixel_x_pos.setText(f"{x:.4f}")
            if hasattr(self, "pixel_y_pos"):
                self.pixel_y_pos.setText(f"{y:.4f}")

    def _on_plot_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        mouse_point = self.vb.mapSceneToView(event.scenePos())
        x = float(mouse_point.x())
        y = float(mouse_point.y())

        # Always update x/y fields if present
        if hasattr(self, "x"):
            self.x.setValue(x)
        if hasattr(self, "y"):
            self.y.setValue(y)

        # Mode handling (return early so hull points don't accumulate)
        if self._mode == "scale":
            self._scale_clicks.append((x, y))
            if len(self._scale_clicks) >= 2:
                self._finish_scale()
            return

        if self._mode == "calibrate":
            # Forward click to controller calibration collector
            self.controller.add_calibration_click(x, y)
            return

        # Normal mode: clicks add hull points (used by convex hull raster)
        self._hull_points.append((x, y))
        self.hull_scatter.setData([p[0] for p in self._hull_points], [p[1] for p in self._hull_points])

    def _install_camera_settings_dock(self) -> None:
        """Create and install the Camera Settings dock widget + View menu."""
        self.cam_dock = CameraSettingsDock(self)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.cam_dock)

        # Sync dock's flip/rotation to our display state
        self.cam_dock.flip_x_cb.setChecked(self._flip_x)
        self.cam_dock.flip_y_cb.setChecked(self._flip_y)

        # Set rotation combo to match default _rotation_k
        k_to_index = {0: 0, -1: 1, 2: 2, 1: 3}
        self.cam_dock.rotation_combo.setCurrentIndex(k_to_index.get(self._rotation_k, 0))

        # Connect display transform signals
        self.cam_dock.rotation_changed.connect(self._set_rotation)
        self.cam_dock.flip_x_changed.connect(self._set_flip_x)
        self.cam_dock.flip_y_changed.connect(self._set_flip_y)

        # Connect save / load / revert buttons
        self.cam_dock.save_requested.connect(self._save_camera_settings)
        self.cam_dock.load_requested.connect(self._load_camera_settings)
        self.cam_dock.revert_requested.connect(self._revert_camera_settings)

        # Also provide legacy checkboxes for .ui files that include them
        if not hasattr(self, "flip_x_checkbox"):
            self.flip_x_checkbox = self.cam_dock.flip_x_cb
        if not hasattr(self, "flip_y_checkbox"):
            self.flip_y_checkbox = self.cam_dock.flip_y_cb

        # --- View menu: toggle dock visibility ---
        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("&View")
        # toggleViewAction() is a built-in QDockWidget method that creates
        # a checkable action to show/hide the dock
        toggle_action = self.cam_dock.toggleViewAction()
        toggle_action.setText("Camera Settings")
        toggle_action.setShortcut(QtGui.QKeySequence("Ctrl+Shift+C"))
        view_menu.addAction(toggle_action)

    def _set_rotation(self, k: int) -> None:
        self._rotation_k = k
        # Force shape recalculation on next frame
        self._last_frame_shape = None
        self._log(f"Rotation set to k={k}")

    def _set_flip_x(self, checked: bool) -> None:
        self._flip_x = bool(checked)
        if hasattr(self, "vb"):
            self.vb.invertX(self._flip_x)

    def _set_flip_y(self, checked: bool) -> None:
        self._flip_y = bool(checked)
        if hasattr(self, "vb"):
            self.vb.invertY(self._flip_y)

    def _save_camera_settings(self) -> None:
        """
        Save current camera + display settings to .ini file.
        Uses the configured camera_params_ini path, or prompts for a path.
        """
        # Determine default save path
        default_path = ""
        if _config is not None and hasattr(_config, "APP_CONFIG"):
            default_path = getattr(_config.APP_CONFIG.camera, "camera_params_ini", "") or ""

        if not default_path:
            default_path = "camera_params.ini"

        save_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Camera Settings", default_path,
            "INI Files (*.ini);;All Files (*)"
        )
        if not save_path:
            return  # user cancelled

        # Gather settings from dock
        settings = self.cam_dock.get_current_settings()

        try:
            from camera import save_settings_to_ini
            save_settings_to_ini(save_path, settings)
            self._loaded_ini_path = save_path
            if hasattr(self, "cam_dock"):
                self.cam_dock.set_loaded_config_label(save_path)
            self._log(f"Camera settings saved to {save_path}")
        except Exception as e:
            self._log(f"Failed to save camera settings: {e}")

    def _load_camera_settings(self) -> None:
        """Open file dialog to pick an INI file and apply it to the running camera."""
        default_dir = ""
        if hasattr(self, "_loaded_ini_path") and self._loaded_ini_path:
            default_dir = os.path.dirname(self._loaded_ini_path)

        ini_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Camera Config", default_dir,
            "INI Files (*.ini);;All Files (*)"
        )
        if not ini_path:
            return  # user cancelled

        self._apply_ini_to_running_camera(ini_path)

    def _revert_camera_settings(self) -> None:
        """Revert camera settings to the most recently loaded/saved INI."""
        if not hasattr(self, "_loaded_ini_path") or not self._loaded_ini_path:
            self._log("No config file to revert to.")
            return
        if not os.path.isfile(self._loaded_ini_path):
            self._log(f"Config file not found: {self._loaded_ini_path}")
            return

        self._apply_ini_to_running_camera(self._loaded_ini_path)
        self._log(f"Reverted to {self._loaded_ini_path}")

    def _apply_ini_to_running_camera(self, ini_path: str) -> None:
        """Parse an INI file and apply all settings to the running camera."""
        from camera import load_ueye_config_from_ini, _load_display_settings_from_ini

        if not hasattr(self, "camera_thread") or self.camera_thread is None:
            self._log("No camera thread running.")
            return

        try:
            cam = None
            if _config is not None and hasattr(_config, "APP_CONFIG"):
                cam = _config.APP_CONFIG.camera

            overrides = {}
            if cam is not None:
                overrides["camera_id"] = cam.camera_id
                overrides["use_freeze"] = cam.use_freeze
                overrides["emit_rgb"] = cam.emit_rgb

            cfg = load_ueye_config_from_ini(ini_path, **overrides)
        except Exception as e:
            self._log(f"Failed to parse config: {e}")
            return

        # Apply timing mode first (affects how pixel clock and exposure behave)
        self.camera_thread.set_prioritize_exposure(cfg.prioritize_exposure)
        if hasattr(self, "cam_dock"):
            self.cam_dock._building = True
            self.cam_dock.timing_mode_combo.setCurrentIndex(1 if cfg.prioritize_exposure else 0)
            self.cam_dock.fps_spin.setValue(cfg.target_fps)
            self.cam_dock._building = False

        # Apply hardware settings via camera thread slots
        self.camera_thread.set_pixel_clock(cfg.pixel_clock_mhz)
        self.camera_thread.set_target_fps(cfg.target_fps)
        self.camera_thread.request_aoi_change(
            cfg.width, cfg.height, cfg.roi_offset_x, cfg.roi_offset_y
        )
        self.camera_thread.set_master_gain(cfg.master_gain)
        self.camera_thread.set_gain_boost(cfg.enable_gain_boost)
        self.camera_thread.set_gamma(cfg.gamma)
        self.camera_thread.set_exposure_ms(cfg.exposure_ms)

        # Apply display settings (rotation, flips)
        try:
            disp = _load_display_settings_from_ini(ini_path)
            if "rotation_k" in disp:
                self._rotation_k = disp["rotation_k"]
            if "flip_x" in disp:
                self._flip_x = disp["flip_x"]
            if "flip_y" in disp:
                self._flip_y = disp["flip_y"]
            if hasattr(self, "cam_dock"):
                k_to_index = {0: 0, -1: 1, 2: 2, 1: 3}
                self.cam_dock._building = True
                self.cam_dock.rotation_combo.setCurrentIndex(
                    k_to_index.get(self._rotation_k, 0)
                )
                self.cam_dock.flip_x_cb.setChecked(self._flip_x)
                self.cam_dock.flip_y_cb.setChecked(self._flip_y)
                self.cam_dock._building = False
            if hasattr(self, "vb"):
                self.vb.invertX(self._flip_x)
                self.vb.invertY(self._flip_y)
        except Exception:
            pass

        # Refresh dock controls from camera (updated ranges/values)
        self.camera_thread.request_info_refresh()

        # Update tracking
        self._loaded_ini_path = ini_path
        if hasattr(self, "cam_dock"):
            self.cam_dock.set_loaded_config_label(ini_path)
        self._log(f"Loaded config from {ini_path}")
        self._last_frame_shape = None  # force display recalculation

    def _start_camera(self) -> None:
        from camera import UEyeCameraThread, UEyeConfig

        cfg = None  # will be set below
        self._loaded_ini_path = ""

        # Read camera settings from config.py if available
        if _config is not None and hasattr(_config, "APP_CONFIG"):
            cam = _config.APP_CONFIG.camera

            # --- Option A: load from uEye Cockpit .ini if configured ---
            ini_path = getattr(cam, "camera_params_ini", None)
            if ini_path and os.path.isfile(ini_path):
                try:
                    from camera import load_ueye_config_from_ini
                    cfg = load_ueye_config_from_ini(
                        ini_path,
                        camera_id=cam.camera_id,
                        use_freeze=cam.use_freeze,
                        emit_rgb=cam.emit_rgb,
                    )
                    self._loaded_ini_path = ini_path
                    self._log(f"Camera config loaded from .ini: {ini_path}")
                except Exception as e:
                    self._log(f"Failed to load camera .ini ({ini_path}): {e}. Falling back to config.py values.")
                    cfg = None

                # Load saved display settings (rotation, flips) if present
                try:
                    from camera import _load_display_settings_from_ini
                    disp = _load_display_settings_from_ini(ini_path)
                    if "rotation_k" in disp:
                        self._rotation_k = disp["rotation_k"]
                    if "flip_x" in disp:
                        self._flip_x = disp["flip_x"]
                    if "flip_y" in disp:
                        self._flip_y = disp["flip_y"]
                    # Sync dock and ViewBox to loaded values
                    if hasattr(self, "cam_dock"):
                        k_to_index = {0: 0, -1: 1, 2: 2, 1: 3}
                        self.cam_dock._building = True
                        self.cam_dock.rotation_combo.setCurrentIndex(k_to_index.get(self._rotation_k, 0))
                        self.cam_dock.flip_x_cb.setChecked(self._flip_x)
                        self.cam_dock.flip_y_cb.setChecked(self._flip_y)
                        self.cam_dock._building = False
                    if hasattr(self, "vb"):
                        self.vb.invertX(self._flip_x)
                        self.vb.invertY(self._flip_y)
                except Exception:
                    pass

            # --- Option B: manual config.py fields ---
            if cfg is None:
                cfg = UEyeConfig(
                    camera_id=cam.camera_id,
                    width=cam.width,
                    height=cam.height,
                    exposure_ms=cam.exposure_ms_default,
                    pixel_clock_mhz=cam.pixel_clock_mhz,
                    use_freeze=cam.use_freeze,
                    emit_rgb=cam.emit_rgb,
                    roi_offset_x=cam.roi_offset_x,
                    roi_offset_y=cam.roi_offset_y,
                    master_gain=cam.master_gain,
                    gamma=cam.gamma,
                    enable_gain_boost=cam.enable_gain_boost,
                    target_fps=cam.target_fps,
                )

            # flips are display-only (your UI transform uses these)
            self._flip_x = bool(cam.flip_x)
            self._flip_y = bool(cam.flip_y)
        else:
            cfg = UEyeConfig()  # fallback defaults

        self.camera_thread = UEyeCameraThread(cfg, parent=self)
        self.camera_thread.new_frame.connect(self.set_frame)
        self.camera_thread.status.connect(self._log)
        self.camera_thread.error.connect(self._log)

        # Wire Camera Settings dock to camera thread
        if hasattr(self, "cam_dock"):
            self.cam_dock.connect_to_camera_thread(self.camera_thread)
            self.cam_dock.set_loaded_config_label(self._loaded_ini_path)
            # Initialize timing mode and FPS from config
            self.cam_dock._building = True
            self.cam_dock.timing_mode_combo.setCurrentIndex(1 if cfg.prioritize_exposure else 0)
            self.cam_dock.fps_spin.setValue(cfg.target_fps)
            self.cam_dock._building = False

        self.camera_thread.start()

        # Optionally apply extra .ini polish (hotpixel correction, etc.)
        if _config is not None and hasattr(_config, "APP_CONFIG"):
            ini_path = getattr(_config.APP_CONFIG.camera, "camera_params_ini", None)
            if ini_path and os.path.isfile(ini_path):
                def _apply_extras():
                    try:
                        from camera import apply_ini_to_camera
                        if hasattr(self.camera_thread, '_cam') and self.camera_thread._cam is not None:
                            apply_ini_to_camera(self.camera_thread._cam, ini_path)
                            self.camera_thread.request_info_refresh()
                    except Exception:
                        pass
                QtCore.QTimer.singleShot(2000, _apply_extras)

        # Wire existing exposure spinbox -> camera thread
        self.exposurevalue.setValue(float(cfg.exposure_ms))
        self.exposurevalue.valueChanged.connect(lambda v: self.camera_thread.set_exposure_ms(float(v)))

        # Bidirectional sync: dock exposure <-> main exposure spinbox
        if hasattr(self, "cam_dock"):
            def _dock_to_main(v):
                self.exposurevalue.blockSignals(True)
                self.exposurevalue.setValue(v)
                self.exposurevalue.blockSignals(False)
            self.cam_dock.exposure_spin.valueChanged.connect(_dock_to_main)

            def _main_to_dock(v):
                if not self.cam_dock._building:
                    self.cam_dock._building = True
                    self.cam_dock.exposure_spin.setValue(v)
                    self.cam_dock._building = False
            self.exposurevalue.valueChanged.connect(_main_to_dock)


    def _apply_image_scale(self, scale: float | None = None) -> None:
        """
        Update plot-units-per-pixel scale and re-apply image mapping.
        """
        if scale is None:
            scale = float(self.scaleImage.value()) if hasattr(self, "scaleImage") else 1.0
        self._img_scale = float(scale)
        self._apply_image_mapping()

    def _apply_image_mapping(self) -> None:
        """
        Apply scale + optional flips to the displayed ImageItem *without* copying the frame buffer.
        Plot coordinates become "scaled units".
        """
        if self._last_frame_shape is None:
            return
        h, w = self._last_frame_shape

        s = float(self._img_scale) if self._img_scale else 1.0
        # Can interpret as mm/pixel if desired.

        # Define the image extents in plot coordinates
        self.img_item.setRect(QtCore.QRectF(0, 0, w * s, h * s))
        self.vb.setRange(xRange=(0, w * s), yRange=(0, h * s), padding=0.0)


    # -------------------------
    # Raster mode helpers (step vs continuous)
    # -------------------------

    def _install_raster_mode_controls(self) -> None:
        """
        Legacy support for raster_gui2.ui
        Adds a 'Continuous' checkbox + 'Step' button without editing the .ui file.
        These are UI-only controls:
        - Continuous checked  => controller runs automatically (continuous raster)
        - Continuous unchecked => controller is armed; user/ZMQ advances via Step/move_to_next
        """
        self._raster_active_ui = False

        # If UI file didn't provide them, create duplicates (fallback)
        if not hasattr(self, "raster_continuous_checkbox"):
            self.raster_continuous_checkbox = QtWidgets.QCheckBox("Continuous")
            self.raster_continuous_checkbox.setChecked(True)
            self.statusBar().addPermanentWidget(self.raster_continuous_checkbox)

        if not hasattr(self, "raster_step_button"):
            self.raster_step_button = QtWidgets.QPushButton("Step")
            self.raster_step_button.setEnabled(False)
            self.statusBar().addPermanentWidget(self.raster_step_button)

        # Set Tooltips
        self.raster_continuous_checkbox.setToolTip("Checked: run continuously.\nUnchecked: step mode.")
        self.raster_step_button.setToolTip("Advance one raster point.")

        # Wire signals
        # Note: We use try/disconnect to avoid double-wiring if this function runs twice
        try: self.raster_continuous_checkbox.stateChanged.disconnect()
        except: pass
        try: self.raster_step_button.clicked.disconnect()
        except: pass

        self.raster_continuous_checkbox.stateChanged.connect(self._update_step_mode_ui)
        self.raster_step_button.clicked.connect(self._step_raster)

        self._update_step_mode_ui()

    def _update_ui_calibration_state(self, calibrated: bool) -> None:
        units = "mm" if calibrated else "motor units"

        for name, text in [
            ("l_stepx", f"Step x ({units}):"),
            ("l_stepy", f"Step y ({units}):"),
            ("lx", f"x ({units}):"),
            ("ly", f"y ({units}):"),
        ]:
            if hasattr(self, name):
                getattr(self, name).setText(text)

        if hasattr(self, "group_move"):
            self.group_move.setTitle(f"Move / Preview ({units})")


    def _update_step_mode_ui(self) -> None:
        """
        Enable Step only when:
        - raster is active, AND
        - continuous is unchecked.

        Lock the mode checkbox while raster is active to avoid controller/UI drift.
        """
        active = bool(getattr(self, "_raster_active_ui", False))
        continuous = bool(self.raster_continuous_checkbox.isChecked())

        self.raster_step_button.setEnabled(active and (not continuous))
        self.raster_continuous_checkbox.setEnabled(not active)


    def _step_raster(self) -> None:
        """
        Advance exactly one raster point.
        If raster isn't armed yet, arm it in step mode first (continuous unchecked), then step.
        """
        # If user is in continuous mode, Step should be disabled anywayâ€”guard for safety.
        if self.raster_continuous_checkbox.isChecked():
            self._log("Step is disabled while Continuous is checked. Uncheck Continuous and press Start (arms step mode).")
            return

        # If raster isn't armed, arm it now (continuous=False => no motion yet).
        if not getattr(self, "_raster_active_ui", False):
            self._start_raster()
            if not getattr(self, "_raster_active_ui", False):
                self._log("Raster could not be armed for step mode.")
                return

        # Prevent double-click queueing until we get a completion callback.
        self.raster_step_button.setEnabled(False)

        # Fire one step (non-blocking UI; controller emits command_done_signal)
        self.controller.raster_step(source="ui", wait=False)


    def _on_command_done(self, cmd_id: str, ok: bool, message: str, tag: str) -> None:
        # Re-enable Step after a raster step completes (success or failure)
        if tag == "raster_step":
            self._update_step_mode_ui()




    # -------------------------
    # UI -> Controller wiring
    # -------------------------

    def _connect_ui_actions(self) -> None:
        # Buttons
        self.move_to_pos.clicked.connect(self._move_to_position)
        self.preview_pos.clicked.connect(self._preview_position)
        self.clearAllManual.clicked.connect(self._clear_manual_points)

        self.jog_up_button_3.clicked.connect(lambda: self._jog(0, +1))
        self.jog_down_button_3.clicked.connect(lambda: self._jog(0, -1))
        self.jog_left_button_3.clicked.connect(lambda: self._jog(-1, 0))
        self.jog_right_button_3.clicked.connect(lambda: self._jog(+1, 0))

        self.homeX_3.clicked.connect(lambda: self.controller.request_home("X", hard=False))
        self.homeX_4.clicked.connect(lambda: self.controller.request_home("X", hard=True))
        self.homeY_3.clicked.connect(lambda: self.controller.request_home("Y", hard=False))
        self.homeY_4.clicked.connect(lambda: self.controller.request_home("Y", hard=True))

        self.x_backlash.valueChanged.connect(lambda v: self.controller.request_set_backlash("X", float(v)))
        self.y_backlash.valueChanged.connect(lambda v: self.controller.request_set_backlash("Y", float(v)))

        self.start_button.clicked.connect(self._start_raster)
        # REMOVE this line (already connected in _install_step_mode_controls)
        # self.raster_step_button.clicked.connect(self._step_raster)
        self.stop_button.clicked.connect(self.controller.stop_raster)
        self.path_button.clicked.connect(self._preview_raster_path)
        self.clearAll.clicked.connect(self._clear_raster_points)
        self.save_button.clicked.connect(self._save_and_clear_raster)

        self.bound_button.clicked.connect(self._display_bounds)

        self.scaleButton.clicked.connect(self._enter_scale_mode)
        self.scaleImage.valueChanged.connect(self._apply_image_scale)
        self.calibrateButton.clicked.connect(self._enter_calibration_mode)
        self.useold.clicked.connect(self.controller.load_calibration)
        self.resetButton.clicked.connect(self._reset_calibration_display)


    def _jog(self, sx: int, sy: int) -> None:
        dx = float(self.dx_button.value()) * float(sx)
        dy = float(self.dy_button.value()) * float(sy)
        self.controller.request_jog_target(dx, dy, source="ui")

    def _move_to_position(self) -> None:
        x = float(self.x.value())
        y = float(self.y.value())
        self.controller.request_move_target(x, y, source="ui")

    def _preview_position(self) -> None:
        x = float(self.x.value())
        y = float(self.y.value())
        # purely visual
        self.manual_scatter.addPoints([x], [y])

    def _clear_manual_points(self) -> None:
        self.manual_scatter.clear()
        self.current_target_marker.clear()
        self._history.clear()

    # -------------------------
    # Raster controls (preview + start)
    # -------------------------

    def _current_bounds(self) -> Tuple[float, float, float, float]:
        xmin = float(self.xlow.value())
        xmax = float(self.xhigh.value())
        ymin = float(self.ylow.value())
        ymax = float(self.yhigh.value())
        return xmin, xmax, ymin, ymax


    def _build_raster_spec(self) -> RasterSpec:
        """
        Read UI controls and build a RasterSpec.

        Spiral origin rule (per your request): center of the current bounds.
        """
        bounds = self._current_bounds()
        xmin, xmax, ymin, ymax = bounds
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)

        # Common steps
        xstep = float(self.xstep.value()) if hasattr(self, "xstep") else float(self.dx_button.value())
        ystep = float(self.ystep.value()) if hasattr(self, "ystep") else float(self.dy_button.value())

        # Map algorithm choice text to spec kind
        alg_text = self.alg_choice.currentText().lower().strip() if hasattr(self, "alg_choice") else "square raster x"
        if "square" in alg_text and "y" in alg_text:
            kind = "square_y"
        elif "square" in alg_text and "x" in alg_text:
            kind = "square_x"
        elif "spiral" in alg_text:
            kind = "spiral"
        elif "hull" in alg_text or "convex" in alg_text:
            kind = "hull"
        else:
            # fallback: try index ordering used in older UIs
            kind = "square_x"

        if kind in ("square_x", "square_y"):
            return RasterSpec(kind=kind, bounds=bounds, xstep=xstep, ystep=ystep)

        if kind == "spiral":
            radius = float(self.radius_spiral.value())
            step = float(self.step_spiral.value())
            angle_step = float(self.angle_spiral.value())
            angle_step_change = float(self.ang_change.value())
            return RasterSpec(
                kind="spiral",
                bounds=bounds,
                origin=(cx, cy),
                radius=radius,
                step=step,
                angle_step=angle_step,
                angle_step_change=angle_step_change,
            )

        # hull raster
        hull_pts = list(self._hull_points)
        return RasterSpec(
            kind="hull",
            bounds=bounds,
            xstep=xstep,
            ystep=ystep,
            hull_points=hull_pts,
            hull_order="xy",
        )


    def _display_bounds(self) -> None:
        xmin, xmax, ymin, ymax = self._current_bounds()
        # Remove old bounds if present
        if self._bounds_item is not None:
            self.plot_widget.removeItem(self._bounds_item)
            self._bounds_item = None

        rect = QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
        # Draw bounds as a QGraphicsRectItem using pg's ROI/GraphicsObject pattern
        pen = pg.mkPen("#cc6600")
        brush = pg.mkBrush("#ebce191a")
        self._bounds_item = QtWidgets.QGraphicsRectItem(rect)
        self._bounds_item.setPen(pen)
        self._bounds_item.setBrush(brush)
        self.plot_widget.addItem(self._bounds_item)

        # Also inform controller of soft bounds if you want:
        # (Weâ€™ll add controller.set_target_bounds() soon)
        # self.controller.set_target_bounds((xmin, xmax, ymin, ymax))

    def _preview_raster_path(self) -> None:
        # Clear old preview
        self._clear_raster_points()

        try:
            spec = self._build_raster_spec()
        except Exception as e:
            self._log(f"Preview Path error: {e}")
            return

        # Hull requires points
        if spec.kind == "hull" and (not spec.hull_points or len(spec.hull_points) < 3):
            self._log("Convex Hull raster requires at least 3 hull points (click to add points).")
            return

        # Build iterator (spiral center is already set to bounds center in _build_raster_spec)
        try:
            it = iter_path_from_spec(spec)
        except Exception as e:
            self._log(f"Preview Path error: {e}")
            return

        # Materialize points for plotting (cap to avoid UI overload)
        pts = collect_points(it, max_points=50000)
        if not pts:
            self._log("Preview Path: no points generated.")
            return

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self.raster_scatter.setData(xs, ys)
        self._log(f"Preview Path: {len(pts)} points.")

        # Optional direction lines
        if hasattr(self, "show_direction_checkbox") and self.show_direction_checkbox.isChecked():
            xline = []
            yline = []
            for i in range(len(pts) - 1):
                x1, y1 = pts[i]
                x2, y2 = pts[i + 1]
                xline.extend([x1, x2, float("nan")])
                yline.extend([y1, y2, float("nan")])
            item = pg.PlotDataItem(xline, yline, pen=pg.mkPen("#aaaaaa", width=1))
            self.plot_widget.addItem(item)
            self._dir_items.append(item)

    def _clear_raster_points(self) -> None:
        # Clear raster preview points/lines
        self.raster_scatter.clear()
        for item in self._dir_items:
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self._dir_items.clear()

        # IMPORTANT: Reset convex hull state (legacy Clear All behavior)
        self._hull_points.clear()
        self.hull_scatter.clear()


    def _save_and_clear_raster(self) -> None:
        # Controller already writes raster logs on stop/finish; this button historically did "Save and Clear"
        self.controller.stop_raster()
        self._clear_raster_points()

    def _start_raster(self) -> None:
        try:
            spec = self._build_raster_spec()
        except Exception as e:
            self._log(f"Start Raster error: {e}")
            return

        if spec.kind == "hull" and (not spec.hull_points or len(spec.hull_points) < 3):
            self._log("Convex Hull raster requires at least 3 hull points (click to add points).")
            return

        # Create a fresh iterator (preview may have consumed one)
        try:
            it = iter_path_from_spec(spec)
        except Exception as e:
            self._log(f"Start Raster error: {e}")
            return

        # Log dir from config if available
        log_dir = None
        try:
            if _config is not None:
                log_dir = getattr(getattr(getattr(_config, "APP_CONFIG", None), "paths", None), "raster_log_dir", None)
        except Exception:
            log_dir = None

        continuous = bool(self.raster_continuous_checkbox.isChecked())
        delay_s = 0.0
        if hasattr(self, "sleepTimer"):
            delay_s = float(self.sleepTimer.value())

        self.controller.start_raster(it, continuous=continuous, log_dir=log_dir, delay_s=(delay_s if continuous else 0.0))

        self._update_step_mode_ui()
        self._log(f"Raster started: {spec.kind}")

    # -------------------------
    # Scale + calibration modes
    # -------------------------

    def _enter_scale_mode(self) -> None:
        self._mode = "scale"
        self._scale_clicks.clear()
        self._log("Scale mode: set Scale Image value to known distance, then click two points.")

    def _finish_scale(self) -> None:
        (x1, y1), (x2, y2) = self._scale_clicks[:2]
        dist = float(np.hypot(x2 - x1, y2 - y1))
        if dist <= 0:
            self._log("Scale failed: zero distance.")
            self._mode = "normal"
            return
        # Preserve old code behavior: scale := scale / dist
        scale = float(self.scaleImage.value()) / dist
        self.scaleImage.setValue(scale)
        self._apply_image_scale(scale)
        self._log(f"Scale updated to {scale:.6g} per unit.")
        self._mode = "normal"

    def _enter_calibration_mode(self) -> None:
        self._mode = "calibrate"
        self.controller.start_calibration(required_points=3)

    def _reset_calibration_display(self) -> None:
        # UI-only reset + clear controller calibration
        self.controller.clear_calibration()
        # Reset displayed matrix/offset fields if present
        for nm, val in [("matrix_11", 1.0), ("matrix_12", 0.0), ("matrix_21", 0.0), ("matrix_22", 1.0),
                        ("offset_a", 0.0), ("offset_b", 0.0)]:
            if hasattr(self, nm):
                getattr(self, nm).setValue(val)
        self._update_ui_calibration_state(False)
        self._log("Calibration reset.")

    # -------------------------
    # Controller -> UI wiring
    # -------------------------

    def _connect_controller_signals(self) -> None:
        c = self.controller

        c.status_signal.connect(self._log)
        c.error_signal.connect(self._log)

        c.target_position_signal.connect(self._on_target_position)
        c.motor_position_signal.connect(self._on_motor_position)

        c.calibration_prompt_signal.connect(self._log)
        c.calibration_progress_signal.connect(self._on_calibration_progress)
        c.calibration_ready_signal.connect(self._on_calibration_ready)
        c.calibration_failed_signal.connect(self._on_calibration_failed)

        c.raster_state_signal.connect(self._on_raster_state)
        c.raster_finished_signal.connect(lambda: self._log("Raster finished."))
        c.raster_log_path_signal.connect(lambda p: self._log(f"Raster log: {p}"))

        c.command_done_signal.connect(self._on_command_done)


    def _on_target_position(self, x: float, y: float) -> None:
        # Update current marker + history
        self.current_target_marker.setData([x], [y])

        if self.checkBox_2.isChecked():  # Save position history
            self._history.append((float(x), float(y)))

        # Display subset of history
        if self.show_all_points_checkbox.isChecked():
            pts = self._history
        else:
            n = int(self.point_display_count.value())
            pts = self._history[-n:] if n > 0 else []

        if pts:
            self.manual_scatter.setData([p[0] for p in pts], [p[1] for p in pts])
        else:
            self.manual_scatter.clear()

    def _on_motor_position(self, mx: float, my: float) -> None:
        if hasattr(self, "motor_x_pos"):
            self.motor_x_pos.setText(f"{mx:.5f}")
        if hasattr(self, "motor_y_pos"):
            self.motor_y_pos.setText(f"{my:.5f}")

        if hasattr(self, "progress_motor_x_pos"):
            self.progress_motor_x_pos.setValue(self._motor_to_percent(mx, "X"))
        if hasattr(self, "progress_motor_y_pos"):
            self.progress_motor_y_pos.setValue(self._motor_to_percent(my, "Y"))

    def _motor_to_percent(self, v: float, axis: str) -> int:
        """
        Convert motor position to a 0â€“100 progress bar value.
        Uses motor bounds from config if available, otherwise defaults to 0..12.
        """
        # Default range
        vmin, vmax = 0.0, 12.0

        # If you later add APP_CONFIG.hardware.motor_bounds = (xmin, xmax, ymin, ymax)
        try:
            if _config is not None and hasattr(_config, "APP_CONFIG"):
                mb = getattr(getattr(_config.APP_CONFIG, "hardware", None), "motor_bounds", None)
                if mb and len(mb) == 4:
                    xmin, xmax, ymin, ymax = map(float, mb)
                    if axis.upper() == "X":
                        vmin, vmax = xmin, xmax
                    else:
                        vmin, vmax = ymin, ymax
        except Exception:
            pass

        if vmax <= vmin:
            vmax = vmin + 1.0

        frac = (float(v) - vmin) / (vmax - vmin)
        frac = max(0.0, min(1.0, frac))
        return int(round(100.0 * frac))


    def _on_calibration_progress(self, collected: int, required: int) -> None:
        self._log(f"Calibration: {collected}/{required} points recorded.")
        if collected >= required:
            # exit mode; controller will emit ready/failed next
            self._mode = "normal"

    def _on_calibration_ready(self, cal) -> None:
        # cal is AffineCalibration
        try:
            M = cal.M
            b = cal.b
            if hasattr(self, "matrix_11"): self.matrix_11.setValue(float(M[0, 0]))
            if hasattr(self, "matrix_12"): self.matrix_12.setValue(float(M[0, 1]))
            if hasattr(self, "matrix_21"): self.matrix_21.setValue(float(M[1, 0]))
            if hasattr(self, "matrix_22"): self.matrix_22.setValue(float(M[1, 1]))
            if hasattr(self, "offset_a"): self.offset_a.setValue(float(b[0]))
            if hasattr(self, "offset_b"): self.offset_b.setValue(float(b[1]))
        except Exception:
            pass
        self._update_ui_calibration_state(True)
        self._log("Calibration ready.")
        self._mode = "normal"

    def _on_calibration_failed(self, msg: str) -> None:
        self._log(msg)
        self._mode = "normal"

    def _on_raster_state(self, active: bool) -> None:
        self._raster_active_ui = bool(active)
        self._log("Raster active." if active else "Raster inactive.")
        self._update_step_mode_ui()
        
        # lock in the mode choice while active
        if hasattr(self, "raster_continuous_checkbox"):
            self.raster_continuous_checkbox.setEnabled(not active)

        if hasattr(self, "raster_step_button"):
            is_step_mode = hasattr(self, "raster_continuous_checkbox") and (not self.raster_continuous_checkbox.isChecked())
            self.raster_step_button.setEnabled(active and is_step_mode)

        # Basic Start/Stop button UX (doesn't change controller behavior)
        try:
            self.start_button.setEnabled(not active)
            self.stop_button.setEnabled(active)
        except Exception:
            pass


    # -------------------------
    # Logging helper
    # -------------------------

    def _log(self, msg: str) -> None:
        msg = str(msg)
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        if hasattr(self, "textEdit_2"):
            self.textEdit_2.append(line)
        else:
            print(line)
