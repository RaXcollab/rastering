"""
camera_settings_dock.py

Dockable camera settings panel extracted from ui.py.
Controls live camera parameters (pixel clock, exposure, gain, gamma, AOI)
and display transforms (rotation, flip).
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtCore, QtWidgets


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

        # Signal suppression during programmatic updates uses widget.blockSignals()

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
        self.fps_slider.setEnabled(fps_mode)
        self.fps_spin.setEnabled(fps_mode)
        self.exposure_slider.setEnabled(not fps_mode)
        self.exposure_spin.setEnabled(not fps_mode)

    # --- FPS slider <-> spin sync (maps 0-10000 -> fps_min-fps_max) ---

    def _fps_slider_to_spin(self, slider_val: int) -> None:
        frac = slider_val / 10000.0
        fps = self._fps_min + frac * (self._fps_max - self._fps_min)
        self.fps_spin.blockSignals(True)
        self.fps_spin.setValue(round(fps, 2))
        self.fps_spin.blockSignals(False)

    def _fps_spin_to_slider(self, fps: float) -> None:
        rng = self._fps_max - self._fps_min
        if rng > 0:
            frac = (fps - self._fps_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self.fps_slider.blockSignals(True)
        self.fps_slider.setValue(int(round(frac * 10000)))
        self.fps_slider.blockSignals(False)

    # --- Exposure slider <-> spin sync (maps 0-10000 -> exp_min-exp_max) ---

    def _exp_slider_to_spin(self, slider_val: int) -> None:
        frac = slider_val / 10000.0
        ms = self._exp_min + frac * (self._exp_max - self._exp_min)
        self.exposure_spin.blockSignals(True)
        self.exposure_spin.setValue(ms)
        self.exposure_spin.blockSignals(False)

    def _exp_spin_to_slider(self, ms: float) -> None:
        rng = self._exp_max - self._exp_min
        if rng > 0:
            frac = (ms - self._exp_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self.exposure_slider.blockSignals(True)
        self.exposure_slider.setValue(int(round(frac * 10000)))
        self.exposure_slider.blockSignals(False)

    # --- Gamma slider <-> spin sync (slider 1-1000 -> 0.01-10.00) ---

    def _gamma_slider_to_spin(self, slider_val: int) -> None:
        frac = (slider_val - 1) / 999.0
        gamma = self._gamma_min + frac * (self._gamma_max - self._gamma_min)
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(round(gamma, 2))
        self.gamma_spin.blockSignals(False)

    def _gamma_spin_to_slider(self, gamma: float) -> None:
        rng = self._gamma_max - self._gamma_min
        if rng > 0:
            frac = (gamma - self._gamma_min) / rng
        else:
            frac = 0.0
        frac = max(0.0, min(1.0, frac))
        self.gamma_slider.blockSignals(True)
        self.gamma_slider.setValue(int(round(1 + frac * 999)))
        self.gamma_slider.blockSignals(False)

    # --- AOI X slider <-> spin sync (aligned to step of 4) ---

    def _aoi_x_slider_to_spin(self, v: int) -> None:
        v = (v // 4) * 4
        self.aoi_x_spin.blockSignals(True)
        self.aoi_x_spin.setValue(v)
        self.aoi_x_spin.blockSignals(False)

    def _aoi_x_spin_to_slider(self, v: int) -> None:
        self.aoi_x_slider.blockSignals(True)
        self.aoi_x_slider.setValue(v)
        self.aoi_x_slider.blockSignals(False)

    # --- AOI Y slider <-> spin sync ---

    def _aoi_y_slider_to_spin(self, v: int) -> None:
        v = (v // 4) * 4
        self.aoi_y_spin.blockSignals(True)
        self.aoi_y_spin.setValue(v)
        self.aoi_y_spin.blockSignals(False)

    def _aoi_y_spin_to_slider(self, v: int) -> None:
        self.aoi_y_slider.blockSignals(True)
        self.aoi_y_slider.setValue(v)
        self.aoi_y_slider.blockSignals(False)

    def _center_aoi(self) -> None:
        w = self.aoi_width_spin.value()
        h = self.aoi_height_spin.value()
        cx = max(0, (self._sensor_w - w) // 2)
        cy = max(0, (self._sensor_h - h) // 2)
        cx = (cx // 4) * 4
        cy = (cy // 4) * 4
        self.aoi_x_spin.setValue(cx)
        self.aoi_y_spin.setValue(cy)

    def _emit_rotation(self, index: int) -> None:
        k_map = {0: 0, 1: -1, 2: 2, 3: 1}
        self.rotation_changed.emit(k_map.get(index, 0))

    # --- Populate from camera_info dict ---

    def update_from_camera_info(self, info: dict) -> None:
        """
        Called when camera thread emits camera_info_signal.
        Updates all control ranges and current values without firing
        change signals back to the camera.
        """
        widgets = [
            self.pclk_combo, self.fps_spin, self.fps_slider,
            self.exposure_spin, self.exposure_slider,
            self.gain_spin, self.gain_slider, self.gain_boost_cb,
            self.gamma_spin, self.gamma_slider,
            self.aoi_width_spin, self.aoi_height_spin,
            self.aoi_x_spin, self.aoi_y_spin,
            self.aoi_x_slider, self.aoi_y_slider,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
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

            clocks = info.get("pixel_clocks", [])
            cur_pclk = info.get("pixel_clock", 0)
            self.pclk_combo.clear()
            for c in clocks:
                self.pclk_combo.addItem(f"{c} MHz", c)
            for i in range(self.pclk_combo.count()):
                if self.pclk_combo.itemData(i) == cur_pclk:
                    self.pclk_combo.setCurrentIndex(i)
                    break

            fps_min = info.get("fps_min", 0.1)
            fps_max = info.get("fps_max", 200.0)
            fps_cur = info.get("fps", 0.0)
            self._fps_min = fps_min
            self._fps_max = fps_max
            self.fps_range_label.setText(f"FPS range: {fps_min:.1f} – {fps_max:.1f}")
            self.fps_spin.setRange(fps_min, fps_max)
            self.fps_spin.setValue(fps_cur)
            fps_rng = fps_max - fps_min
            if fps_rng > 0:
                fps_frac = (fps_cur - fps_min) / fps_rng
            else:
                fps_frac = 0.0
            self.fps_slider.setValue(int(round(max(0, min(1, fps_frac)) * 10000)))

            self._exp_min = info.get("exposure_min", 0.01)
            self._exp_max = info.get("exposure_max", 1000.0)
            exp_inc = info.get("exposure_inc", 0.01)
            exp_cur = info.get("exposure", 30.0)
            self.exposure_spin.setRange(self._exp_min, self._exp_max)
            self.exposure_spin.setSingleStep(max(exp_inc, 0.001))
            self.exposure_spin.setValue(exp_cur)
            rng = self._exp_max - self._exp_min
            if rng > 0:
                frac = (exp_cur - self._exp_min) / rng
            else:
                frac = 0.0
            self.exposure_slider.setValue(int(round(max(0, min(1, frac)) * 10000)))
            self.exp_range_label.setText(
                f"Range: {self._exp_min:.3f} – {self._exp_max:.3f} ms"
            )

            self.gain_spin.setValue(info.get("gain", 0))
            self.gain_slider.setValue(info.get("gain", 0))
            self.gain_boost_cb.setChecked(info.get("gain_boost", False))

            self._gamma_min = info.get("gamma_min", 0.01)
            self._gamma_max = info.get("gamma_max", 2.2)
            self.gamma_spin.setRange(self._gamma_min, self._gamma_max)
            gamma_cur = info.get("gamma", 1.0)
            self.gamma_spin.setValue(gamma_cur)
            grng = self._gamma_max - self._gamma_min
            if grng > 0:
                gfrac = (gamma_cur - self._gamma_min) / grng
            else:
                gfrac = 0.0
            self.gamma_slider.setValue(int(round(1 + max(0, min(1, gfrac)) * 999)))

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
            for w in widgets:
                w.blockSignals(False)

    def get_current_settings(self) -> dict:
        """Read all current dock values into a dict for saving."""
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
        Wire dock controls -> camera thread parameter slots.
        Call once after camera thread is created.
        """
        # Timing mode
        self.timing_mode_combo.currentIndexChanged.connect(
            lambda idx: cam_thread.set_prioritize_exposure(idx == 1)
        )

        # FPS: spin is canonical (slider syncs to it via _fps_slider_to_spin).
        # Slider also needs a direct camera connection because _fps_slider_to_spin
        # blocks spin signals, preventing the spin's lambda from firing.
        self.fps_spin.valueChanged.connect(
            lambda v: cam_thread.set_target_fps(float(v))
        )
        self.fps_slider.valueChanged.connect(
            lambda v: cam_thread.set_target_fps(
                round(self._fps_min + (v / 10000.0)
                      * (self._fps_max - self._fps_min), 2)
            )
        )

        # Exposure: spin is the canonical control (slider syncs to it)
        self.exposure_spin.valueChanged.connect(
            lambda v: cam_thread.set_exposure_ms(float(v))
        )

        # Gain
        self.gain_spin.valueChanged.connect(
            lambda v: cam_thread.set_master_gain(int(v))
        )

        # Gain boost
        self.gain_boost_cb.toggled.connect(
            lambda v: cam_thread.set_gain_boost(bool(v))
        )

        # Gamma: spin is canonical. Slider drives spin via _gamma_slider_to_spin
        # (which blocks spin signals), so we connect the spin to the camera thread.
        # The slider also gets a direct connection for when the user drags it
        # (slider signals are not blocked in that case).
        self.gamma_spin.valueChanged.connect(
            lambda v: cam_thread.set_gamma(float(v))
        )
        self.gamma_slider.valueChanged.connect(
            lambda v: cam_thread.set_gamma(
                round(self._gamma_min + ((v - 1) / 999.0)
                      * (self._gamma_max - self._gamma_min), 2)
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

        # Camera info signal -> populate ranges
        cam_thread.camera_info_signal.connect(self.update_from_camera_info)

    def _on_pclk_changed(self, cam_thread) -> None:
        data = self.pclk_combo.currentData()
        if data is not None:
            cam_thread.set_pixel_clock(int(data))
