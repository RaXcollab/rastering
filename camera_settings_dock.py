"""
camera_settings_dock.py

Dockable Spinnaker / Blackfly camera settings panel. Replaces the uEye-era
controls (pixel clock combo, FPS-vs-Exposure timing-mode combo, gain-boost
checkbox, int 0-100 gain spin) with Spinnaker-native equivalents:

  Timing     -- Acq Frame Rate (+ enable), Exposure (ms)
  Pixel Fmt  -- combo populated from camera_info.pixel_formats
  Analog     -- Gain (float dB), Gamma (+ enable)
  GigE       -- Packet Size, Throughput Limit (0 = auto)
  Blackfly   -- Black Level (+ clamping), Defect Correction   [decision 8]
  AOI        -- Width / Height / Start X / Start Y (unchanged)
  Display    -- Rotation, Flip X / Flip Y (display-only, unchanged)

The legacy attribute names ``fps_spin`` and ``fps_slider`` are PRESERVED
(referenced unconditionally by ui.py); the user-visible labels are
relabelled to "Acq Frame Rate". The slot-resolution string remains
``set_target_fps`` per AUDIT:S3 -- CameraThread.set_target_fps still
forwards to acq_frame_rate. All slider+spin pairs route through the
shared ``_bind_param_controls`` helper so the slider-display sync and
the camera commit are wired together (structural fix for the
exposure-slider-omission class of bug, commits 56d1733 / 67d44dd).
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5 import QtCore, QtWidgets


class CameraSettingsDock(QtWidgets.QDockWidget):

    # Emitted when the user changes rotation / flip (display-only transforms).
    rotation_changed = QtCore.pyqtSignal(int)    # k value for np.rot90
    flip_x_changed = QtCore.pyqtSignal(bool)
    flip_y_changed = QtCore.pyqtSignal(bool)

    # Emitted when the user clicks the Save / Load / Revert buttons.
    save_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()
    revert_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Camera Settings", parent)
        self.setObjectName("CameraSettingsDock")
        self.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # --- Group: Timing (Acquisition Frame Rate + Exposure) ---------------
        grp_timing = QtWidgets.QGroupBox("Timing")
        fl_timing = QtWidgets.QFormLayout(grp_timing)

        # Acq Frame Rate enable (Parity 5: the Blackfly-native equivalent of
        # uEye's "prioritize exposure" timing mode -- when unchecked, exposure
        # can exceed the 1/fps frame period, useful for long-exposure imaging).
        self.acq_fps_enable_cb = QtWidgets.QCheckBox("Acq Frame Rate Enable")
        self.acq_fps_enable_cb.setChecked(True)
        self.acq_fps_enable_cb.setToolTip(
            "When unchecked, exposure can exceed the 1/fps frame period.\n"
            "Spinnaker AcquisitionFrameRateEnable node."
        )
        fl_timing.addRow(self.acq_fps_enable_cb)

        self.fps_range_label = QtWidgets.QLabel("Acq FPS range: --")
        self.fps_range_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_timing.addRow(self.fps_range_label)

        # NOTE: attribute names ``fps_slider`` / ``fps_spin`` PRESERVED for
        # ui.py compat (AUDIT:B1 -- ui.py:415-420,558-563 reference these
        # unconditionally). Only the user-visible labels change to
        # "Acq Frame Rate".
        fps_row = QtWidgets.QHBoxLayout()
        self.fps_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fps_slider.setMinimum(0)
        self.fps_slider.setMaximum(10000)
        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setRange(0.1, 200.0)
        self.fps_spin.setValue(20.0)
        fps_row.addWidget(self.fps_slider, 3)
        fps_row.addWidget(self.fps_spin, 1)
        self.fps_label = QtWidgets.QLabel("Acq Frame Rate:")
        fl_timing.addRow(self.fps_label, fps_row)

        # Exposure (ms; Spinnaker accepts microseconds at the node layer,
        # camera.py converts).
        exp_row = QtWidgets.QHBoxLayout()
        self.exposure_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exposure_slider.setMinimum(0)
        self.exposure_slider.setMaximum(10000)
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

        # --- Group: Pixel Format ---------------------------------------------
        grp_pixfmt = QtWidgets.QGroupBox("Pixel Format")
        fl_pixfmt = QtWidgets.QFormLayout(grp_pixfmt)
        self.pixfmt_combo = QtWidgets.QComboBox()
        self.pixfmt_combo.setToolTip(
            "Spinnaker PixelFormat. Mono8 is the default; the rastering "
            "display path expects uint8 2-D frames -- non-Mono8 formats "
            "require dock-side range scaling (not yet implemented)."
        )
        fl_pixfmt.addRow("Format:", self.pixfmt_combo)
        layout.addWidget(grp_pixfmt)

        # --- Group: Analog (Gain in dB + Gamma + Gamma Enable) ---------------
        grp_analog = QtWidgets.QGroupBox("Analog")
        fl_analog = QtWidgets.QFormLayout(grp_analog)

        # Gain: float dB (Parity 2 -- Blackfly is ~0..48 dB, NOT uEye int
        # 0..100). The slider integer endpoint is 0..1000; _bind_param_controls
        # maps to the live gain_db_min/max range from camera_info.
        gain_row = QtWidgets.QHBoxLayout()
        self.gain_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gain_slider.setRange(0, 1000)
        self.gain_spin = QtWidgets.QDoubleSpinBox()
        self.gain_spin.setDecimals(2)
        self.gain_spin.setSuffix(" dB")
        self.gain_spin.setRange(0.0, 48.0)  # placeholder; replaced by camera_info
        self.gain_spin.setSingleStep(0.1)
        gain_row.addWidget(self.gain_slider, 3)
        gain_row.addWidget(self.gain_spin, 1)
        fl_analog.addRow("Gain:", gain_row)

        self.gain_range_label = QtWidgets.QLabel("Range: --")
        self.gain_range_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_analog.addRow(self.gain_range_label)

        # Gamma + gamma-enable (Spinnaker: GammaEnable is a separate node;
        # writing Gamma when disabled is rejected on some Blackfly firmware,
        # so the enable checkbox gates the gamma slider's effect).
        self.gamma_enable_cb = QtWidgets.QCheckBox("Gamma Enable")
        self.gamma_enable_cb.setToolTip(
            "Spinnaker GammaEnable node. When unchecked, gamma value is "
            "ignored by the camera. Most users keep this off for raw imaging."
        )
        fl_analog.addRow(self.gamma_enable_cb)

        gamma_row = QtWidgets.QHBoxLayout()
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(1, 1000)  # maps to gamma_min..gamma_max
        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 10.0)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setSingleStep(0.05)
        gamma_row.addWidget(self.gamma_slider, 3)
        gamma_row.addWidget(self.gamma_spin, 1)
        fl_analog.addRow("Gamma:", gamma_row)

        layout.addWidget(grp_analog)

        # --- Group: GigE Transport -------------------------------------------
        # Replaces uEye Pixel Clock (Parity 4 -- GigE has no pixel clock;
        # bandwidth is controlled at the link layer via packet size +
        # device-link throughput limit).
        grp_gige = QtWidgets.QGroupBox("GigE Transport")
        fl_gige = QtWidgets.QFormLayout(grp_gige)

        self.packet_size_spin = QtWidgets.QSpinBox()
        self.packet_size_spin.setRange(1400, 9000)
        self.packet_size_spin.setSingleStep(100)
        self.packet_size_spin.setValue(9000)  # jumbo frames by default
        self.packet_size_spin.setSuffix(" B")
        self.packet_size_spin.setToolTip(
            "GevSCPSPacketSize. 9000 requires NIC jumbo frames enabled. "
            "1500 if jumbo is off; lower if the link drops packets."
        )
        fl_gige.addRow("Packet Size:", self.packet_size_spin)

        self.throughput_spin = QtWidgets.QSpinBox()
        self.throughput_spin.setRange(0, 1_000_000_000)
        self.throughput_spin.setSingleStep(1_000_000)
        self.throughput_spin.setValue(0)
        self.throughput_spin.setSuffix(" B/s")
        self.throughput_spin.setSpecialValueText("auto (camera default)")
        self.throughput_spin.setToolTip(
            "DeviceLinkThroughputLimit. 0 = auto (camera default). Cap "
            "for multi-camera or shared-link setups."
        )
        fl_gige.addRow("Throughput:", self.throughput_spin)

        layout.addWidget(grp_gige)

        # --- Group: Blackfly Native (decision 8, nice-to-have) ---------------
        # Each widget is auto-disabled at update_from_camera_info time if
        # the corresponding node is unavailable on this Blackfly model.
        grp_blkfly = QtWidgets.QGroupBox("Blackfly Native")
        fl_blkfly = QtWidgets.QFormLayout(grp_blkfly)

        self.black_level_spin = QtWidgets.QDoubleSpinBox()
        self.black_level_spin.setRange(-100.0, 100.0)
        self.black_level_spin.setDecimals(3)
        self.black_level_spin.setSingleStep(0.1)
        self.black_level_spin.setValue(0.0)
        self.black_level_spin.setToolTip(
            "Spinnaker BlackLevel node. Sensor-dependent; check the "
            "camera datasheet for the valid range."
        )
        fl_blkfly.addRow("Black Level:", self.black_level_spin)

        self.blacklevel_clamp_cb = QtWidgets.QCheckBox("Black Level Clamping")
        self.blacklevel_clamp_cb.setChecked(True)
        self.blacklevel_clamp_cb.setToolTip(
            "Spinnaker BlackLevelClampingEnable -- the Blackfly-native "
            "equivalent of uEye IS_AUTO_BLACKLEVEL_ON."
        )
        fl_blkfly.addRow(self.blacklevel_clamp_cb)

        self.defect_correction_cb = QtWidgets.QCheckBox("Defect Correction")
        self.defect_correction_cb.setToolTip(
            "Spinnaker DefectCorrectionEnable -- the Blackfly-native "
            "equivalent of uEye hotpixel-correction mode."
        )
        fl_blkfly.addRow(self.defect_correction_cb)

        layout.addWidget(grp_blkfly)

        # --- Group: AOI -------------------------------------------------------
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

        aoi_x_row = QtWidgets.QHBoxLayout()
        self.aoi_x_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aoi_x_slider.setRange(0, 4096)
        self.aoi_x_slider.setSingleStep(4)
        self.aoi_x_slider.setPageStep(40)
        self.aoi_x_spin = QtWidgets.QSpinBox()
        self.aoi_x_spin.setRange(0, 4096)
        self.aoi_x_spin.setSingleStep(4)
        self.aoi_x_spin.setToolTip("Spinnaker OffsetX -- absolute top-left.")
        aoi_x_row.addWidget(self.aoi_x_slider, 3)
        aoi_x_row.addWidget(self.aoi_x_spin, 1)
        fl_aoi.addRow("Start X:", aoi_x_row)

        aoi_y_row = QtWidgets.QHBoxLayout()
        self.aoi_y_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.aoi_y_slider.setRange(0, 4096)
        self.aoi_y_slider.setSingleStep(4)
        self.aoi_y_slider.setPageStep(40)
        self.aoi_y_spin = QtWidgets.QSpinBox()
        self.aoi_y_spin.setRange(0, 4096)
        self.aoi_y_spin.setSingleStep(4)
        self.aoi_y_spin.setToolTip("Spinnaker OffsetY -- absolute top-left.")
        aoi_y_row.addWidget(self.aoi_y_slider, 3)
        aoi_y_row.addWidget(self.aoi_y_spin, 1)
        fl_aoi.addRow("Start Y:", aoi_y_row)

        aoi_btn_row = QtWidgets.QHBoxLayout()
        self.aoi_apply_btn = QtWidgets.QPushButton("Apply AOI")
        self.aoi_apply_btn.setToolTip("Apply new AOI (camera briefly stops streaming)")
        self.aoi_center_btn = QtWidgets.QPushButton("Center")
        self.aoi_center_btn.setToolTip("Center the AOI on the sensor")
        aoi_btn_row.addWidget(self.aoi_apply_btn)
        aoi_btn_row.addWidget(self.aoi_center_btn)
        fl_aoi.addRow(aoi_btn_row)

        self.sensor_label = QtWidgets.QLabel("Sensor: --")
        self.sensor_label.setStyleSheet("color: gray; font-size: 11px;")
        fl_aoi.addRow(self.sensor_label)

        layout.addWidget(grp_aoi)

        # --- Group: Display transforms ---------------------------------------
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

        # --- Config label + Save / Load / Revert ------------------------------
        self.config_label = QtWidgets.QLabel("Loaded: (none)")
        self.config_label.setWordWrap(True)
        self.config_label.setStyleSheet("color: gray; font-size: 10px; padding: 2px;")
        layout.addWidget(self.config_label)

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

        # Internal physical-range state. Populated by update_from_camera_info;
        # _bind_param_controls reads via get_pmin/get_pmax closures.
        self._exp_min = 0.01
        self._exp_max = 1000.0
        self._fps_min = 0.1
        self._fps_max = 200.0
        self._gamma_min = 0.01
        self._gamma_max = 2.2
        self._gain_min = 0.0
        self._gain_max = 48.0

        # Sensor dimensions.
        self._sensor_w = 1280
        self._sensor_h = 1024

        # Camera-thread reference -- set lazily by connect_to_camera_thread;
        # _bind_param_controls resolves slot methods on it by string name.
        self._cam_thread = None

        # Wire all the internal-only signals (slider<->spin sync, button
        # clicks, etc). External signals are wired in connect_to_camera_thread.
        self._wire_signals()

    # ----- internal wiring ----------------------------------------------------

    def _wire_signals(self) -> None:
        # FPS / Exposure / Gamma / Gain: slider<->spin display sync AND the
        # camera commit are wired together by _bind_param_controls. The
        # slot-resolution string for FPS stays ``set_target_fps`` per
        # AUDIT:S3 (CameraThread forwards it to acq_frame_rate internally).
        # Gain uses ``set_gain_db`` (Parity 2 -- float dB, not int 0-100).
        self._bind_param_controls(
            self.fps_slider, self.fps_spin, "set_target_fps",
            smin=0, smax=10000,
            get_pmin=lambda: self._fps_min, get_pmax=lambda: self._fps_max,
            ndigits=2,
        )
        self._bind_param_controls(
            self.exposure_slider, self.exposure_spin, "set_exposure_ms",
            smin=0, smax=10000,
            get_pmin=lambda: self._exp_min, get_pmax=lambda: self._exp_max,
            ndigits=None,
        )
        self._bind_param_controls(
            self.gamma_slider, self.gamma_spin, "set_gamma",
            smin=1, smax=1000,
            get_pmin=lambda: self._gamma_min, get_pmax=lambda: self._gamma_max,
            ndigits=2,
        )
        # Review B-2: gain now uses _bind_param_controls instead of the
        # legacy int 1:1 wire + separate set_master_gain lambda. This is
        # the same fix as 56d1733 -- one-place wiring of display sync +
        # camera commit prevents the "slider syncs spin but never commits"
        # class of regression.
        self._bind_param_controls(
            self.gain_slider, self.gain_spin, "set_gain_db",
            smin=0, smax=1000,
            get_pmin=lambda: self._gain_min, get_pmax=lambda: self._gain_max,
            ndigits=2,
        )

        # AOI sliders <-> spinboxes (unchanged from uEye era).
        self.aoi_x_slider.valueChanged.connect(self._aoi_x_slider_to_spin)
        self.aoi_x_spin.valueChanged.connect(self._aoi_x_spin_to_slider)
        self.aoi_y_slider.valueChanged.connect(self._aoi_y_slider_to_spin)
        self.aoi_y_spin.valueChanged.connect(self._aoi_y_spin_to_slider)
        self.aoi_center_btn.clicked.connect(self._center_aoi)

        # Display transforms.
        self.rotation_combo.currentIndexChanged.connect(self._emit_rotation)
        self.flip_x_cb.toggled.connect(self.flip_x_changed.emit)
        self.flip_y_cb.toggled.connect(self.flip_y_changed.emit)

        # Save / Load / Revert.
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.load_btn.clicked.connect(self.load_requested.emit)
        self.revert_btn.clicked.connect(self.revert_requested.emit)

    # ----- slider <-> physical-value mapping (single source of truth) --------

    @staticmethod
    def _slider_to_phys(sval, smin, smax, pmin, pmax, ndigits=None):
        """Map an integer slider position to its physical value."""
        span = (smax - smin) or 1
        frac = max(0.0, min(1.0, (sval - smin) / span))
        phys = pmin + frac * (pmax - pmin)
        return round(phys, ndigits) if ndigits is not None else phys

    @staticmethod
    def _phys_to_slider(phys, smin, smax, pmin, pmax):
        """Inverse of _slider_to_phys, clamped to the slider range."""
        prng = pmax - pmin
        frac = (phys - pmin) / prng if prng > 0 else 0.0
        frac = max(0.0, min(1.0, frac))
        return int(round(smin + frac * (smax - smin)))

    def _bind_param_controls(self, slider, spin, cam_method, *, smin, smax,
                             get_pmin, get_pmax, ndigits=None):
        """Wire a slider<->spinbox pair for one camera parameter.

        The slider AND the spinbox each commit to the camera once per user
        gesture; the mirror update is done with blockSignals so the partner
        widget never re-emits (no double-fire, no feedback loop). The
        camera-thread reference is resolved lazily (slot by name) so this
        is wired ONCE at construction and survives the later
        ``connect_to_camera_thread`` call; before the camera exists the
        gesture still syncs the widgets and simply skips the camera call.

        Wiring the display sync and the camera commit in the SAME place is
        the structural fix for the exposure-slider omission (56d1733): a
        slider's display sync can no longer exist without its camera path
        -- they're the same connection."""
        def _cam():
            ct = getattr(self, "_cam_thread", None)
            return getattr(ct, cam_method, None) if ct is not None else None

        def _on_slider(sval):
            phys = self._slider_to_phys(
                sval, smin, smax, get_pmin(), get_pmax(), ndigits)
            spin.blockSignals(True)
            spin.setValue(phys)
            spin.blockSignals(False)
            fn = _cam()
            if fn is not None:
                fn(phys)

        def _on_spin(phys):
            slider.blockSignals(True)
            slider.setValue(
                self._phys_to_slider(phys, smin, smax, get_pmin(), get_pmax()))
            slider.blockSignals(False)
            fn = _cam()
            if fn is not None:
                fn(float(phys))

        slider.valueChanged.connect(_on_slider)
        spin.valueChanged.connect(_on_spin)

    # ----- AOI sync helpers ---------------------------------------------------

    def _aoi_x_slider_to_spin(self, v: int) -> None:
        v = (v // 4) * 4
        self.aoi_x_spin.blockSignals(True)
        self.aoi_x_spin.setValue(v)
        self.aoi_x_spin.blockSignals(False)

    def _aoi_x_spin_to_slider(self, v: int) -> None:
        self.aoi_x_slider.blockSignals(True)
        self.aoi_x_slider.setValue(v)
        self.aoi_x_slider.blockSignals(False)

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

    # ----- populate from camera_info dict -------------------------------------

    def update_from_camera_info(self, info: dict) -> None:
        """Refresh widget ranges + current values from a camera_info_signal
        dict. Reads Spinnaker-native keys (gain_db, acq_fps, exposure_us /
        exposure_ms, gamma, pixel_format, etc.). Blocks all widget signals
        during the update so populating doesn't re-fire commits back to the
        camera (the AUDIT:S2 echo-loop fix from the uEye era)."""
        widgets = [
            self.fps_spin, self.fps_slider, self.acq_fps_enable_cb,
            self.exposure_spin, self.exposure_slider,
            self.pixfmt_combo,
            self.gain_spin, self.gain_slider,
            self.gamma_spin, self.gamma_slider, self.gamma_enable_cb,
            self.packet_size_spin, self.throughput_spin,
            self.black_level_spin, self.blacklevel_clamp_cb,
            self.defect_correction_cb,
            self.aoi_width_spin, self.aoi_height_spin,
            self.aoi_x_spin, self.aoi_y_spin,
            self.aoi_x_slider, self.aoi_y_slider,
        ]
        for w in widgets:
            w.blockSignals(True)
        try:
            # Sensor + AOI ranges
            sw = int(info.get("sensor_width", 1280) or 1280)
            sh = int(info.get("sensor_height", 1024) or 1024)
            self._sensor_w = sw
            self._sensor_h = sh
            self.sensor_label.setText(f"Sensor: {sw} × {sh}")
            self.aoi_width_spin.setMaximum(sw)
            self.aoi_height_spin.setMaximum(sh)
            self.aoi_x_spin.setMaximum(max(0, sw - 4))
            self.aoi_y_spin.setMaximum(max(0, sh - 4))
            self.aoi_x_slider.setMaximum(max(0, sw - 4))
            self.aoi_y_slider.setMaximum(max(0, sh - 4))

            # Acq Frame Rate range + value
            fps_min = float(info.get("acq_fps_min", info.get("fps_min", 0.1)))
            fps_max = float(info.get("acq_fps_max", info.get("fps_max", 200.0)))
            fps_cur = float(info.get("acq_fps", 0.0))
            self._fps_min = fps_min
            self._fps_max = fps_max
            self.fps_range_label.setText(f"Acq FPS range: {fps_min:.2f} – {fps_max:.2f}")
            self.fps_spin.setRange(fps_min, fps_max)
            self.fps_spin.setValue(fps_cur)
            if fps_max > fps_min:
                self.fps_slider.setValue(
                    int(round((fps_cur - fps_min) / (fps_max - fps_min) * 10000))
                )
            self.acq_fps_enable_cb.setChecked(bool(info.get("acq_fps_enable", True)))

            # Exposure (ms) range + value -- exposure_us / _min / _max are
            # the Spinnaker-native keys (microseconds); we display in ms.
            exp_us_min = info.get("exposure_us_min")
            exp_us_max = info.get("exposure_us_max")
            exp_us_inc = info.get("exposure_us_inc")
            exp_us_cur = info.get("exposure_us")
            if exp_us_min is not None and exp_us_max is not None:
                self._exp_min = float(exp_us_min) / 1000.0
                self._exp_max = float(exp_us_max) / 1000.0
                exp_inc_ms = float(exp_us_inc) / 1000.0 if exp_us_inc else 0.001
                exp_cur_ms = (float(exp_us_cur) / 1000.0) if exp_us_cur else 30.0
            else:
                # Legacy fallback (ms keys).
                self._exp_min = float(info.get("exposure_min", 0.01))
                self._exp_max = float(info.get("exposure_max", 1000.0))
                exp_inc_ms = float(info.get("exposure_inc", 0.001) or 0.001)
                exp_cur_ms = float(info.get("exposure", 30.0))
            self.exposure_spin.setRange(self._exp_min, self._exp_max)
            self.exposure_spin.setSingleStep(max(exp_inc_ms, 0.001))
            self.exposure_spin.setValue(exp_cur_ms)
            if self._exp_max > self._exp_min:
                self.exposure_slider.setValue(
                    int(round((exp_cur_ms - self._exp_min) /
                              (self._exp_max - self._exp_min) * 10000))
                )
            self.exp_range_label.setText(
                f"Range: {self._exp_min:.3f} – {self._exp_max:.3f} ms"
            )

            # Pixel format combo
            pix_formats = info.get("pixel_formats", []) or []
            cur_fmt = str(info.get("pixel_format", "Mono8") or "Mono8")
            self.pixfmt_combo.clear()
            for f in pix_formats:
                self.pixfmt_combo.addItem(str(f))
            # Ensure the current format is present + selected even if the
            # enumerated list omitted it.
            idx = self.pixfmt_combo.findText(cur_fmt)
            if idx < 0:
                self.pixfmt_combo.addItem(cur_fmt)
                idx = self.pixfmt_combo.findText(cur_fmt)
            self.pixfmt_combo.setCurrentIndex(max(0, idx))

            # Gain (dB) range + value (Parity 2 -- float, not int 0-100)
            gain_db_min = float(info.get("gain_db_min", 0.0))
            gain_db_max = float(info.get("gain_db_max", 48.0))
            gain_db_cur = float(info.get("gain_db", 0.0))
            self._gain_min = gain_db_min
            self._gain_max = gain_db_max
            self.gain_range_label.setText(
                f"Range: {gain_db_min:.2f} – {gain_db_max:.2f} dB"
            )
            self.gain_spin.setRange(gain_db_min, gain_db_max)
            self.gain_spin.setValue(gain_db_cur)
            if gain_db_max > gain_db_min:
                self.gain_slider.setValue(
                    int(round((gain_db_cur - gain_db_min) /
                              (gain_db_max - gain_db_min) * 1000))
                )

            # Gamma + enable
            self._gamma_min = float(info.get("gamma_min", 0.01))
            self._gamma_max = float(info.get("gamma_max", 4.0))
            gamma_cur = float(info.get("gamma", 1.0))
            self.gamma_spin.setRange(self._gamma_min, self._gamma_max)
            self.gamma_spin.setValue(gamma_cur)
            grng = self._gamma_max - self._gamma_min
            gfrac = (gamma_cur - self._gamma_min) / grng if grng > 0 else 0.0
            self.gamma_slider.setValue(int(round(1 + max(0, min(1, gfrac)) * 999)))
            self.gamma_enable_cb.setChecked(bool(info.get("gamma_enable", False)))

            # GigE transport
            pkt = info.get("packet_size")
            if pkt:
                self.packet_size_spin.setValue(int(pkt))
            tput = info.get("throughput_limit") or 0
            tput_max = int(info.get("throughput_limit_max", 1_000_000_000) or 1_000_000_000)
            if tput_max > 0:
                self.throughput_spin.setMaximum(tput_max)
            self.throughput_spin.setValue(int(tput))

            # Decision-8 Blackfly-native. Each widget disables itself if the
            # corresponding node is unavailable on this Blackfly model. We
            # detect "unavailable" by checking that the camera_info key is
            # explicitly present -- get_camera_info always emits these keys
            # (with default 0/False) when the node is missing; but the
            # min/max range goes to 0/0 in that case, which lets us tell.
            bl_min = float(info.get("black_level_min", 0.0))
            bl_max = float(info.get("black_level_max", 0.0))
            bl_cur = float(info.get("black_level", 0.0))
            bl_available = (bl_max > bl_min)
            self.black_level_spin.setEnabled(bl_available)
            if bl_available:
                self.black_level_spin.setRange(bl_min, bl_max)
                self.black_level_spin.setValue(bl_cur)
            self.blacklevel_clamp_cb.setChecked(bool(info.get("black_level_clamping", True)))
            self.defect_correction_cb.setChecked(bool(info.get("defect_correction", False)))

            # AOI current values (Spinnaker-native roi_* keys preferred,
            # fall back to legacy aoi_* keys that camera.py also emits).
            aoi_w = int(info.get("roi_width", info.get("aoi_width", sw)))
            aoi_h = int(info.get("roi_height", info.get("aoi_height", sh)))
            aoi_x = int(info.get("roi_offset_x", info.get("aoi_x", 0)))
            aoi_y = int(info.get("roi_offset_y", info.get("aoi_y", 0)))
            self.aoi_width_spin.setValue(aoi_w)
            self.aoi_height_spin.setValue(aoi_h)
            self.aoi_x_spin.setValue(aoi_x)
            self.aoi_y_spin.setValue(aoi_y)
            self.aoi_x_slider.setValue(aoi_x)
            self.aoi_y_slider.setValue(aoi_y)
        finally:
            for w in widgets:
                w.blockSignals(False)

    # ----- save/load roundtrip ------------------------------------------------

    def get_current_settings(self) -> dict:
        """Read all current dock values into a dict for ``save_settings_to_ini``.
        Spinnaker-native key names; ``save_settings_to_ini`` in camera.py
        writes the [Image size] / [Timing] / [Analog] / [GigE] / [Display]
        sections from these keys."""
        k_map = {0: 0, 1: -1, 2: 2, 3: 1}
        rot_k = k_map.get(self.rotation_combo.currentIndex(), 0)

        return {
            # Timing
            "acq_frame_rate": self.fps_spin.value(),
            "acq_frame_rate_enable": self.acq_fps_enable_cb.isChecked(),
            "exposure_ms": self.exposure_spin.value(),
            # Pixel format
            "pixel_format": self.pixfmt_combo.currentText(),
            # Analog
            "gain_db": self.gain_spin.value(),
            "gamma": self.gamma_spin.value(),
            "gamma_enable": self.gamma_enable_cb.isChecked(),
            # GigE transport
            "gige_packet_size": self.packet_size_spin.value(),
            "device_link_throughput_limit": (
                self.throughput_spin.value() if self.throughput_spin.value() > 0
                else None
            ),
            # Blackfly native
            "black_level": self.black_level_spin.value(),
            "black_level_clamping": self.blacklevel_clamp_cb.isChecked(),
            "defect_correction": self.defect_correction_cb.isChecked(),
            # AOI (legacy key names preserved -- save_settings_to_ini writes
            # them to the [Image size] section).
            "aoi_width": self.aoi_width_spin.value(),
            "aoi_height": self.aoi_height_spin.value(),
            "aoi_x": self.aoi_x_spin.value(),
            "aoi_y": self.aoi_y_spin.value(),
            # Display (legacy keys -- written to the [Display] section).
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

    # ----- camera-thread wiring -----------------------------------------------

    def connect_to_camera_thread(self, cam_thread) -> None:
        """Wire dock-side widget events to the camera thread's parameter slots.
        Called once after the camera thread is created. The slider+spin
        camera commits are already wired by ``_wire_signals`` via
        ``_bind_param_controls`` (which resolves the slot reference lazily
        by string name on this stored ``_cam_thread``); only the
        non-slider widgets (checkboxes / combos / single spinboxes) are
        connected here."""
        self._cam_thread = cam_thread

        # Acq Frame Rate enable + Pixel Format
        self.acq_fps_enable_cb.toggled.connect(
            lambda v: cam_thread.set_frame_rate_enable(bool(v))
        )
        self.pixfmt_combo.currentIndexChanged.connect(
            lambda _i: cam_thread.set_pixel_format(self.pixfmt_combo.currentText())
        )

        # Gamma enable (gamma value itself is handled by _bind_param_controls).
        self.gamma_enable_cb.toggled.connect(
            lambda v: cam_thread.set_gamma_enable(bool(v))
        )

        # GigE transport
        self.packet_size_spin.valueChanged.connect(
            lambda v: cam_thread.set_packet_size(int(v))
        )
        self.throughput_spin.valueChanged.connect(
            lambda v: cam_thread.set_throughput_limit(int(v))
        )

        # Decision-8 Blackfly-native
        self.black_level_spin.valueChanged.connect(
            lambda v: cam_thread.set_black_level(float(v))
        )
        self.blacklevel_clamp_cb.toggled.connect(
            lambda v: cam_thread.set_black_level_clamping(bool(v))
        )
        self.defect_correction_cb.toggled.connect(
            lambda v: cam_thread.set_defect_correction(bool(v))
        )

        # AOI apply button
        self.aoi_apply_btn.clicked.connect(
            lambda: cam_thread.request_aoi_change(
                self.aoi_width_spin.value(),
                self.aoi_height_spin.value(),
                self.aoi_x_spin.value(),
                self.aoi_y_spin.value(),
            )
        )

        # camera_info_signal -> populate ranges
        cam_thread.camera_info_signal.connect(self.update_from_camera_info)
