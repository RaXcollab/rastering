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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg

from raster_paths import RasterSpec, iter_path_from_spec, collect_points
from raster_controller import load_last_calibration_path
from camera import UEyeCameraThread, UEyeConfig
from camera_settings_dock import CameraSettingsDock
from PyQt5 import QtCore, QtGui, QtWidgets, uic

UI_FILE = "raster_gui.ui"

# Optional: read default flip settings from config.py if available
try:
    import config as _config
except Exception:
    _config = None


TargetXY = Tuple[float, float]


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
        self._mode = "normal"   # normal | calibrate
        self._hull_points: List[TargetXY] = []
        self._update_ui_calibration_state(False)  # initial uncalibrated

        # last position history (for jogging points display)
        self._history: List[TargetXY] = []

        # Cached planned-raster preview points so we can re-filter on toggle
        # without regenerating the path iterator.
        self._raster_preview_pts: List[TargetXY] = []

        # Bundled camera_settings dict from the most recently loaded calibration.
        # Populated by note_loaded_cal_bundle; None until a cal with bundled
        # camera_settings is loaded. Drives the Apply-Camera-Settings button.
        self._loaded_cal_bundle_camera_settings: Optional[Dict[str, Any]] = None

        # Frametime metrics
        self._last_frame_time = time.perf_counter()
        self._fps_smoothed = None

        # Plot uses pixel coordinates (1:1 with image). The Scale (mm/px) widget
        # was removed; affine calibration handles target-space conversion when needed.
        self._img_scale: float = 1.0
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

        # --- Read live motor backlash and populate spinboxes (without firing
        #     editingFinished and re-sending the value back to the motor). Must
        #     happen after _connect_ui_actions so the spinboxes have signal
        #     handlers, and after the controller is constructed so the motor
        #     thread can service the read. Safe to fail (logs a warning).
        self._populate_backlash_from_motor()

        # --- Sync User Home widgets from controller state ---
        self._populate_user_home_from_controller()

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
        # F2 selection marker: hollow green ring, visually distinct from the red
        # filled live-target dot. Marks the selected path point before "Move".
        self.selection_marker = pg.ScatterPlotItem(
            size=14, symbol="o", pen=pg.mkPen("#00e000", width=2), brush=None
        )

        self.plot_widget.addItem(self.hull_scatter)
        self.plot_widget.addItem(self.raster_scatter)
        self.plot_widget.addItem(self.manual_scatter)
        self.plot_widget.addItem(self.current_target_marker)
        self.plot_widget.addItem(self.selection_marker)

        # Direction lines (optional)
        self._dir_items: List[pg.PlotDataItem] = []

        # Bounds rectangle
        self._bounds_item = None

        # Mouse click
        self.plot_widget.scene().sigMouseClicked.connect(self._on_plot_click)

        # Crosshair tracks mouse position in plot (pixel) coordinates.
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
            self._vline.setPos(float(mouse_point.x()))
            self._hline.setPos(float(mouse_point.y()))

    def _on_plot_click(self, event) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return
        mouse_point = self.vb.mapSceneToView(event.scenePos())
        x = float(mouse_point.x())
        y = float(mouse_point.y())

        # Ctrl+click -> SELECT the nearest raster path point (no motion). Modeless:
        # a plain click still does calibration / hull-point / spinbox behavior.
        if event.modifiers() & QtCore.Qt.ControlModifier:
            self._select_on_path(x, y)
            return

        # Populate Move-to-Position spinboxes with the click coordinate expressed
        # in MOTOR units. When calibrated, apply the affine transform to the click
        # (plot-space pixels) to get motor coordinates; pre-calibration, plot space
        # is interpreted directly as motor space (passthrough).
        if hasattr(self, "x") and hasattr(self, "y"):
            cal = getattr(self.controller, "calibration", None)
            if cal is not None:
                mx_click, my_click = cal.target_to_motor(x, y)
            else:
                mx_click, my_click = x, y
            self.x.setValue(float(mx_click))
            self.y.setValue(float(my_click))

        if self._mode == "calibrate":
            # Forward click to controller calibration collector (target-space pixels)
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
            self.cam_dock.timing_mode_combo.blockSignals(True)
            self.cam_dock.timing_mode_combo.setCurrentIndex(1 if cfg.prioritize_exposure else 0)
            self.cam_dock.timing_mode_combo.blockSignals(False)
            self.cam_dock.fps_spin.blockSignals(True)
            self.cam_dock.fps_spin.setValue(cfg.target_fps)
            self.cam_dock.fps_spin.blockSignals(False)

        # Apply imaging-quality fields (timing + gain + gamma + exposure).
        # Geometry (AOI + rotation + flip) goes through the shared helper.
        self.camera_thread.set_pixel_clock(cfg.pixel_clock_mhz)
        self.camera_thread.set_target_fps(cfg.target_fps)
        self.camera_thread.set_master_gain(cfg.master_gain)
        self.camera_thread.set_gain_boost(cfg.enable_gain_boost)
        self.camera_thread.set_gamma(cfg.gamma)
        self.camera_thread.set_exposure_ms(cfg.exposure_ms)

        # Rotation / flips are stored in a custom [Display] section in our
        # extended INIs; absent keys mean "leave alone". Read them BEFORE
        # calling the geometry helper so the helper sees None for missing
        # values.
        rot_k: Optional[int] = None
        fx: Optional[bool] = None
        fy: Optional[bool] = None
        try:
            disp = _load_display_settings_from_ini(ini_path)
            if "rotation_k" in disp:
                rot_k = int(disp["rotation_k"])
            if "flip_x" in disp:
                fx = bool(disp["flip_x"])
            if "flip_y" in disp:
                fy = bool(disp["flip_y"])
        except Exception:
            pass

        self._apply_camera_geometry(
            aoi_width=cfg.width,
            aoi_height=cfg.height,
            aoi_start_x=cfg.roi_offset_x,
            aoi_start_y=cfg.roi_offset_y,
            rotation_k=rot_k,
            flip_x=fx,
            flip_y=fy,
        )

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
                        self.cam_dock.rotation_combo.blockSignals(True)
                        self.cam_dock.rotation_combo.setCurrentIndex(k_to_index.get(self._rotation_k, 0))
                        self.cam_dock.rotation_combo.blockSignals(False)
                        self.cam_dock.flip_x_cb.blockSignals(True)
                        self.cam_dock.flip_x_cb.setChecked(self._flip_x)
                        self.cam_dock.flip_x_cb.blockSignals(False)
                        self.cam_dock.flip_y_cb.blockSignals(True)
                        self.cam_dock.flip_y_cb.setChecked(self._flip_y)
                        self.cam_dock.flip_y_cb.blockSignals(False)
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
            self.cam_dock.timing_mode_combo.blockSignals(True)
            self.cam_dock.timing_mode_combo.setCurrentIndex(1 if cfg.prioritize_exposure else 0)
            self.cam_dock.timing_mode_combo.blockSignals(False)
            self.cam_dock.fps_spin.blockSignals(True)
            self.cam_dock.fps_spin.setValue(cfg.target_fps)
            self.cam_dock.fps_spin.blockSignals(False)

        self.camera_thread.start()

        # Optionally apply extra .ini polish (hotpixel correction, etc.)
        if _config is not None and hasattr(_config, "APP_CONFIG"):
            ini_path = getattr(_config.APP_CONFIG.camera, "camera_params_ini", None)
            if ini_path and os.path.isfile(ini_path):
                # Apply .ini extras (hotpixel, hw gamma, AOI) via camera thread's
                # pending-parameter pattern — never touch _cam from the GUI thread.
                QtCore.QTimer.singleShot(2000, lambda: self.camera_thread.request_ini_extras(ini_path))

        # Exposure is edited from the Camera Settings dock only; the top-bar
        # spinbox was removed. Initialize the dock spinbox to the running config
        # value so it reflects what the camera was opened with.
        if hasattr(self, "cam_dock"):
            self.cam_dock.exposure_spin.blockSignals(True)
            self.cam_dock.exposure_spin.setValue(float(cfg.exposure_ms))
            self.cam_dock.exposure_spin.blockSignals(False)


    def _apply_image_scale(self, scale: float | None = None) -> None:
        """Re-apply the image-to-plot mapping. Plot uses pixel coordinates (scale=1)."""
        self._apply_image_mapping()

    def _apply_image_mapping(self) -> None:
        """
        Set the displayed ImageItem extents to match the frame's pixel dimensions.
        Plot coordinates equal pixel coordinates (1:1).
        """
        if self._last_frame_shape is None:
            return
        h, w = self._last_frame_shape
        self.img_item.setRect(QtCore.QRectF(0, 0, w, h))
        self.vb.setRange(xRange=(0, w), yRange=(0, h), padding=0.0)


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
        self._selected_index = -1

        # If UI file didn't provide them, create duplicates (fallback)
        if not hasattr(self, "raster_continuous_checkbox"):
            self.raster_continuous_checkbox = QtWidgets.QCheckBox("Continuous")
            self.raster_continuous_checkbox.setChecked(True)
            self.statusBar().addPermanentWidget(self.raster_continuous_checkbox)

        if not hasattr(self, "raster_step_button"):
            self.raster_step_button = QtWidgets.QPushButton("Step")
            self.raster_step_button.setEnabled(False)
            self.statusBar().addPermanentWidget(self.raster_step_button)

        # F2 "go to arbitrary site" controls (select-then-confirm). The spinbox
        # and Ctrl+click only SELECT a path point; the Move button is the sole
        # action that commits the motor move -- selection never moves motors.
        if not hasattr(self, "goto_index_spin"):
            self.goto_index_spin = QtWidgets.QSpinBox()
            self.goto_index_spin.setKeyboardTracking(False)
            self.goto_index_spin.setMinimum(0)
            self.goto_index_spin.setMaximum(0)
            self.goto_index_spin.setEnabled(False)
            self.goto_index_spin.setPrefix("pt ")
            self.statusBar().addPermanentWidget(self.goto_index_spin)
        if not hasattr(self, "goto_move_button"):
            self.goto_move_button = QtWidgets.QPushButton("Move to selected")
            self.goto_move_button.setEnabled(False)
            self.statusBar().addPermanentWidget(self.goto_move_button)

        # Set Tooltips
        self.raster_continuous_checkbox.setToolTip("Checked: run continuously.\nUnchecked: step mode.")
        self.raster_step_button.setToolTip("Advance one raster point.")
        self.goto_index_spin.setToolTip("Select a raster point by index (no motion).\nCtrl+click the image to select the nearest point.")
        self.goto_move_button.setToolTip("Move to the selected raster point.")

        # Wire signals
        # Note: We use try/disconnect to avoid double-wiring if this function runs twice
        try: self.raster_continuous_checkbox.stateChanged.disconnect()
        except: pass
        try: self.raster_step_button.clicked.disconnect()
        except: pass
        try: self.goto_index_spin.valueChanged.disconnect()
        except: pass
        try: self.goto_move_button.clicked.disconnect()
        except: pass

        self.raster_continuous_checkbox.stateChanged.connect(self._update_step_mode_ui)
        self.raster_step_button.clicked.connect(self._step_raster)
        self.goto_index_spin.valueChanged.connect(self._on_goto_index_changed)
        self.goto_move_button.clicked.connect(self._on_goto_move_clicked)

        self._update_step_mode_ui()

    def _update_ui_calibration_state(self, calibrated: bool) -> None:
        # All numerical inputs are interpreted as motor units regardless of calibration.
        # Calibration is still used for click-on-image -> motor mapping (see _on_plot_click).
        # `calibrated` is ignored; signature preserved for callers.
        del calibrated  # silence unused warning
        units = "motor units"

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

    # ------------------------------------------------------------------
    # F2: select-then-confirm "go to an arbitrary site" on the raster path.
    # Selection (spinbox / Ctrl+click) NEVER moves motors; only the explicit
    # "Move to selected" button commits the move.
    # ------------------------------------------------------------------

    def _on_goto_index_changed(self, n: int) -> None:
        """Spinbox changed -> SELECT point n (no motion). Uses the controller's
        armed path when running; falls back to the cached preview otherwise."""
        if getattr(self, "_raster_active_ui", False):
            self.controller.select_path_index(int(n))   # -> selection_changed_signal
        else:
            pts = self._raster_preview_pts
            if pts:
                i = max(0, min(int(n), len(pts) - 1))
                self._apply_selection(i, pts[i][0], pts[i][1])

    def _on_goto_move_clicked(self) -> None:
        """Explicit commit: move to the selected point. Never auto-moves on
        selection. Arms the raster in step mode first if needed."""
        if self.raster_continuous_checkbox.isChecked():
            self._log("Go-to-site is disabled in continuous mode. Uncheck Continuous.")
            return
        if self._selected_index < 0:
            self._log("No raster point selected.")
            return
        if not getattr(self, "_raster_active_ui", False):
            # Arm in step mode so the controller materializes the path.
            self._start_raster()
            if not getattr(self, "_raster_active_ui", False):
                self._log("Raster could not be armed for go-to-site.")
                return
        ok = self.controller.request_go_to_path_index(self._selected_index, source="ui")
        if not ok:
            self._log("Go-to-site rejected (no path, or a continuous run is in progress).")

    def _select_on_path(self, x: float, y: float) -> None:
        """Ctrl+click -> SELECT the nearest path point (no motion)."""
        if getattr(self, "_raster_active_ui", False):
            self.controller.select_nearest_path_point(x, y)   # -> selection_changed_signal
            return
        pts = self._raster_preview_pts
        if not pts:
            self._log("No raster path to select. Preview or arm a path first.")
            return
        best_i, best_d = 0, None
        for i, (px, py) in enumerate(pts):
            d = (px - x) ** 2 + (py - y) ** 2
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        self._apply_selection(best_i, pts[best_i][0], pts[best_i][1])

    def _apply_selection(self, i, x: float, y: float) -> None:
        """Update the selection marker + spinbox + Move button for selected point
        i at (x, y). i < 0 (or None) clears the selection."""
        cleared = (i is None or int(i) < 0)
        self._selected_index = -1 if cleared else int(i)
        if hasattr(self, "selection_marker"):
            if cleared:
                self.selection_marker.clear()
            else:
                self.selection_marker.setData([float(x)], [float(y)])
        if hasattr(self, "goto_move_button"):
            cont = self.raster_continuous_checkbox.isChecked() if hasattr(self, "raster_continuous_checkbox") else False
            self.goto_move_button.setEnabled((not cleared) and (not cont))
        if hasattr(self, "goto_index_spin") and not cleared:
            self.goto_index_spin.blockSignals(True)
            if self.goto_index_spin.maximum() < int(i):
                self.goto_index_spin.setMaximum(int(i))
            self.goto_index_spin.setValue(int(i))
            self.goto_index_spin.blockSignals(False)

    def _on_selection_changed(self, i: int, x: float, y: float) -> None:
        """Slot for controller.selection_changed_signal (i == -1 clears)."""
        self._apply_selection(i, x, y)




    # -------------------------
    # UI -> Controller wiring
    # -------------------------

    def _connect_ui_actions(self) -> None:
        # Buttons
        self.move_to_pos.clicked.connect(self._move_to_position)
        self.preview_pos.clicked.connect(self._preview_position)
        self.clearAllManual.clicked.connect(self._clear_manual_points)
        self.clearAllRasterManual.clicked.connect(self._clear_raster_points)

        self.jog_up_button_3.clicked.connect(lambda: self._jog(0, +1))
        self.jog_down_button_3.clicked.connect(lambda: self._jog(0, -1))
        self.jog_left_button_3.clicked.connect(lambda: self._jog(-1, 0))
        self.jog_right_button_3.clicked.connect(lambda: self._jog(+1, 0))

        # Device Home: Kinesis Home() to mechanical reference.
        self.device_home_x.clicked.connect(lambda: self.controller.request_home("X", hard=True))
        self.device_home_y.clicked.connect(lambda: self.controller.request_home("Y", hard=True))
        self.device_home_both.clicked.connect(self._device_home_both)

        # User Home: per-axis Set / Go and combined Home Both.
        self.user_home_x_set.clicked.connect(lambda: self._on_user_home_set("X"))
        self.user_home_y_set.clicked.connect(lambda: self._on_user_home_set("Y"))
        self.user_home_x_go.clicked.connect(lambda: self.controller.request_go_user_home("X"))
        self.user_home_y_go.clicked.connect(lambda: self.controller.request_go_user_home("Y"))
        self.user_home_both.clicked.connect(self._user_home_both)

        # Backlash commits on an explicit "Set" button -- a single, unambiguous
        # event. QDoubleSpinBox.editingFinished fires on BOTH Enter AND
        # focus-out with no de-dup, so an Enter-then-click-away enqueued the
        # Set twice (visible as duplicate "backlash X set to <v>" log lines).
        # The handler is fully non-blocking; see _on_backlash_set.
        self.x_backlash_set.clicked.connect(lambda: self._on_backlash_set("X"))
        self.y_backlash_set.clicked.connect(lambda: self._on_backlash_set("Y"))

        self.start_button.clicked.connect(self._start_raster)
        # REMOVE this line (already connected in _install_step_mode_controls)
        # self.raster_step_button.clicked.connect(self._step_raster)
        self.stop_button.clicked.connect(self.controller.stop_raster)
        self.path_button.clicked.connect(self._preview_raster_path)
        self.clearAll.clicked.connect(self._clear_raster_points)
        self.save_button.clicked.connect(self._save_and_clear_raster)

        self.bound_button.clicked.connect(self._display_bounds)

        self.calibrateButton.clicked.connect(self._enter_calibration_mode)
        self.useold.clicked.connect(self._on_use_last_calibration)
        self.resetButton.clicked.connect(self._reset_calibration_display)

        # Named-file calibration save / load + bundled camera-settings apply.
        self.saveCalibrationButton.clicked.connect(self._on_save_calibration)
        self.loadCalibrationButton.clicked.connect(self._on_load_calibration)
        self.applyCameraFromCalButton.clicked.connect(self._on_apply_camera_from_cal)

        # Display-options redraws — must happen immediately on user input, not
        # only when a new motor position arrives.
        self.point_display_count.valueChanged.connect(lambda _v: self._refresh_manual_scatter())
        self.show_all_points_checkbox.stateChanged.connect(lambda _s: self._refresh_manual_scatter())
        self.raster_point_display_count.valueChanged.connect(lambda _v: self._refresh_raster_scatter())
        self.show_all_raster_points_checkbox.stateChanged.connect(lambda _s: self._refresh_raster_scatter())
        self.show_current_marker_checkbox.stateChanged.connect(
            lambda s: self.current_target_marker.setVisible(bool(s))
        )


    def _jog(self, sx: int, sy: int) -> None:
        # sx, sy in {-1, 0, +1} are SCREEN directions: +x=right, +y=up.
        # Convert to motor-axis signs accounting for current display rotation/flips,
        # so "Jog Up" always moves the laser spot toward the top of the displayed
        # image regardless of orientation.
        msx, msy = self._screen_to_motor_unit_vector(int(sx), int(sy))
        # Step magnitudes are per motor axis, applied based on which motor axis
        # the screen direction maps to.
        step_x = float(self.dx_button.value())
        step_y = float(self.dy_button.value())
        dmx = float(msx) * step_x
        dmy = float(msy) * step_y
        self.controller.request_jog_motor(dmx, dmy, source="ui")

    def _screen_to_motor_unit_vector(self, sx: int, sy: int) -> Tuple[int, int]:
        """
        Map a screen-direction unit vector (sx, sy) — where +x is right and +y is
        up on the user's display — to a motor-axis unit vector (msx, msy).

        Pipeline:
          1. screen -> plot:    apply ViewBox flips (invertX/invertY).
          2. plot   -> camera:  apply inverse of np.rot90(_rotation_k) (image is
                                rotated before display; we undo that for deltas).
          3. camera -> motor:   physical mapping for this rastering rig:
                                motor_dx = -cam_drow, motor_dy = -cam_dcol.
                                (Empirically derived from the existing-confirmed
                                "Jog Right/Left" behavior with default
                                rotation_k=-1, no flips. If the hardware ever
                                changes, adjust the cam->motor block below.)
        """
        # 1. screen -> plot
        plot_dx = -sx if self._flip_x else sx
        plot_dy = -sy if self._flip_y else sy

        # 2. plot -> camera frame (invert the np.rot90(k) applied to displayed frame)
        k = int(self._rotation_k) % 4  # normalize -1 -> 3
        if k == 0:
            cam_dcol, cam_drow = plot_dx, plot_dy
        elif k == 1:    # 90° CCW
            cam_dcol, cam_drow = -plot_dy, plot_dx
        elif k == 2:    # 180°
            cam_dcol, cam_drow = -plot_dx, -plot_dy
        else:           # k == 3, 90° CW (this is _rotation_k = -1)
            cam_dcol, cam_drow = plot_dy, -plot_dx

        # 3. camera -> motor (rig-specific)
        motor_dx = -cam_drow
        motor_dy = -cam_dcol
        return int(motor_dx), int(motor_dy)

    def _move_to_position(self) -> None:
        # Spinbox values are interpreted as motor coordinates; route via motor-direct command.
        mx = float(self.x.value())
        my = float(self.y.value())
        self.controller.request_move_motor(mx, my, source="ui")

    def _preview_position(self) -> None:
        x = float(self.x.value())
        y = float(self.y.value())
        # purely visual
        self.manual_scatter.addPoints([x], [y])

    def _clear_manual_points(self) -> None:
        # Clears the manual jog history overlay. Does NOT touch
        # current_target_marker — that's controlled by the
        # "Show current position" checkbox so the live cursor isn't
        # blanked by an unrelated user action.
        self.manual_scatter.clear()
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

        # Enforce the drawn box as the controller's target-space bounds:
        # calibrated MOVE_TARGET commands (raster steps, manual moves, and
        # go-to-site) outside this box are now rejected by the controller.
        self.controller.set_target_bounds((xmin, xmax, ymin, ymax))

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

        # Cache the full preview so the Display-Options filter can re-render
        # the overlay on toggle without regenerating the iterator.
        self._raster_preview_pts = [(float(p[0]), float(p[1])) for p in pts]
        # Let the user pick a point index against the previewed path (selection
        # only; no motion until "Move to selected").
        if hasattr(self, "goto_index_spin") and not getattr(self, "_raster_active_ui", False):
            n = len(self._raster_preview_pts)
            self.goto_index_spin.setEnabled(n > 0)
            self.goto_index_spin.setMaximum(max(0, n - 1))
        self._refresh_raster_scatter()
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
        # Clear raster preview points/lines + cached preview
        self._raster_preview_pts = []
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

        # Clear any F2 selection tied to the (now-cleared) path.
        if hasattr(self, "selection_marker"):
            self._apply_selection(-1, 0.0, 0.0)
        if hasattr(self, "goto_index_spin") and not getattr(self, "_raster_active_ui", False):
            self.goto_index_spin.setEnabled(False)
            self.goto_index_spin.setMaximum(0)


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
    # Calibration mode
    # -------------------------

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
        # Clear the in-memory bundled camera_settings stash; without an active
        # calibration there's nothing to revert to, so disable the Apply button.
        self._loaded_cal_bundle_camera_settings = None
        if hasattr(self, "applyCameraFromCalButton"):
            self.applyCameraFromCalButton.setEnabled(False)
        self._update_ui_calibration_state(False)
        self._log("Calibration reset.")

    def _on_use_last_calibration(self) -> None:
        """'Use Last Value' button: reload the last-used bundled calibration
        (full apply: affine + user_home + backlash; bundled camera_settings
        re-enables the Apply button). Falls back to a status message if no
        last-used path is recorded yet."""
        if self.controller.is_raster_running:
            self._log("Cannot load calibration while raster is running.")
            return
        last_path = load_last_calibration_path()
        if not last_path:
            self._log("No last-used calibration recorded. Use 'Load Calibration...' first.")
            return
        try:
            data = self.controller.load_calibration_from_path(last_path)
        except Exception as e:
            self._log(f"Failed to reload last calibration: {e}")
            return
        self.note_loaded_cal_bundle(data, source_path=last_path)
        self._apply_loaded_backlash_widgets(data)
        self._populate_user_home_from_controller()

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
        c.backlash_reading_signal.connect(self._on_backlash_reading)
        c.selection_changed_signal.connect(self._on_selection_changed)


    def _populate_backlash_from_motor(self) -> None:
        """
        Read motor backlash for both axes and populate the spinboxes WITHOUT
        triggering editingFinished — which would round-trip the value right
        back to the motor. Also updates the Reading label per axis. Skips
        silently (with a log warning) if the read fails (e.g., motor not
        connected at startup).
        """
        for axis in ("X", "Y"):
            self._refresh_backlash_reading(axis, also_setpoint=True, context="startup")

    def _refresh_backlash_reading(self, axis: str, *, also_setpoint: bool = False, context: str = "refresh") -> None:
        """
        Read live motor backlash for `axis` and write it to the Reading label.
        If `also_setpoint` is True, also update the setpoint spinbox (with
        signals blocked to avoid re-firing editingFinished). `context` is
        used in failure log messages.
        """
        spinbox = self.x_backlash if axis == "X" else self.y_backlash
        reading_label = self.x_backlash_reading if axis == "X" else self.y_backlash_reading
        try:
            res = self.controller.request_get_backlash(axis, wait=True, timeout_s=2.0)
        except Exception as e:
            self._log(f"Could not read motor {axis} backlash at {context}: {e}")
            return
        if res is None or not res.ok or res.value is None:
            msg = res.message if res else "no result"
            self._log(f"Could not read motor {axis} backlash at {context}: {msg}")
            return
        value = float(res.value)
        reading_label.setText(f"{value:.5f}")
        if also_setpoint:
            spinbox.blockSignals(True)
            try:
                spinbox.setValue(value)
            finally:
                spinbox.blockSignals(False)

    def _set_backlash_widgets(self, axis: str, value: float) -> None:
        """Write `value` to both the Reading label and the Setpoint spinbox
        for `axis` (spinbox signals blocked for hygiene). Used by the
        calibration-load path to seed widgets from the loaded bundle."""
        reading = self.x_backlash_reading if axis == "X" else self.y_backlash_reading
        spin = self.x_backlash if axis == "X" else self.y_backlash
        reading.setText(f"{float(value):.5f}")
        spin.blockSignals(True)
        try:
            spin.setValue(float(value))
        finally:
            spin.blockSignals(False)

    def _apply_loaded_backlash_widgets(self, data: Dict[str, Any]) -> None:
        """After a calibration load, sync the Backlash widgets to the bundle.

        For axes whose backlash was in the bundle, load_calibration_from_path
        has ALREADY enqueued a (priority-100) request_set_backlash. We must
        NOT issue a re-read here: request_get_backlash is priority 50, so it
        dequeues AHEAD of that SET (the FIFO is a min-heap) and reads the
        stale pre-load backlash -- seeding the Setpoint spinbox wrong. Seed
        both widgets from the loaded value instead; the SET's own reply
        corrects the Reading label to the motor-accepted value via
        backlash_reading_signal (identical to the manual-Set path).

        Axes absent from the bundle (legacy schema) had no SET enqueued, so
        a plain idle-FIFO re-read is safe and correct there.
        """
        bl = data.get("backlash") if isinstance(data, dict) else None
        for axis in ("X", "Y"):
            v = bl.get(axis.lower()) if isinstance(bl, dict) else None
            if v is not None:
                self._set_backlash_widgets(axis, float(v))
            else:
                self._refresh_backlash_reading(axis, also_setpoint=True, context="after cal load")

    def _on_backlash_set(self, axis: str) -> None:
        """
        "Set" button handler: push the Setpoint to the motor as the new
        backlash. Fire-and-forget -- the GUI thread never waits on the motor
        FIFO, so the prompt stays responsive even while a Device Home runs
        (the Set simply queues behind it).

        The SET command self-reports the motor's post-set read-back: its
        result drives BOTH the status acknowledgment ("backlash <axis> set
        to <v>") and the Reading label (via backlash_reading_signal). No
        separate GET is issued -- a second command would priority-invert
        ahead of the Set (GET priority 50 < SET priority 100) and re-read
        the stale, pre-set value.
        """
        spinbox = self.x_backlash if axis == "X" else self.y_backlash
        self.controller.request_set_backlash(axis, float(spinbox.value()))

    def _on_backlash_reading(self, axis: str, value: float) -> None:
        """Async readback from a non-blocking GET_BACKLASH
        (backlash_reading_signal). Updates only the Reading label -- the
        Setpoint keeps the user's typed value so a slow readback never
        clobbers an in-progress edit."""
        label = self.x_backlash_reading if axis == "X" else self.y_backlash_reading
        label.setText(f"{float(value):.5f}")

    def _populate_user_home_from_controller(self) -> None:
        """Read User Home X / Y from the controller and populate both the
        Reading labels and the Setpoint spinboxes. Cheap and always succeeds
        (controller state is in-process)."""
        for axis in ("X", "Y"):
            v = float(self.controller.get_user_home(axis))
            self._set_user_home_widgets(axis, v)

    def _set_user_home_widgets(self, axis: str, value: float) -> None:
        """Write `value` to both the Reading label and the Setpoint spinbox
        for `axis`, blocking spinbox signals so editingFinished does not fire."""
        reading = self.user_home_x_reading if axis == "X" else self.user_home_y_reading
        spin = self.user_home_x_setpoint if axis == "X" else self.user_home_y_setpoint
        reading.setText(f"{value:.5f}")
        spin.blockSignals(True)
        try:
            spin.setValue(value)
        finally:
            spin.blockSignals(False)

    def _on_user_home_set(self, axis: str) -> None:
        """Commit the User Home setpoint to controller state. No motor motion."""
        spin = self.user_home_x_setpoint if axis == "X" else self.user_home_y_setpoint
        v = self.controller.set_user_home(axis, float(spin.value()))
        # Refresh reading label to reflect what was actually stored.
        reading = self.user_home_x_reading if axis == "X" else self.user_home_y_reading
        reading.setText(f"{float(v):.5f}")
        self._log(f"User Home {axis} set to {float(v):.5f}")

    def _device_home_both(self) -> None:
        """Enqueue Device Home for X then Y. The motor command FIFO serializes
        the two operations naturally."""
        self.controller.request_home("X", hard=True)
        self.controller.request_home("Y", hard=True)

    def _user_home_both(self) -> None:
        """Move both axes to the stored User Home (mx, my) via a single full-XY
        MOVE_MOTOR. Symmetric to _device_home_both for readability."""
        self.controller.request_go_user_home(None)

    # -------------------------
    # Named-file calibration save / load
    # -------------------------

    def _get_cal_bundled_camera_settings(self) -> Optional[Dict[str, Any]]:
        """
        Snapshot the geometry-relevant subset of the camera dock's current
        settings: AOI (width/height/start_x/start_y), rotation_k, flip_x/y.
        Excludes imaging-quality settings (pixel clock, exposure, gain, gamma)
        -- those are not tied to the calibration.
        Returns None if the camera dock isn't installed yet.
        """
        if not hasattr(self, "cam_dock") or self.cam_dock is None:
            return None
        d = self.cam_dock.get_current_settings()
        return {
            "aoi": {
                "width": int(d["aoi_width"]),
                "height": int(d["aoi_height"]),
                "start_x": int(d["aoi_x"]),
                "start_y": int(d["aoi_y"]),
            },
            "rotation_k": int(d["rotation_k"]),
            "flip_x": bool(d["flip_x"]),
            "flip_y": bool(d["flip_y"]),
        }

    def _apply_bundled_camera_settings(self, cs: Dict[str, Any]) -> None:
        """
        Apply the bundled camera_settings dict produced by
        _get_cal_bundled_camera_settings. Routes through the shared
        _apply_camera_geometry helper so the code path is identical to
        a manual INI load for the AOI + rotation + flip fields.
        """
        aoi = cs.get("aoi") or {}
        self._apply_camera_geometry(
            aoi_width=int(aoi.get("width", 0)),
            aoi_height=int(aoi.get("height", 0)),
            aoi_start_x=int(aoi.get("start_x", 0)),
            aoi_start_y=int(aoi.get("start_y", 0)),
            rotation_k=int(cs["rotation_k"]) if "rotation_k" in cs else None,
            flip_x=bool(cs["flip_x"]) if "flip_x" in cs else None,
            flip_y=bool(cs["flip_y"]) if "flip_y" in cs else None,
        )
        self._log("Applied bundled camera settings from calibration.")

    def _apply_camera_geometry(
        self,
        *,
        aoi_width: int,
        aoi_height: int,
        aoi_start_x: int,
        aoi_start_y: int,
        rotation_k: Optional[int],
        flip_x: Optional[bool],
        flip_y: Optional[bool],
    ) -> None:
        """
        Apply the geometry-relevant subset of camera settings: AOI + display
        rotation + display flips. Shared by _apply_ini_to_running_camera
        (manual INI load) and _apply_bundled_camera_settings (calibration
        bundle apply).

        rotation_k / flip_x / flip_y of None mean "leave the existing value
        alone" -- matches the INI-load behavior where the [Display] section
        keys are optional.

        AOI is sent asynchronously to the camera thread; rotation / flip /
        ViewBox transforms are applied synchronously on the UI thread. This
        ordering matches the prior open-coded paths.
        """
        if not hasattr(self, "camera_thread") or self.camera_thread is None:
            self._log("No camera thread running; cannot apply camera geometry.")
            return

        try:
            self.camera_thread.request_aoi_change(
                int(aoi_width), int(aoi_height), int(aoi_start_x), int(aoi_start_y)
            )
        except Exception as e:
            self._log(f"Failed to apply AOI: {e}")

        if rotation_k is not None:
            self._rotation_k = int(rotation_k)
        if flip_x is not None:
            self._flip_x = bool(flip_x)
        if flip_y is not None:
            self._flip_y = bool(flip_y)

        if hasattr(self, "cam_dock"):
            k_to_index = {0: 0, -1: 1, 2: 2, 1: 3}
            self.cam_dock.rotation_combo.blockSignals(True)
            self.cam_dock.rotation_combo.setCurrentIndex(k_to_index.get(self._rotation_k, 0))
            self.cam_dock.rotation_combo.blockSignals(False)
            self.cam_dock.flip_x_cb.blockSignals(True)
            self.cam_dock.flip_x_cb.setChecked(self._flip_x)
            self.cam_dock.flip_x_cb.blockSignals(False)
            self.cam_dock.flip_y_cb.blockSignals(True)
            self.cam_dock.flip_y_cb.setChecked(self._flip_y)
            self.cam_dock.flip_y_cb.blockSignals(False)

        if hasattr(self, "vb"):
            self.vb.invertX(self._flip_x)
            self.vb.invertY(self._flip_y)

        self._last_frame_shape = None  # force display recalculation

    def _on_save_calibration(self) -> None:
        """Save the current calibration + bundled state to a user-chosen file."""
        if self.controller.is_raster_running:
            self._log("Cannot save calibration while raster is running.")
            return
        if self.controller.calibration is None:
            self._log("No calibration to save. Calibrate first.")
            return
        default_dir = os.getcwd()
        default_name = "cal_" + time.strftime("%Y%m%d_%H%M") + ".json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration As...", os.path.join(default_dir, default_name),
            "Calibration JSON (*.json);;All files (*.*)"
        )
        if not path:
            return
        notes, _ok = QtWidgets.QInputDialog.getText(
            self, "Calibration notes (optional)",
            "Short description (helps identify this calibration later):"
        )
        try:
            cs = self._get_cal_bundled_camera_settings()
            self.controller.save_calibration_to_path(path, camera_settings=cs, notes=str(notes or ""))
        except Exception as e:
            self._log(f"Failed to save calibration: {e}")

    def _on_load_calibration(self) -> None:
        """Browse for and load a calibration JSON, applying motor-coord
        parameters immediately."""
        if self.controller.is_raster_running:
            self._log("Cannot load calibration while raster is running.")
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Calibration...", os.getcwd(),
            "Calibration JSON (*.json);;All files (*.*)"
        )
        if not path:
            return
        try:
            data = self.controller.load_calibration_from_path(path)
        except Exception as e:
            self._log(f"Failed to load calibration: {e}")
            return
        self.note_loaded_cal_bundle(data, source_path=path)
        # Backlash + user home were applied by the controller; sync widgets.
        # _apply_loaded_backlash_widgets avoids a priority-50 GET that would
        # invert ahead of the just-enqueued priority-100 SET (review Issue 1).
        self._apply_loaded_backlash_widgets(data)
        self._populate_user_home_from_controller()

    def note_loaded_cal_bundle(self, data: Dict[str, Any], *, source_path: str) -> None:
        """Stash the bundled camera_settings dict (if present) and enable
        the Apply-Camera-Settings button. Called from both _on_load_calibration
        and the startup auto-load path in main_rastering.py."""
        cs = data.get("camera_settings") if isinstance(data, dict) else None
        self._loaded_cal_bundle_camera_settings = cs if isinstance(cs, dict) else None
        if hasattr(self, "applyCameraFromCalButton"):
            self.applyCameraFromCalButton.setEnabled(self._loaded_cal_bundle_camera_settings is not None)
        if self._loaded_cal_bundle_camera_settings is not None:
            self._log(f"Loaded calibration with bundled camera settings: {os.path.basename(source_path)}")
        else:
            self._log(f"Loaded calibration (no bundled camera settings): {os.path.basename(source_path)}")

    def _on_apply_camera_from_cal(self) -> None:
        """Apply the camera_settings block from the most recently loaded
        calibration. The button is disabled until such a bundle is loaded."""
        cs = getattr(self, "_loaded_cal_bundle_camera_settings", None)
        if not cs:
            self._log("No bundled camera settings available. Load a calibration first.")
            return
        if self.controller.is_raster_running:
            self._log("Cannot apply camera settings while raster is running.")
            return
        self._apply_bundled_camera_settings(cs)


    def _on_target_position(self, x: float, y: float) -> None:
        # Update current marker + history
        self.current_target_marker.setData([x], [y])

        if self.checkBox_2.isChecked():  # Save position history
            self._history.append((float(x), float(y)))

        self._refresh_manual_scatter()

    def _refresh_manual_scatter(self) -> None:
        """Redraw the manual-scatter overlay from `_history`, applying the
        Display-Points / Last-N filter. Safe to call when motor is idle."""
        if self.show_all_points_checkbox.isChecked():
            pts = self._history
        else:
            n = int(self.point_display_count.value())
            pts = self._history[-n:] if n > 0 else []

        if pts:
            self.manual_scatter.setData([p[0] for p in pts], [p[1] for p in pts])
        else:
            self.manual_scatter.clear()

    def _refresh_raster_scatter(self) -> None:
        """Redraw the raster-preview overlay from the cached `_raster_preview_pts`,
        applying the Display-Raster-Points / Last-N filter. Safe to call when
        motor is idle."""
        if not self._raster_preview_pts:
            self.raster_scatter.clear()
            return
        if self.show_all_raster_points_checkbox.isChecked():
            pts = self._raster_preview_pts
        else:
            n = int(self.raster_point_display_count.value())
            pts = self._raster_preview_pts[-n:] if n > 0 else []
        if pts:
            self.raster_scatter.setData([p[0] for p in pts], [p[1] for p in pts])
        else:
            self.raster_scatter.clear()

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
        # cal is AffineCalibration. The controller emits a rich "Calibration complete:
        # scale~..., cond(A)~..." status message; we just populate the matrix display
        # and clear calibrate mode here.
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
        self._mode = "normal"

    def _on_calibration_failed(self, msg: str) -> None:
        self._log(msg)
        self._mode = "normal"

    def _on_raster_state(self, active: bool) -> None:
        self._raster_active_ui = bool(active)
        self._log("Raster active." if active else "Raster inactive.")
        self._update_step_mode_ui()

        # Disable calibration file operations while raster is running: the
        # save flow snapshots motor backlash via wait=True request_get_backlash
        # which would block the motor thread mid-raster; load would clobber
        # the active coordinate frame.
        for btn_name in ("saveCalibrationButton", "loadCalibrationButton"):
            btn = getattr(self, btn_name, None)
            if btn is not None:
                btn.setEnabled(not active)
        if active:
            # Apply-Camera-Settings is also unsafe while running; re-enable
            # only if we have a bundle loaded.
            if hasattr(self, "applyCameraFromCalButton"):
                self.applyCameraFromCalButton.setEnabled(False)
        else:
            if hasattr(self, "applyCameraFromCalButton"):
                self.applyCameraFromCalButton.setEnabled(
                    getattr(self, "_loaded_cal_bundle_camera_settings", None) is not None
                )

        # lock in the mode choice while active
        if hasattr(self, "raster_continuous_checkbox"):
            self.raster_continuous_checkbox.setEnabled(not active)

        if hasattr(self, "raster_step_button"):
            is_step_mode = hasattr(self, "raster_continuous_checkbox") and (not self.raster_continuous_checkbox.isChecked())
            self.raster_step_button.setEnabled(active and is_step_mode)

        # F2 go-to-site widgets: while armed the controller holds the
        # materialized path -> size the index spinbox to it; when stopped,
        # disable the spinbox and clear any selection (the path is gone).
        if hasattr(self, "goto_index_spin"):
            if active:
                total = int(getattr(self.controller, "_raster_total_steps", 0))
                self.goto_index_spin.setEnabled(total > 0)
                self.goto_index_spin.setMaximum(max(0, total - 1))
            else:
                self.goto_index_spin.setEnabled(False)
                self._apply_selection(-1, 0.0, 0.0)

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
