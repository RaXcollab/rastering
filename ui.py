"""
ui.py

Qt UI layer built from the Qt Designer file raster_gui.ui (see UI_FILE).

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
from raster_controller import load_last_calibration_path, save_user_defaults, load_user_defaults
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

# Two faces of raster_remote_arm_button (see _update_step_mode_ui).
_ARM_TIP = ("Arm the configured path for BLACS-driven stepping "
            "(BLACS also auto-arms if you skip this).")
_TAKE_BACK_TIP = (
    "Take the raster back: Control returns to Local and holds -- BLACS steps\n"
    "are acknowledged without advancing until you hand control back (here or\n"
    "in the BLACS tab). Stop tears the path down for real.")
_REMOTE_OWNED_TIP = (
    "BLACS owns this raster -- Auto Raster and Step are locked out so a local "
    "click can't re-arm it from scratch or step it behind BLACS's back.\n"
    "Press 'Return to local control' to drive it from the GUI, or Stop to disarm.")
_STEP_TIP = "Advance one raster point."
_GOTO_TIP = "Move to the selected raster point."
# Go-to-site is the one local action that OVERRIDES remote ownership instead of
# being locked out by it: a targeted operator move is unambiguous intent.
_GOTO_TAKEOVER_TIP = (
    "Move to the selected raster point.\n"
    "BLACS owns this raster -- moving takes local control, and it HOLDS: "
    "BLACS cannot reclaim it by stepping. Hand it back with 'Give to BLACS' "
    "or the tab's Control toggle.")


class RasterMainWindow(QtWidgets.QMainWindow):
    # Bridge for ZMQ-initiated arm requests: emitted from the ZMQ server
    # thread, delivered (queued connection) to the Qt main thread where the
    # raster-spec widgets live. Payload: (want_continuous, reply callable).
    _remote_arm_requested = QtCore.pyqtSignal(bool, object)

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
        self._bounds_inited_from_frame = False
        self._move_preview_pts: List[TargetXY] = []
        self._update_ui_calibration_state(False)  # initial uncalibrated
        # Gate Auto Raster Start/Step on calibration from the first paint:
        # uncalibrated -> disabled with a "Calibrate first" reason.
        self._update_step_mode_ui()

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

        # --- Apply persisted UI defaults LAST, so a saved settings_defaults.json
        #     overrides the live-read backlash / fresh widget values. No-op when
        #     the file doesn't exist (typical first run).
        self._apply_user_defaults()

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
        self.move_preview_scatter = pg.ScatterPlotItem(size=12, symbol="x", pen=pg.mkPen("#00d0d0"))
        self.current_target_marker = pg.ScatterPlotItem(size=10, brush=pg.mkBrush("#ff0000"))
        # F2 selection marker: hollow green ring, visually distinct from the red
        # filled live-target dot. Marks the selected path point before "Move".
        self.selection_marker = pg.ScatterPlotItem(
            size=14, symbol="o", pen=pg.mkPen("#00e000", width=2), brush=None
        )

        self.plot_widget.addItem(self.hull_scatter)
        self.plot_widget.addItem(self.raster_scatter)
        self.plot_widget.addItem(self.manual_scatter)
        self.plot_widget.addItem(self.move_preview_scatter)
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
            self._init_bounds_from_frame(w, h)   # one-time full-frame scan-bounds default
            if getattr(self, "_dead_zone_items", None):
                # AOI / camera-settings changes move the frame under the
                # shading; recompute so unreachable columns are never
                # unmarked (or phantom-marked) after a reshape.
                self._draw_dead_zone()
        
    def closeEvent(self, event):
        try:
            self._close_pos_history_file()
        except Exception:
            pass
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

        # Normal mode: in hull mode a click adds a convex-hull vertex; otherwise
        # a click drops a "where Move-to-Position will go" preview dot at the spot.
        alg = self.alg_choice.currentText().lower() if hasattr(self, "alg_choice") else ""
        if "hull" in alg or "convex" in alg:
            self._hull_points.append((x, y))
            self.hull_scatter.setData([p[0] for p in self._hull_points], [p[1] for p in self._hull_points])
            # Keep an existing hull preview in sync as vertices are added.
            if self._raster_preview_pts:
                self._on_raster_param_changed()
        else:
            self._add_move_preview_point(x, y)

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

    def _init_bounds_from_frame(self, w: int, h: int) -> None:
        """One-time: default the scan bounds + spiral center to the full camera
        frame (pixels) so a fresh raster covers the whole image. Guarded so it
        never overwrites the operator's edits after the first frame."""
        if self._bounds_inited_from_frame:
            return
        self._bounds_inited_from_frame = True
        for name, val in (("xlow", 0.0), ("xhigh", float(w)), ("ylow", 0.0), ("yhigh", float(h)),
                          ("spiral_cx", w / 2.0), ("spiral_cy", h / 2.0)):
            if hasattr(self, name):
                getattr(self, name).setValue(val)


    # -------------------------
    # Raster mode helpers (step vs continuous)
    # -------------------------

    def _install_raster_mode_controls(self) -> None:
        """
        Build the stepping / remote-control widgets the .ui doesn't provide and
        put them in the Automatic Controls tab (raster_gui.ui gives us only the
        Continuous checkbox + Step button, in autoModeLayout).
        Mode semantics:
        - Continuous checked  => controller runs automatically (continuous raster)
        - Continuous unchecked => controller is armed; user/ZMQ advances via Step/move_to_next
        """
        self._raster_active_ui = False
        self._selected_index = -1
        self._selected_xy = None
        self._last_raster_source = None
        self._pos_history_file = None
        self._pos_history_write_warned = False
        if not hasattr(self, "_raster_preview_pts"):
            self._raster_preview_pts = []

        # Home for the stepping / remote widgets: a group box appended to the
        # Automatic Controls tab's own layout (autoLayout in raster_gui.ui),
        # right under Preview/Auto Raster/Stop + Continuous/Step -- these belong
        # with the auto-raster controls, not in the far corner of the status bar.
        _auto_layout = getattr(self, "autoLayout", None)
        if _auto_layout is not None:
            self.raster_remote_group = QtWidgets.QGroupBox("Stepping / Remote control")
            _grid = QtWidgets.QGridLayout(self.raster_remote_group)
            _grid.setContentsMargins(6, 4, 6, 4)
            _auto_layout.insertWidget(2, self.raster_remote_group)

            def _place(w, row, col, span=1):
                _grid.addWidget(w, row, col, 1, span)
        else:
            # ponytail: no Automatic Controls tab (stripped .ui) -- fall back to
            # the old status-bar home rather than crash the operator's GUI.
            def _place(w, row, col, span=1):
                self.statusBar().addPermanentWidget(w)

        # If UI file didn't provide them, create duplicates (fallback).
        # raster_gui.ui places these two in autoModeLayout already.
        if not hasattr(self, "raster_continuous_checkbox"):
            self.raster_continuous_checkbox = QtWidgets.QCheckBox("Continuous")
            self.raster_continuous_checkbox.setChecked(True)
            _place(self.raster_continuous_checkbox, 0, 0)

        if not hasattr(self, "raster_step_button"):
            self.raster_step_button = QtWidgets.QPushButton("Step")
            self.raster_step_button.setEnabled(False)
            _place(self.raster_step_button, 0, 1)

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
            _place(self.goto_index_spin, 1, 0)
        if not hasattr(self, "goto_move_button"):
            self.goto_move_button = QtWidgets.QPushButton("Move to selected")
            self.goto_move_button.setEnabled(False)
            _place(self.goto_move_button, 1, 1)

        # Remote (BLACS) control: an always-visible indicator of who owns the
        # raster, plus an explicit "arm it for BLACS" button that doubles as
        # "take it back". The indicator is display-only -- Stop stays live while
        # BLACS drives (Auto Raster and Step do not; see _update_step_mode_ui).
        if not hasattr(self, "raster_remote_arm_button"):
            self.raster_remote_arm_button = QtWidgets.QPushButton("Arm for remote stepping")
            _place(self.raster_remote_arm_button, 2, 0, 2)
        if not hasattr(self, "raster_rearm_button"):
            self.raster_rearm_button = QtWidgets.QPushButton("Re-arm from pending")
            self.raster_rearm_button.setToolTip(
                "Replace the armed path with the pattern currently on screen.\n"
                "Does not move the motors and does not advance the cursor, so it\n"
                "cannot desync BLACS's shot count.")
            # Row 4, not the plan's row 3: rows 3/0 and 3/1 are already taken by
            # raster_source_label and raster_shots_label below.
            _place(self.raster_rearm_button, 4, 0, 2)
            self.raster_rearm_button.clicked.connect(self._on_rearm_clicked)
        if not hasattr(self, "raster_source_label"):
            self.raster_source_label = QtWidgets.QLabel()
            _place(self.raster_source_label, 3, 0)
        if not hasattr(self, "raster_shots_label"):
            self.raster_shots_label = QtWidgets.QLabel()
            _place(self.raster_shots_label, 3, 1)
        self._on_raster_source(None)
        self._on_raster_shots_per_step(None)

        # Set Tooltips
        self.raster_continuous_checkbox.setToolTip("Checked: run continuously.\nUnchecked: step mode.")
        self.raster_step_button.setToolTip(_STEP_TIP)
        self.goto_index_spin.setToolTip("Select a raster point by index (no motion).\nCtrl+click the image to select the nearest point.")
        self.goto_move_button.setToolTip(_GOTO_TIP)
        self.raster_shots_label.setToolTip(
            "Shots BLACS fires at each raster point before asking for the next "
            "one.\nDisplay only -- set it on the BLACS Rastering tab.")

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
        try: self.raster_remote_arm_button.clicked.disconnect()
        except: pass

        self.raster_continuous_checkbox.stateChanged.connect(self._update_step_mode_ui)
        self.raster_step_button.clicked.connect(self._step_raster)
        self.raster_remote_arm_button.clicked.connect(self._arm_for_remote)
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
        Sole owner of Auto Raster / Step / Move-to-selected enablement -- every
        gate (calibration, remote ownership, active state, continuous run) is
        decided here, so no caller may setEnabled those three directly. Stop and
        Preview Path are never gated: Stop is the operator's kill switch,
        Preview touches no motors.

        Enable Step only when:
        - raster is active, AND
        - continuous is unchecked.

        Lock the mode checkbox while raster is active to avoid controller/UI drift.
        """
        active = bool(getattr(self, "_raster_active_ui", False))
        continuous = bool(self.raster_continuous_checkbox.isChecked())
        # Auto Raster Start + Step require a calibration: without one, target-space
        # path points map straight onto motor units (passthrough) and the raster
        # would drive the motors to nonsense positions. Gate both, with a reason.
        calibrated = getattr(self.controller, "calibration", None) is not None
        _cal_hint = "Calibrate first -- raster needs a calibration to map target coordinates to motor positions."
        # While BLACS owns an active raster the local drivers are locked out, not
        # merely redundant: Start re-arms from scratch and Step advances the
        # cursor BLACS is counting on (ownership is a persistent flag only human
        # actions change -- a BLACS step can no longer seize it back).
        # "Return to local control" and Stop are the ways back.
        remote_owned = active and self._last_raster_source == "remote"
        _hint = _cal_hint if not calibrated else (_REMOTE_OWNED_TIP if remote_owned else "")

        if hasattr(self, "start_button"):
            self.start_button.setEnabled(calibrated and not remote_owned)
            self.start_button.setToolTip(_hint)

        self.raster_step_button.setEnabled(
            calibrated and active and (not continuous) and not remote_owned)
        self.raster_step_button.setToolTip(_hint or _STEP_TIP)

        if hasattr(self, "goto_move_button"):
            # Go-to-site is deliberately NOT gated on ownership: clicking it takes
            # the raster back (request_go_to_path_index) rather than being locked
            # out. Only a genuinely continuous run blocks it -- there the run loop
            # owns the cursor -- so read the controller, never the local checkbox,
            # which drifts when BLACS re-arms an already-armed raster.
            self.goto_move_button.setEnabled(
                calibrated and getattr(self, "_selected_index", -1) >= 0
                and not self.controller.is_continuous)
            self.goto_move_button.setToolTip(
                _cal_hint if not calibrated
                else (_GOTO_TAKEOVER_TIP if remote_owned else _GOTO_TIP))
        if hasattr(self, "raster_remote_arm_button"):
            # One button, three faces (single clicked connection --
            # _arm_for_remote branches on the same state): arm for BLACS while
            # idle, take the raster back while BLACS owns an active one, hand it
            # over while we hold an active one. Without the extra faces the
            # button greys out on arming with no way back either direction.
            if remote_owned:
                self.raster_remote_arm_button.setText("Return to local control")
                self.raster_remote_arm_button.setEnabled(True)
                self.raster_remote_arm_button.setToolTip(_TAKE_BACK_TIP)
            elif active:
                # Armed and locally held: third face -- hand it back. Without
                # this the tab checkbox is the ONLY route back to BLACS, and
                # spec section 2 promises the axis is settable from either
                # screen.
                self.raster_remote_arm_button.setText("Give to BLACS")
                self.raster_remote_arm_button.setEnabled(True)
                self.raster_remote_arm_button.setToolTip(
                    "Hand this armed raster to BLACS: its next move_to_next "
                    "advances from the current point.")
            else:
                self.raster_remote_arm_button.setText("Arm for remote stepping")
                self.raster_remote_arm_button.setEnabled(calibrated)
                self.raster_remote_arm_button.setToolTip(
                    _ARM_TIP if calibrated else _cal_hint)
        self.raster_continuous_checkbox.setEnabled(not active)
        # "Delay (s)" only applies to continuous runs -- grey it out in step mode
        # so it's clear it has no effect there.
        if hasattr(self, "sleepTimer"):
            self.sleepTimer.setEnabled(continuous)
            self.sleepTimer.setToolTip("Delay between points (continuous mode only; ignored in step mode).")


    def _update_armed_pending_status(self) -> None:
        """Say plainly when the armed path and the on-screen pattern differ.
        Silent when they match -- the operator only needs telling when the
        thing that will run is not the thing they are looking at."""
        if not getattr(self, "_raster_active_ui", False):
            return
        armed = len(self.controller.armed_path_points())
        pending = len(self._raster_preview_pts)
        if pending and pending != armed:
            self._log(f"armed {armed} pts | pending {pending} pts "
                      f"-- press Re-arm to run the pattern on screen")

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


    def _arm_for_remote(self) -> None:
        """The dual-purpose remote button. Which action fires is decided by the
        same state that picked the button's label in _update_step_mode_ui:

        - BLACS owns an active raster -> take it back (local control).
        - we hold an active raster -> hand it to BLACS in place.
        - otherwise -> arm the configured path in STEP mode for BLACS to drive.

        Arming takes the same path the ZMQ remote-arm slot does, minus the reply
        plumbing -- it just saves the operator the "wait for BLACS to auto-arm"
        round trip. Failures land on the status bar via _start_raster.
        """
        if getattr(self, "_raster_active_ui", False):
            if self._last_raster_source == "remote":
                self.controller.take_local_control()
            else:
                # Third face: hand the armed raster to BLACS in place.
                self.controller.give_remote_control()
            return
        self.raster_continuous_checkbox.setChecked(False)
        self._start_raster(source="remote")

    def _on_raster_shots_per_step(self, n) -> None:
        """Controller -> UI: shots-per-step BLACS last programmed (None = unknown)."""
        self.raster_shots_label.setText(
            "Shots/step: --" if n is None else f"Shots/step: {int(n)}")

    def _on_raster_source(self, source) -> None:
        """Controller -> UI: who owns the raster (None / "local" / "remote").

        Indicator plus the remote button's mode: Stop and "Return to local
        control" stay enabled in remote mode so the operator can always take the
        raster back -- Auto Raster and Step are the ones that get locked out.
        """
        self._last_raster_source = source
        self._update_step_mode_ui()
        if source == "remote":
            self.raster_source_label.setText("Control: REMOTE (BLACS)")
            self.raster_source_label.setStyleSheet("color: #cc7000; font-weight: bold;")
        elif source == "local":
            self.raster_source_label.setText("Control: Local")
            self.raster_source_label.setStyleSheet("")
        else:
            self.raster_source_label.setText("Control: --")
            self.raster_source_label.setStyleSheet("")

    def _request_remote_arm(self, want_continuous: bool, reply) -> None:
        """Controller's ``remote_arm_provider``. Runs on the ZMQ server
        thread -- touch no widgets here; just hop to the main thread."""
        self._remote_arm_requested.emit(bool(want_continuous), reply)

    def _on_remote_arm_requested(self, want_continuous: bool, reply) -> None:
        """Main-thread slot: arm the raster from the GUI's configured path on
        behalf of a ZMQ client (BLACS). Always calls ``reply`` exactly once;
        failure reasons go back over the wire, not only to the status bar."""
        try:
            if getattr(self.controller, "calibration", None) is None:
                self._log("ZMQ arm request refused: no calibration set.")
                reply(False, "not_calibrated",
                      "no calibration set; calibrate in the GUI first")
                return
            if getattr(self, "_raster_active_ui", False):
                # Raced with a local arm between the server's active-check and
                # this slot; the raster is active, which is what was asked for.
                reply(True)
                return
            self.raster_continuous_checkbox.setChecked(bool(want_continuous))
            self._start_raster(source="remote")
            if getattr(self, "_raster_active_ui", False):
                reply(True)
            else:
                # _start_raster already logged the specific reason (no points,
                # spec error, ...); send the actionable cause to the client.
                reply(False, "no_raster_configured",
                      "raster could not be armed; check the path settings in the GUI")
        except Exception as e:
            reply(False, "arm_failed", f"arm failed: {e}")

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
        selection. Arms the raster in step mode first if needed.

        Resolves the move by the selected COORDINATE (re-select-nearest on the
        armed path), not the bare preview index -- so editing the raster spec
        between Preview and Move can't send the motors to the wrong site.
        """
        # Gate on the controller, not the Continuous checkbox: the checkbox is a
        # local mode *request* that drifts from the armed run (BLACS re-arming an
        # already-armed raster switches the controller to step mode without
        # touching it), and a stale checked box was blocking goto outright.
        if self.controller.is_continuous:
            self._log("Go-to-site is disabled while a continuous raster is running. Press Stop first.")
            return
        if self._selected_index < 0 or self._selected_xy is None:
            self._log("No raster point selected.")
            return
        took_over = (getattr(self, "_raster_active_ui", False)
                     and self._last_raster_source == "remote")
        if not getattr(self, "_raster_active_ui", False):
            # Arm in step mode so the controller materializes the path.
            self._start_raster()
            if not getattr(self, "_raster_active_ui", False):
                self._log("Raster could not be armed for go-to-site.")
                return
        # Re-resolve against the ARMED controller path by coordinate: the preview
        # the index was picked from may be stale (spec edited since Preview).
        self.controller.select_nearest_path_point(self._selected_xy[0], self._selected_xy[1])
        ok = self.controller.goto_selected_point(source="ui")
        if not ok:
            self._log("Go-to-site rejected (no path, or a continuous run is in progress).")
        elif took_over:
            self._log("Go-to-site took local control -- BLACS reclaims on its next stepped shot.")

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
        self._selected_xy = None if cleared else (float(x), float(y))
        if hasattr(self, "selection_marker"):
            if cleared:
                self.selection_marker.clear()
            else:
                self.selection_marker.setData([float(x)], [float(y)])
        self._update_step_mode_ui()     # sole owner of the Move button's gates
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

        self.enforce_bounds_checkbox.stateChanged.connect(self._on_enforce_bounds_toggled)

        self.calibrateButton.clicked.connect(self._enter_calibration_mode)
        self.useold.clicked.connect(self._on_use_last_calibration)
        self.resetButton.clicked.connect(self._reset_calibration_display)

        # Named-file calibration save / load + bundled camera-settings apply.
        self.saveCalibrationButton.clicked.connect(self._on_save_calibration)
        self.loadCalibrationButton.clicked.connect(self._on_load_calibration)
        self.applyCameraFromCalButton.clicked.connect(self._on_apply_camera_from_cal)

        self.save_defaults_button.clicked.connect(self._on_save_defaults)

        # Display-options redraws — must happen immediately on user input, not
        # only when a new motor position arrives.
        self.point_display_count.valueChanged.connect(lambda _v: self._refresh_manual_scatter())
        self.show_all_points_checkbox.stateChanged.connect(lambda _s: self._refresh_manual_scatter())
        self.raster_point_display_count.valueChanged.connect(lambda _v: self._refresh_raster_scatter())
        self.show_all_raster_points_checkbox.stateChanged.connect(lambda _s: self._refresh_raster_scatter())
        self.show_current_marker_checkbox.stateChanged.connect(
            lambda s: self.current_target_marker.setVisible(bool(s))
        )

        # Live preview auto-refresh: when any raster-spec input changes, re-render
        # the preview overlay to match (only if a preview is currently shown).
        # keyboardTracking(False) -> valueChanged fires once per commit
        # (Enter / focus-out), not on every keystroke.
        raster_spinboxes = [
            self.xstep, self.ystep, self.xlow, self.xhigh, self.ylow, self.yhigh,
            self.radius_spiral, self.step_spiral, self.angle_spiral, self.ang_change,
        ]
        if hasattr(self, "spiral_cx"):
            raster_spinboxes += [self.spiral_cx, self.spiral_cy]
        for sb in raster_spinboxes:
            sb.setKeyboardTracking(False)
            sb.valueChanged.connect(self._on_raster_param_changed)
        self.alg_choice.currentIndexChanged.connect(self._on_raster_param_changed)
        self.show_direction_checkbox.stateChanged.connect(self._on_raster_param_changed)

        # "Save position history" now writes a CSV to disk while checked.
        self.checkBox_2.stateChanged.connect(self._on_save_history_toggled)


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
        """Toggle the move-preview dots: if any are shown, clear them; otherwise
        seed from the current target spinboxes (rendered at the IMAGE-pixel
        location via the inverse affine when calibrated). Subsequent clicks
        accumulate; press again to clear."""
        if self._move_preview_pts:
            self._move_preview_pts.clear()
            self.move_preview_scatter.clear()
            self._log("Move-preview dots cleared.")
            return
        mx, my = float(self.x.value()), float(self.y.value())
        cal = getattr(self.controller, "calibration", None)
        px, py = cal.motor_to_target(mx, my) if cal is not None else (mx, my)
        self._add_move_preview_point(px, py)
        self._log("Move-preview: showing current target. Click the image to add more; press again to clear.")

    def _add_move_preview_point(self, x: float, y: float) -> None:
        """Drop a 'where Move-to-Position will go' marker at the clicked image
        location (pixel space). Separate from manual_scatter (owned by the motor
        history) so _refresh_manual_scatter can't clobber it."""
        self._move_preview_pts.append((float(x), float(y)))
        self.move_preview_scatter.setData([p[0] for p in self._move_preview_pts],
                                          [p[1] for p in self._move_preview_pts])

    def _clear_manual_points(self) -> None:
        # Clears the manual jog history overlay. Does NOT touch
        # current_target_marker — that's controlled by the
        # "Show current position" checkbox so the live cursor isn't
        # blanked by an unrelated user action.
        self.manual_scatter.clear()
        self._history.clear()
        if hasattr(self, "move_preview_scatter"):
            self._move_preview_pts.clear()
            self.move_preview_scatter.clear()

    # -------------------------
    # User defaults (settings_defaults.json)
    # -------------------------

    def _gather_user_defaults(self) -> Dict[str, Any]:
        # Short timeout: never block the GUI thread on the Save button if the
        # motor is busy (e.g. a manual Device Home mid-flight). A None readback
        # persists null backlash for that axis, which _apply_user_defaults skips.
        bx, by = self.controller._read_motor_backlash_xy(timeout_s=2.0)
        uhx, uhy = self.controller.get_user_home_xy()
        return {
            "backlash": {"x": bx, "y": by},
            "user_home": {"x": float(uhx), "y": float(uhy)},
            "jog_step": {"x": float(self.dx_button.value()), "y": float(self.dy_button.value())},
            "display": {
                "point_display_count": int(self.point_display_count.value()),
                "show_all_points": bool(self.show_all_points_checkbox.isChecked()),
                "raster_point_display_count": int(self.raster_point_display_count.value()),
                "show_all_raster_points": bool(self.show_all_raster_points_checkbox.isChecked()),
                "show_current_marker": bool(self.show_current_marker_checkbox.isChecked()),
                "show_direction": bool(self.show_direction_checkbox.isChecked()),
            },
        }

    def _on_save_defaults(self) -> None:
        if self.controller.is_raster_running:
            self._log("Cannot save defaults while a raster is running.")
            return
        try:
            save_user_defaults(self._gather_user_defaults())
            self._log("Saved current backlash / user home / jog step / display options as defaults.")
        except Exception as e:
            self._log(f"Failed to save defaults: {e}")

    def _apply_user_defaults(self) -> None:
        d = load_user_defaults()
        if not d:
            return
        disp = d.get("display", {})
        for name, key in (("point_display_count", "point_display_count"),
                          ("raster_point_display_count", "raster_point_display_count")):
            if hasattr(self, name) and key in disp:
                getattr(self, name).setValue(int(disp[key]))
        for name, key in (("show_all_points_checkbox", "show_all_points"),
                          ("show_all_raster_points_checkbox", "show_all_raster_points"),
                          ("show_current_marker_checkbox", "show_current_marker"),
                          ("show_direction_checkbox", "show_direction")):
            if hasattr(self, name) and key in disp:
                getattr(self, name).setChecked(bool(disp[key]))
        js = d.get("jog_step", {})
        if hasattr(self, "dx_button") and "x" in js:
            self.dx_button.setValue(float(js["x"]))
        if hasattr(self, "dy_button") and "y" in js:
            self.dy_button.setValue(float(js["y"]))
        uh = d.get("user_home", {})
        if "x" in uh and "y" in uh:
            self.controller.set_user_home_xy(float(uh["x"]), float(uh["y"]))
            if hasattr(self, "_populate_user_home_from_controller"):
                self._populate_user_home_from_controller()
        bl = d.get("backlash", {})
        for axis_key, axis in (("x", "X"), ("y", "Y")):
            v = bl.get(axis_key)
            if v is not None:
                try:
                    self.controller.request_set_backlash(axis, float(v))
                except Exception as e:
                    self._log(f"Default backlash {axis} not applied: {e}")

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
            # Dedicated spiral origin (independent of the scan bounds); fall back
            # to the bounds center if the inputs aren't present. The spiral is
            # still clipped to `bounds`.
            ox = float(self.spiral_cx.value()) if hasattr(self, "spiral_cx") else cx
            oy = float(self.spiral_cy.value()) if hasattr(self, "spiral_cy") else cy
            return RasterSpec(
                kind="spiral",
                bounds=bounds,
                origin=(ox, oy),
                radius=radius,
                step=step,
                angle_step=angle_step,
                angle_step_change=angle_step_change,
            )

        # hull raster
        hull_pts = list(self._hull_points)
        # Convex hull fills its OWN clicked region (its bbox), independent of the
        # scan-bounds spinboxes. Passing the (small/defaulted) scan bounds here
        # clipped a hull clicked across the 0..500px image down to 0 points.
        return RasterSpec(
            kind="hull",
            bounds=None,
            xstep=xstep,
            ystep=ystep,
            hull_points=hull_pts,
            hull_order="xy",
        )


    def _on_enforce_bounds_toggled(self, _state) -> None:
        """Checkbox: ON draws + enforces the scan-bounds box (rejects raster/
        go-to-site MOVES outside it); OFF clears the box + enforcement. Move-
        rejection only -- it does NOT change the raster region (the preview always
        uses the bound spinboxes). While checked, the box + enforcement track the
        limit spinboxes live (see _on_raster_param_changed)."""
        if self.enforce_bounds_checkbox.isChecked():
            self._draw_and_enforce_bounds()
            xmin, xmax, ymin, ymax = self._current_bounds()
            self._log(f"Move-enforcement ON: x[{xmin}, {xmax}] y[{ymin}, {ymax}] -- raster/go-to-site moves outside are rejected (manual motor moves unaffected).")
        else:
            self._clear_bounds()
            self._log("Move-enforcement OFF (raster region unchanged).")

    def _draw_and_enforce_bounds(self) -> None:
        """Draw the scan-bounds box AND enforce it on the controller. Idempotent
        redraw -- safe to call on every limit change while the box is shown."""
        xmin, xmax, ymin, ymax = self._current_bounds()
        if getattr(self, "_bounds_item", None) is not None:
            try:
                self.plot_widget.removeItem(self._bounds_item)
            except Exception:
                pass
            self._bounds_item = None
        rect = QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
        self._bounds_item = QtWidgets.QGraphicsRectItem(rect)
        self._bounds_item.setPen(pg.mkPen("#cc6600"))
        self._bounds_item.setBrush(pg.mkBrush("#ebce191a"))
        self.plot_widget.addItem(self._bounds_item)
        self.controller.set_target_bounds((xmin, xmax, ymin, ymax))

    def _draw_dead_zone(self) -> None:
        """Shade the part of the frame the motors cannot reach. Recomputed on
        every calibration change: the unreachable region is a property of the
        mapping, not of the image, so it moves when the calibration does."""
        for item in getattr(self, "_dead_zone_items", []):
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self._dead_zone_items = []

        cal = getattr(self.controller, "calibration", None)
        bounds = getattr(self.controller, "motor_bounds", None)
        if cal is None or bounds is None:
            return

        # _last_frame_shape is (h, w) from the live camera frame; the 500x500
        # default only applies before the first frame has landed.
        shape = getattr(self, "_last_frame_shape", None)
        h, w = shape if shape else (500, 500)
        step = 10  # px; a coarse mask is enough to show the operator the edge
        xs, ys = [], []
        for px in range(0, int(w), step):
            for py in range(0, int(h), step):
                if not self.controller._within_bounds(
                        self.controller._target_to_motor_clamped(cal, (px, py)),
                        bounds):
                    xs.append(px)
                    ys.append(py)
        if not xs:
            return
        item = pg.ScatterPlotItem(
            xs, ys, size=step, pxMode=False, pen=None,
            brush=pg.mkBrush(200, 40, 40, 60), symbol="s")
        # Stacking: the camera ImageItem and every overlay default to z=0,
        # ordered by insertion -- an item added later at z<=-? is a trap:
        # ViewBox only re-bumps z below its own -100. Pin the image UNDER
        # everything once, and sit the shading between image and overlays.
        if hasattr(self, "img_item"):
            self.img_item.setZValue(-1)
        item.setZValue(-0.5)  # above the image, below all z=0 overlays
        self.plot_widget.addItem(item)
        self._dead_zone_items = [item]

    def _clear_bounds(self) -> None:
        """Remove the scan-bounds box and turn OFF the controller's enforcement."""
        if getattr(self, "_bounds_item", None) is not None:
            try:
                self.plot_widget.removeItem(self._bounds_item)
            except Exception:
                pass
            self._bounds_item = None
        try:
            self.controller.clear_target_bounds()
        except Exception:
            pass

    def _preview_raster_path(self) -> None:
        # Preview a fresh path: clear the overlay + any stale go-to-site selection,
        # but KEEP the convex-hull vertices -- they are the INPUT for hull mode, so
        # clearing them here would make hull Preview always fail "needs 3 points".
        self._clear_raster_overlay()
        if hasattr(self, "selection_marker"):
            self._apply_selection(-1, 0.0, 0.0)
        self._render_preview(quiet=False)

    def _clear_raster_overlay(self) -> None:
        """Clear the rendered PENDING path -- whichever layer currently holds it
        -- WITHOUT touching the convex-hull input points or the F2 selection, so
        the live auto-refresh on a param change can't wipe the hull input.

        While armed that layer is the grey pending scatter and raster_scatter is
        left alone: it holds the RUNNING path, and every caller here also runs
        while armed (param change, Preview Path, Clear All), so a pending-side
        clear must not be able to blank the armed display for the rest of a run.
        """
        self._clear_pending_overlay()
        if not getattr(self, "_raster_active_ui", False):
            self.raster_scatter.clear()
        for item in self._dir_items:
            try:
                self.plot_widget.removeItem(item)
            except Exception:
                pass
        self._dir_items.clear()

    def _ensure_pending_scatter(self) -> None:
        """Lazy pending overlay: grey open circles, visually distinct from the
        armed path's filled dots so the two can NEVER be confused."""
        if getattr(self, "pending_scatter", None) is None:
            self.pending_scatter = pg.ScatterPlotItem(
                pen=pg.mkPen("#999999", width=1), brush=None,
                symbol="o", size=6)
            self.plot_widget.addItem(self.pending_scatter)

    def _clear_pending_overlay(self) -> None:
        """Clear ONLY the pending (grey) overlay + its cache. Never touches
        raster_scatter: while armed that shows the running path, and a
        pending-side clear must not be able to blank it."""
        self._raster_preview_pts = []
        if getattr(self, "pending_scatter", None) is not None:
            self.pending_scatter.clear()

    def _render_preview(self, *, quiet: bool = False) -> None:
        """Build the path from the CURRENT settings and draw the overlay. The
        caller clears the overlay first. Used by the Preview button (quiet=False)
        and the live auto-refresh on param change (quiet=True)."""
        try:
            spec = self._build_raster_spec()
        except Exception as e:
            if not quiet:
                self._log(f"Preview Path error: {e}")
            return

        # Hull requires points
        if spec.kind == "hull" and (not spec.hull_points or len(spec.hull_points) < 3):
            if not quiet:
                self._log("Convex Hull raster requires at least 3 hull points (click to add points).")
            return

        # NOTE: the path generators defer their body to the first next(), so
        # iter_path_from_spec can't raise yet -- collect_points is where a
        # degenerate hull / too-fine grid / step=0 actually raises. Wrap BOTH.
        try:
            it = iter_path_from_spec(spec)
            pts = collect_points(it, max_points=50000)
        except Exception as e:
            if not quiet:
                self._log(f"Preview Path error: {e}")
            return

        if not pts:
            # Surface even in quiet (auto-refresh) mode -- an empty preview is
            # noteworthy. Hint per pattern kind (the overlay was already cleared).
            hints = {
                "spiral": "spiral center/radius/step may not intersect the scan bounds",
                "hull": "step sizes may be larger than the hull, or the hull is degenerate",
                "square_x": "check the step sizes against the scan bounds",
                "square_y": "check the step sizes against the scan bounds",
            }
            hint = hints.get(spec.kind, "check the pattern parameters")
            self._log(f"Preview: 0 points for the current '{spec.kind}' settings ({hint}).")
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
        armed = bool(getattr(self, "_raster_active_ui", False))
        if armed:
            # Pending-while-armed: its OWN grey overlay, never the armed one.
            # Direction lines are deliberately skipped -- grey lines tracing
            # the pending pattern over the armed dots is exactly the
            # read-off-the-wrong-path failure this split exists to prevent.
            self._ensure_pending_scatter()
            self.pending_scatter.setData(
                [p[0] for p in self._raster_preview_pts],
                [p[1] for p in self._raster_preview_pts])
        else:
            self._refresh_raster_scatter()
        if not quiet:
            self._log(f"Preview Path: {len(pts)} points.")

        # Optional direction lines (idle only -- see the armed branch above)
        if (not armed and hasattr(self, "show_direction_checkbox")
                and self.show_direction_checkbox.isChecked()):
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

    def _on_raster_param_changed(self, *args) -> None:
        """Live-refresh the PENDING preview so it always matches the current
        raster settings. Only refreshes an EXISTING preview; it now runs while
        armed too -- the armed path is drawn from the controller, so pending
        edits can no longer be mistaken for the path that is running."""
        # Keep the scan-bounds box + its enforcement in sync with the limit
        # spinboxes whenever the box is currently shown (so enforcement never
        # silently lags the displayed limits).
        if getattr(self, "_bounds_item", None) is not None and not getattr(self, "_raster_active_ui", False):
            self._draw_and_enforce_bounds()
        # No longer returns early while armed. The armed path is drawn from the
        # controller (see _refresh_raster_scatter), so a live pending preview
        # can no longer be mistaken for it -- freezing it was the lie.
        if not self._raster_preview_pts:
            return
        # While armed this clears only the pending layer (see
        # _clear_raster_overlay), so an early return in _render_preview (spec
        # raise, hull<3, 0 points) can no longer leave the ARMED path invisible
        # for the rest of the run.
        self._clear_raster_overlay()
        self._render_preview(quiet=True)
        self._update_armed_pending_status()

    def _clear_raster_points(self) -> None:
        # Clear All: rendered overlay + hull input + F2 selection. While armed
        # the overlay clear drops only the pending layer -- the armed path is
        # running state, not a pattern the operator drew; Stop clears that.
        self._clear_raster_overlay()

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

    def _on_rearm_clicked(self) -> None:
        """Swap pending into armed. Deliberately NOT gated on ownership: this
        changes WHICH path is armed, never the cursor, so BLACS's shot count
        cannot desync. Ownership: carry the LIVE flag through when a raster
        is active (BLACS keeps driving the swapped path); arm as local when
        idle. Never _last_raster_source -- that is sticky across Stop (stop
        preserves ownership since 2026-08-07), so a stale 'remote' would hand
        a freshly drawn pattern to BLACS and lock the operator out of Step."""
        if getattr(self.controller, "_raster_active", False):
            src = getattr(self.controller, "_raster_source", None) or "local"
        else:
            src = "local"
        self._start_raster(source=src, rearm=True)

    def _start_raster(self, *, source: str = "local", rearm: bool = False) -> None:
        # `source` is keyword-only: start_button.clicked would otherwise pass its
        # `checked` bool in as the source.
        # Hard guard: never raster without a calibration. An uncalibrated raster
        # runs in passthrough (target pixels treated as motor units) and drives the
        # motors to nonsense positions. Belt-and-suspenders with the disabled Start
        # button (also covers Step's arm path and any programmatic caller).
        if getattr(self.controller, "calibration", None) is None:
            self._log("Calibrate first -- raster needs a calibration to map target coordinates to motor positions. Use Calibrate, or load / Use-Last a saved calibration.")
            return
        # Never re-arm a raster BLACS owns. The greyed-out Start button is the
        # real gate; this catches the click that lands in the window where
        # ownership already flipped remote but the queued raster_source_signal
        # hasn't repainted the button yet. Read the controller, not the UI mirror.
        if (not rearm
                and getattr(self.controller, "_raster_active", False)
                and getattr(self.controller, "_raster_source", None) == "remote"):
            self._log("BLACS owns the raster -- press 'Return to local control' or Stop before starting a local one.")
            return
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

        # Log dir from config if available; per-pass JSON logs default off
        log_dir = None
        try:
            if _config is not None:
                paths = getattr(getattr(_config, "APP_CONFIG", None), "paths", None)
                if getattr(paths, "raster_log_enabled", False):
                    log_dir = getattr(paths, "raster_log_dir", None)
        except Exception:
            log_dir = None

        continuous = bool(self.raster_continuous_checkbox.isChecked())
        delay_s = 0.0
        if hasattr(self, "sleepTimer"):
            delay_s = float(self.sleepTimer.value())

        self.controller.start_raster(it, continuous=continuous, log_dir=log_dir, delay_s=(delay_s if continuous else 0.0), source=source)

        self._update_step_mode_ui()
        self._log(f"Raster started: {spec.kind}")
        # Arming consumes the pending pattern -- it IS the armed one now, so
        # the grey overlay (and any direction lines tracing it) is stale.
        self._clear_raster_overlay()
        self._refresh_raster_scatter()

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
        # Calibration cleared -> re-disable Auto Raster Start/Step.
        self._update_step_mode_ui()
        self._draw_dead_zone()   # cal is None -> clears the shading
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
        c.raster_source_signal.connect(self._on_raster_source)
        c.raster_shots_per_step_signal.connect(self._on_raster_shots_per_step)
        c.raster_finished_signal.connect(lambda: self._log("Raster finished."))
        c.raster_log_path_signal.connect(lambda p: self._log(f"Raster log: {p}"))

        c.command_done_signal.connect(self._on_command_done)
        c.backlash_reading_signal.connect(self._on_backlash_reading)
        c.selection_changed_signal.connect(self._on_selection_changed)

        # ZMQ-initiated arm: server thread -> _request_remote_arm (emit only)
        # -> queued signal -> _on_remote_arm_requested (main thread).
        self._remote_arm_requested.connect(self._on_remote_arm_requested)
        c.remote_arm_provider = self._request_remote_arm


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
        # A calibration is now loaded -> enable Auto Raster Start/Step.
        self._update_step_mode_ui()
        self._draw_dead_zone()

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
            f = getattr(self, "_pos_history_file", None)
            if f is not None:
                try:
                    f.write(f"{time.time()},{x},{y}\n")
                    f.flush()
                except Exception as e:
                    if not getattr(self, "_pos_history_write_warned", False):
                        self._pos_history_write_warned = True
                        self._log(f"Position-history write failed (further errors suppressed): {e}")

        self._refresh_manual_scatter()

    def _on_save_history_toggled(self, *args) -> None:
        """Open/close the position-history CSV. While 'Save position history' is
        checked, each target position is appended to a timestamped CSV (in the
        raster log dir if configured, else cwd); unchecking closes the file."""
        if self.checkBox_2.isChecked():
            if getattr(self, "_pos_history_file", None) is not None:
                return  # already open
            d = None
            try:
                if _config is not None:
                    d = getattr(getattr(getattr(_config, "APP_CONFIG", None), "paths", None), "raster_log_dir", None)
            except Exception:
                d = None
            d = d or os.getcwd()
            try:
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, f"position_history_{time.strftime('%Y%m%d_%H%M%S')}.csv")
                self._pos_history_file = open(path, "w", newline="")
                self._pos_history_file.write("timestamp,x,y\n")
                self._pos_history_write_warned = False
                self._log(f"Saving position history -> {path}")
            except Exception as e:
                self._pos_history_file = None
                self._log(f"Could not open position-history file: {e}")
        else:
            self._close_pos_history_file()

    def _close_pos_history_file(self) -> None:
        f = getattr(self, "_pos_history_file", None)
        if f is not None:
            try:
                f.close()
                self._log("Position history saved.")
            except Exception:
                pass
            self._pos_history_file = None

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
        # While armed, the ARMED path is the truth -- never the preview cache,
        # which freezes on arm and is what made the screen disagree with the
        # running path (2026-08-07 incident).
        if getattr(self, "_raster_active_ui", False):
            source_pts = self.controller.armed_path_points()
        else:
            source_pts = self._raster_preview_pts
        if not source_pts:
            self.raster_scatter.clear()
            return
        if self.show_all_raster_points_checkbox.isChecked():
            pts = source_pts
        else:
            n = int(self.raster_point_display_count.value())
            pts = source_pts[-n:] if n > 0 else []
        if pts:
            self.raster_scatter.setData([p[0] for p in pts], [p[1] for p in pts])
        else:
            self.raster_scatter.clear()

    def _on_motor_position(self, mx: float, my: float) -> None:
        # Dual-frame readout. Motor mm is always real (fixed 0-12 travel);
        # pixels exist only while a calibration defines them -- uncalibrated
        # target coords ARE mm (controller passthrough), so printing those as
        # pixels would be a frame lie. Hence "px N/A" rather than a number.
        cal = getattr(self.controller, "calibration", None)
        if cal is not None:
            px, py = cal.motor_to_target(mx, my)
            px_txt, py_txt = f"px {px:.1f}", f"px {py:.1f}"
        else:
            px_txt = py_txt = "px N/A"
        if hasattr(self, "motor_x_pos"):
            self.motor_x_pos.setText(f"{mx:.5f} mm | {px_txt}")
        if hasattr(self, "motor_y_pos"):
            self.motor_y_pos.setText(f"{my:.5f} mm | {py_txt}")

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
        # Calibration now exists -> enable Auto Raster Start/Step.
        self._update_step_mode_ui()
        self._draw_dead_zone()

    def _on_calibration_failed(self, msg: str) -> None:
        self._log(msg)
        self._mode = "normal"

    def _on_raster_state(self, active: bool) -> None:
        self._raster_active_ui = bool(active)
        self._log("Raster active." if active else "Raster inactive.")
        # The overlay's SOURCE changes with this flag (armed path vs pending
        # preview), so it must be redrawn here or the screen keeps showing the
        # pre-arm preview -- including points the arm-time filter dropped.
        # start_raster emits True unconditionally, so a Re-arm lands here too.
        self._refresh_raster_scatter()
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

        # Auto Raster + Step enablement is _update_step_mode_ui's alone (called
        # above): it also weighs calibration and remote ownership, which a bare
        # active-state rule here would silently override.

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

        # Stop is live exactly while something is running -- the operator's kill
        # switch, never gated on calibration or on who owns the raster.
        try:
            self.stop_button.setEnabled(active)
        except Exception:
            pass

        # The pending pattern is stale the moment the state flips, along with
        # any direction lines tracing it. Arming consumed it (it IS the armed
        # path now); stopping returns the plot to a single normal preview of
        # the current parameters rather than leaving it blank.
        self._clear_raster_overlay()
        if active:
            self._refresh_raster_scatter()
        else:
            self._render_preview(quiet=True)


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
