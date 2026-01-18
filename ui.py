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

        # --- Build plot display into placeholder widget "plot" ---
        self._init_plot()

        # --- Wire UI -> controller ---
        self._connect_ui_actions()

        # --- Wire controller -> UI ---
        self._connect_controller_signals()
        
        # --- Install flip controls --- 
        # This is for raster_gui2.ui legacy support; raster_gui3.ui has built-in checkboxes
        self._install_flip_controls()
        
        # Camera setup
        self._start_camera()

        # Flip settings from config (edit config.py: APP_CONFIG.camera.flip_x / flip_y)
        self._log(f"Image flips: flip_x={self._flip_x}, flip_y={self._flip_y} (set in config.py)")

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
        
        # Rotate Image if desired
        # k=1  -> 90 degrees counter-clockwise
        # k=-1 -> 90 degrees clockwise
        # k=2  -> 180 degrees
        frame = np.rot90(frame, k=-1)
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

        # Preserve old behavior: clicks add hull points (used by convex hull raster)
        self._hull_points.append((x, y))
        self.hull_scatter.setData([p[0] for p in self._hull_points], [p[1] for p in self._hull_points])

        # Mode handling
        if self._mode == "scale":
            self._scale_clicks.append((x, y))
            if len(self._scale_clicks) >= 2:
                self._finish_scale()
            return

        if self._mode == "calibrate":
            # Forward click to controller calibration collector
            self.controller.add_calibration_click(x, y)
            return

        # normal: no automatic move; user can hit "Move to Position"

    def _install_flip_controls(self) -> None:
        # If the UI file already provided the checkboxes, use them.
        # Otherwise, create them and put them in the status bar (legacy support).
        
        if not hasattr(self, "flip_x_checkbox"):
            self.flip_x_checkbox = QtWidgets.QCheckBox("Flip X")
            self.statusBar().addPermanentWidget(self.flip_x_checkbox)
            
        if not hasattr(self, "flip_y_checkbox"):
            self.flip_y_checkbox = QtWidgets.QCheckBox("Flip Y")
            self.statusBar().addPermanentWidget(self.flip_y_checkbox)

        # Sync state
        self.flip_x_checkbox.setChecked(bool(self._flip_x))
        self.flip_y_checkbox.setChecked(bool(self._flip_y))

        # Wire signals
        self.flip_x_checkbox.toggled.connect(self._set_flip_x)
        self.flip_y_checkbox.toggled.connect(self._set_flip_y)

    def _set_flip_x(self, checked: bool) -> None:
        self._flip_x = bool(checked)
        if hasattr(self, "vb"):
            self.vb.invertX(self._flip_x)

    def _set_flip_y(self, checked: bool) -> None:
        self._flip_y = bool(checked)
        if hasattr(self, "vb"):
            self.vb.invertY(self._flip_y)


    def _start_camera(self) -> None:
        from camera import UEyeCameraThread, UEyeConfig

        # Read camera settings from config.py if available
        if _config is not None and hasattr(_config, "APP_CONFIG"):
            cam = _config.APP_CONFIG.camera
            cfg = UEyeConfig(
                camera_id=cam.camera_id,
                width=cam.width,
                height=cam.height,
                exposure_ms=cam.exposure_ms_default,
                pixel_clock_mhz = cam.pixel_clock_mhz,
                use_freeze=cam.use_freeze,
                emit_rgb=cam.emit_rgb,
                roi_offset_x=cam.roi_offset_x,
                roi_offset_y=cam.roi_offset_y,
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
        self.camera_thread.start()

        # Wire exposure spinbox → camera thread setter
        self.exposurevalue.setValue(float(cfg.exposure_ms))
        self.exposurevalue.valueChanged.connect(lambda v: self.camera_thread.set_exposure_ms(float(v)))


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


    def _on_raster_state(self, active: bool) -> None:
        """
        Controller emits this when raster starts/stops/finishes.
        This is the authoritative source for whether the raster is "armed/running".
        """
        self._raster_active_ui = bool(active)

        # Optional: keep Start/Stop buttons sane (if present)
        if hasattr(self, "start_button"):
            self.start_button.setEnabled(not active)
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(active)

        self._update_step_mode_ui()
        self._log("Raster active." if active else "Raster inactive.")


    def _step_raster(self) -> None:
        """
        Advance exactly one raster point.
        If raster isn't armed yet, arm it in step mode first (continuous unchecked), then step.
        """
        # If user is in continuous mode, Step should be disabled anyway—guard for safety.
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

        # Exposure control (UI can also call camera setters later)
        # For now, emit status.
        self.exposurevalue.valueChanged.connect(lambda v: self._log(f"Exposure setpoint changed to {v} (wire to camera.py)"))

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
        # (We’ll add controller.set_target_bounds() soon)
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
        Convert motor position to a 0–100 progress bar value.
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
