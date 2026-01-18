from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, \
    QDoubleSpinBox, QComboBox, QGridLayout, QWidget, QVBoxLayout, QProgressBar, \
    QGraphicsPixmapItem, QLabel, QSlider, QMessageBox, QCheckBox, QSpinBox
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QImage, QTransform
import pyqtgraph as pg

import sys
import time
from datetime import datetime
import numpy as np

from PIL import Image as Image_pil

from scipy.spatial import Delaunay

from toolbox import *

import threading
import zmq
import json
import os

## Camera stuff
from pyueye import ueye

def start_server(app_worker, raster_controls):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:55535")

    def handle_request(request):
        try:
            data = json.loads(request)
            action = data['action']
            connection = data['connection']
            if action == 'PROGRAM_VALUE':
                setpoint_value = data['value']
                return SET_VALUE(connection, setpoint_value)
            elif action == 'CHECK_VALUE':
                current_value = GET_VALUE(connection)
                return json.dumps({"status": "SUCCESS", "value": current_value})
            else:
                return json.dumps({"status": "ERROR", "message": "Invalid action"})
        except Exception as e:
            print(str(e))
            return json.dumps({"status": "ERROR", "message": str(e)})

    def SET_VALUE(connection, setpoint_value):
        worker = app_worker
        ui = raster_controls
        timeout_sec = 60
        if connection == "laser_raster_x_coord":
            start= time.time()
            worker.raster_manager.moveX(float(setpoint_value))
            while time.time() - start < timeout_sec:
                if worker.raster_manager.device_a.taskComplete == True:
                    return json.dumps({"status": "SUCCESS"})
                time.sleep(0.1)
            print("Timeout error when moving motor")
            return json.dumps({"status": "ERROR"})
        
        elif connection == "laser_raster_y_coord":
            start= time.time()
            worker.raster_manager.moveY(float(setpoint_value))
            while time.time() - start < timeout_sec:
                if worker.raster_manager.device_b.taskComplete == True:
                    return json.dumps({"status": "SUCCESS"})
                time.sleep(0.1)
            print("Timeout error when moving motor")
            return json.dumps({"status": "ERROR"})
        
        elif connection == "arm_raster":
            print("Received command to arm")
            ui.worker.running = True
            ui.thread = QThread(parent=ui)
            ui.worker.moveToThread(ui.thread)
            ui.thread.finished.connect(ui.worker.stop)
            ui.thread.start()

        elif connection == "move_to_next":    
            ui.worker.manual_work()
            start= time.time()
            while time.time() - start < timeout_sec:
                    if worker.raster_manager.device_a.taskComplete and worker.raster_manager.device_b.taskComplete:
                        return json.dumps({"status": "SUCCESS"})
                    time.sleep(0.1)
            print("Timeout error when moving motor")
            return json.dumps({"status": "ERROR"})
        
        elif connection == "disarm_raster":
            print("Received command to disarm")
            ui.worker.running = False
            ui.worker.mpl_instance.needs_update = True
            ui.thread.stop()

    def GET_VALUE(connection):
        worker = app_worker
        if connection == "laser_raster_x_coord_monitor" or connection == "laser_raster_x_coord":
            val = worker.raster_manager.get_current_x()
        elif connection == "laser_raster_y_coord_monitor" or connection == "laser_raster_y_coord":
            val = worker.raster_manager.get_current_y()
        return val

    while True:
        request = socket.recv()
        response = handle_request(request)
        socket.send(response.encode())

class CameraThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

        self.counter = 0
        self.time_emitted = 0

        ## Camera initialization
        self.hcam = ueye.HIDS(0)
        self.ret = ueye.is_InitCamera(self.hcam, None)
   
        # set the color mode.
        self.ret = ueye.is_SetColorMode(self.hcam, ueye.IS_CM_BGR8_PACKED)

        # set the region of interest (Camera dependent).
        self.width = 1280
        self.height = 1024
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(self.width)
        rect_aoi.s32Height = ueye.int(self.height)
        ueye.is_AOI(self.hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))

        # allocate memory for live view.
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        self.bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
        self.ret = ueye.is_AllocImageMem(self.hcam, self.width, self.height, self.bitspixel,
                                    self.mem_ptr, self.mem_id)
                
        # set active memory region.
        self.ret = ueye.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)

        # continuous capture to memory.
        self.ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
        self.lineinc = self.width * int((self.bitspixel + 7) / 8)

        # set initial exposure
        time_exposure_ = 3
        time_exposure = ueye.double(time_exposure_)
        self.ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))

    def run(self):
        while self.running:
                # Capture a frame from the camera
            img = ueye.get_data(self.mem_ptr, self.width, self.height, self.bitspixel, self.lineinc, copy=True)

            # Turn it into something readable
            img = np.reshape(img, (self.height,self.width,3))

            #print("Before flip, first entry: ", img[0,0,:])

            img = np.flip(img, 1).copy()
            img = np.flip(img, 0).copy()
            
            #print("After flip, first entry: ", img[0,0,:])

            self.msleep(100)
            self.counter += 1

            self.new_frame.emit(img)

    def stop(self):
        self.running = False

class MplCanvas(QWidget):
    clicked = pyqtSignal(float, float)
    newScale = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # Load the pixel positions and don't scale
        self.pixel_positions = []
        self.to_scale = False

        self.plotWidget = pg.PlotWidget(background='w')
        self.marker = [1, 1]
        self.have_bounds = False
        self.styles = {"color": "b", "font-size": "28px"}
        self.plotWidget.setLabel("left", "y (mm)", **self.styles)
        self.plotWidget.setLabel("bottom", "x (mm)", **self.styles)
        self.plotWidget.setAutoVisible(y=True)
        self.plotWidget.showGrid(x=True, y=True, alpha=0.3)
        lay = QVBoxLayout(self)
        lay.addWidget(self.plotWidget)
        self.scatter = pg.ScatterPlotItem(size=10)
        self.line_path = pg.PlotCurveItem(size=10)
        self.scatter.setZValue(11)
        self.plotWidget.addItem(self.scatter)

        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#632222d1"))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#632222d1"))
        self.plotWidget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plotWidget.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plotWidget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        self.count = 0

        # set up initial scaling factor
        # for now, we will keep it constant
        self.scale = 0.002

        # set up the image item
        self.img = None

        self.hull = []
        self.xmin = 100
        self.xmax = -100
        self.ymin = 100
        self.ymax = -100
        self.hull_scatter = pg.ScatterPlotItem(size=10)
        self.hull_scatter.setZValue(10)
        self.plotWidget.addItem(self.hull_scatter)
        self.plotWidget.scene().sigMouseClicked.connect(self.on_click)       
    
        self.calibrated = False
        # self.calibration_scale = (1.0, 1.0)
        # self.calibration_offset = (0.0, 0.0)
        self.calibration_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        self.calibration_offset = np.array([0.0, 0.0])

        # Set up imaging thread
        self.cam_thread = CameraThread() 
        self.cam_thread.new_frame.connect(self.update_frame) 
        self.cam_thread.start()

        self.time_plotted = time.time()

        self.oldframe = 0

        self.pos_history= []
        self.path_history = pg.ScatterPlotItem(size=10)
        self.path_history.setZValue(19)
        self.plotWidget.addItem(self.path_history)

        # self.transform = QTransform()
        # self.transform.scale(-1,1)

    def mouseMoved(self, e):
        pos = e[0]
        if self.plotWidget.sceneBoundingRect().contains(pos):
            mousePoint = self.plotWidget.getPlotItem().vb.mapSceneToView(pos)
            mousecoord = "({:.4f}, {:.4f})".format(mousePoint.x(), mousePoint.y())
            self.plotWidget.setLabel("top", mousecoord, **self.styles)
            self.crosshair_v.setPos(mousePoint.x())
            self.crosshair_h.setPos(mousePoint.y())

    def update_plot(self, x, y):
        """
        Plots position history of the beam
        """
        self.pos_history.append((x, y))
        self.num_points = int(self.point_display_count.value())
        self.show_all = self.show_all_points_checkbox.isChecked()

        # Determine how many points to display
        if self.show_all:
            self.display_points = self.pos_history
        else:
            self.display_points = self.pos_history[-self.num_points:]
        
        spots = []
        
        for i, (px, py) in enumerate(self.display_points):
            if i == len(self.display_points) - 1:
                color = "#ff0000"
            elif i == len(self.display_points) - 2:
                color = "#ff9900ff"
            else:
                color = "#46FF2E"
            spots.append({'pos': (px, py), 'brush': pg.mkBrush(color)})

        self.path_history.setData(spots)
    
    def update_jog_display(self):
        if self.pos_history:
            x, y = self.pos_history[-1]
            self.path_history.setData([])
            self.update_plot(x, y)

    def update_frame(self, image_array):
        """
        Removes old image and adds new image to canvas
        """
        if image_array.ndim == 3 and image_array.shape[2] == 3:  # RGB
            h, w, c = image_array.shape
            q_image = QImage(image_array.data, w, h, 3 * w, QImage.Format_RGB888)
        elif image_array.ndim == 2:  # Grayscale image
            h, w = image_array.shape
            q_image = QImage(image_array.data, w, h, w, QImage.Format_Grayscale8)
        elif image_array.ndim == 3 and image_array.shape[2] == 4:  # RGBA
            h, w, c = image_array.shape
            q_image = QImage(image_array.data, w, h, 4 * w, QImage.Format_RGBA8888)

        # Convert the QImage to a QPixmap and set it as the label's pixmap
        pixmap = QPixmap.fromImage(q_image)

        oldimage = self.img

        if self.img is not None:
            # Remove the old image (if exists) from the plot
            self.plotWidget.getPlotItem().removeItem(self.img)

        # Add in the new frame
        self.img = QGraphicsPixmapItem(pixmap)
        self.img.setScale(self.scale) 
        self.img.setRotation(0*180)
        # self.img.setTransform(QTransform().scale(-1,-1).translate(-1 * self.scale * pixmap.height(), -1 * self.scale* pixmap.width()))
        # self.img.setOpacity(0.6)
        self.img.setZValue(1)

        if oldimage == self.img:
            print("Same frame again")

        self.plotWidget.addItem(self.img)
        # print("Time to update: ", time.time() - self.time_plotted)  
        # self.time_plotted = time.time()

    def record_scale_point(self, pixel_x, pixel_y):
        """
        Records a scale point where the user clicks on the canvas.
        """
        # Store the pixel positions as pairs
        self.pixel_positions.append((pixel_x, pixel_y))
        print(f"Recorded Pixel: ({pixel_x}, {pixel_y})")

        # If we have two scale points, calculate the transformation matrix
        if len(self.pixel_positions) == 2:
            self.calculate_scale()

    ## Comment out scling for now, needs debugging 

    def calculate_scale(self):
        """
        Calculate scaling for the image, based on user's input
        """
        (x1_pixel, y1_pixel), (x2_pixel, y2_pixel) = self.pixel_positions
        self.scale = self.scale/np.sqrt((x1_pixel - x2_pixel)**2 + (y1_pixel - y2_pixel)**2)
        self.newScale.emit(self.scale)

    def scaling(self):
        self.pixel_positions.clear()
        self.to_scale = True
        print("Click two points to scale...")
        
    def setscale(self, value):
        self.scale = value

    def setexposure(self, value):
        time_exposure = value
        time_exposure_ = ueye.double(time_exposure)
        ueye.is_Exposure(self.cam_thread.hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure_, ueye.sizeof(time_exposure_))

    def closeEvent(self, event):
        # Make sure to release the camera when the widget is closed
        ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
        ueye.is_ExitCamera(self.hcam)
        event.accept()

    def display_bounds(self, x1, y1, x2, y2):
        if self.have_bounds:
            self.plotWidget.removeItem(self.rect_item)
        self.rect_item = RectItem(QtCore.QRectF(x1, y1, x2-x1, y2-y1))
        self.plotWidget.addItem(self.rect_item)
        self.have_bounds = True

    def on_click(self, e):
        if not self.to_scale:
            pos = e.scenePos()
            mousePoint = self.plotWidget.getPlotItem().vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            self.hull_scatter.addPoints([x], [y], brush=pg.mkBrush("#c402cf"))
            self.hull.append([x, y])
            if x > self.xmax:
                self.xmax = x
            if y > self.ymax:
                self.ymax = y
            if x < self.xmin:
                self.xmin = x
            if y < self.ymin:
                self.ymin = y
            if x is None or y is None:
                return  # Ignore clicks outside axes
            self.clicked.emit(x, y)
            return
        else:
            pos = e.scenePos()
            mousePoint = self.plotWidget.getPlotItem().vb.mapSceneToView(pos)
            x = mousePoint.x()
            y = mousePoint.y()
            self.record_scale_point(x, y)
            if len(self.pixel_positions) >= 2:
                self.to_scale = False

    def plot_motor_bounds(self):
        pass


class RectItem(pg.GraphicsObject):
    def __init__(self, rect, parent=None):
        super().__init__(parent)
        self._rect = rect
        self.picture = QtGui.QPicture()
        self._generate_picture()

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self):
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen("#cc6600"))
        painter.setBrush(pg.mkBrush("#ebce191a"))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())

class Worker(QObject):
    finished = pyqtSignal()

    def __init__(self, mpl_instance, parent=None):
        QObject.__init__(self, parent=parent)

        self.mpl_instance = mpl_instance
        self.running = False

        self.log_data = []
        self.log_path = None

        global boundaries, xstep, ystep, saving_dir

        #device_a = KCube("27268551", name="A")
        device_a = KCube("27270471", name="A")
        #device_b = KCube("27268560", name="B")
        device_b = KCube("27270522", name="B")
             
        self.raster_manager = ArrayPatternRasterX(device_a, device_b, boundaries=boundaries, xstep=xstep, ystep=ystep)
        
        self.sleep_timer = 2

    def auto_work(self):
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(os.getcwd(), f"raster_log_{timestamp_str}.json")
        self.log_data = []

        print(f"[LOGGING] Started logging to: {self.log_path}")

        while self.running:

            time.sleep(self.sleep_timer)

            self.raster_manager.update_motors()
            last_x = self.raster_manager.get_current_x()
            last_y = self.raster_manager.get_current_y()

            self.log_data.append({
            "timestamp": time.time(),
            "x": last_x,
            "y": last_y
            })

            self.mpl_instance.marker[0] = last_x
            self.mpl_instance.marker[1] = last_y
            self.mpl_instance.needs_update = False
            self.mpl_instance.update_plot(last_x, last_y)

        # Save JSON to disk
        try:
            with open(self.log_path, 'w') as f:
                json.dump(self.log_data, f, indent=2)
            print(f"[LOGGING] Saved to {self.log_path}")
        except Exception as e:
            print(f"[LOGGING ERROR] Could not write to {self.log_path}: {e}")
                    
        self.finished.emit()

    def manual_work(self):
        if self.running:
            self.raster_manager.update_motors()
            last_x = self.raster_manager.get_current_x()
            last_y = self.raster_manager.get_current_y()
            self.mpl_instance.marker[0] = last_x
            self.mpl_instance.marker[1] = last_y
            self.mpl_instance.needs_update = False
            self.mpl_instance.update_plot(last_x, last_y)

    def change_raster_algorithm(self, ind):
        try:
            algo = {0: "Square Raster X", 1: "Square Raster Y", 2: "Spiral Raster", 3: "Convex Hull Raster"}
            print(f"Changed algorithm to {algo[ind]}.")

            device_a = self.raster_manager.device_a
            device_b = self.raster_manager.device_b
            boundaries = self.raster_manager.boundaries
            xstep = self.raster_manager.xstep_size
            ystep = self.raster_manager.ystep_size
            x_direction = self.raster_manager.x_direction
            y_direction = self.raster_manager.y_direction
            
            if algo[ind] == "Square Raster X":
                self.raster_manager = ArrayPatternRasterX(device_a, device_b, boundaries, xstep, ystep)
            elif algo[ind] == "Square Raster Y":
                self.raster_manager = ArrayPatternRasterY(device_a, device_b, boundaries, xstep, ystep)
            elif algo[ind] == "Spiral Raster":
                self.raster_manager = SpiralRaster(device_a, device_b, boundaries, radius, step, alpha, del_alpha)
            elif algo[ind] == "Convex Hull Raster":
                self.raster_manager = ConvexHullRaster(device_a, device_b, boundaries, xstep, ystep)
            else:
                raise RuntimeWarning
            
            self.raster_manager.x_direction = x_direction
            self.raster_manager.y_direction = y_direction
        
        except Exception as e:
            print("Error:", e)

    def stop(self):
        self.running = False
        
    def update_marker(self):
        last_x = self.raster_manager.get_current_x()
        last_y = self.raster_manager.get_current_y()
        self.mpl_instance.marker[0] = last_x
        self.mpl_instance.marker[1] = last_y

    def setsleep(self, value):
        self.sleep_timer = value

class CalibrationManager(QObject):
    calibration_updated = pyqtSignal(object)

    def __init__(self, canvas, raster_manager, UI):
        super().__init__()

        self.canvas = canvas
        self.raster_manager = raster_manager
        self.ui = UI

        # Initialize the pixel and motor positions
        self.pixel_positions = []
        self.motor_positions = []

        self.to_calibrate = False

        self.save_path = "calibration_data.json"

        self.canvas.clicked.connect(self.handle_click)

        self.calibration_updated.connect(self.raster_manager.set_calibration)
        self.calibration_updated.connect(self.canvas.plot_motor_bounds)
        self.calibration_updated.connect(self.ui.show_calibration)
        self.calibration_updated.connect(self.ui.calibration_done_popup)


    def record_calibration_point(self, pixel_x, pixel_y):
        """
        Records a calibration point where the user clicks on the canvas.
        It also retrieves the corresponding motor position.
        """
        motor_a = self.raster_manager.device_a.get_position()  # Get current motor A position
        motor_b = self.raster_manager.device_b.get_position()  # Get current motor B position

        # Store the pixel and motor positions as pairs
        self.pixel_positions.append((pixel_x, pixel_y))
        self.motor_positions.append((motor_a, motor_b))

        # Debug: Print the values
        print(f"Recorded Pixel: ({pixel_x}, {pixel_y}), Motor: ({motor_a}, {motor_b})")

        # If we have two calibration points, calculate the transformation matrix
        # if len(self.pixel_positions) == 2:
        #     self.calculate_calibration()
        if len(self.pixel_positions) == 3:
            self.calculate_calibration()

    def save_calibration(self):
        # calibration_data = {
        #     "scale_x": self.scale_x,
        #     "scale_y": self.scale_y,
        #     "offset_x": self.offset_x,
        #     "offset_y": self.offset_y
        # }
        calibration_data = {
                "calibration_matrix": self.calibration_matrix.tolist(),
                "calibration_offset": self.calibration_offset.tolist()
        }
        with open(self.save_path, "w") as file:
            json.dump(calibration_data, file)
        print(f"Calibration saved to {self.save_path}.")

    def load_calibration(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as file:
                calibration_data = json.load(file)
                # self.scale_x = calibration_data["scale_x"]
                # self.scale_y = calibration_data["scale_y"]
                # self.offset_x = calibration_data["offset_x"]
                # self.offset_y = calibration_data["offset_y"]
                self.calibration_matrix = np.array(calibration_data["calibration_matrix"])
                self.calibration_offset = np.array(calibration_data["calibration_offset"])

                # Apply these to the canvas
                # self.canvas.calibration_scale = (self.scale_x, self.scale_y)
                # self.canvas.calibration_offset = (self.offset_x, self.offset_y)
                self.canvas.calibration_matrix = self.calibration_matrix
                self.canvas.calibration_offset = self.calibration_offset
                self.canvas.calibrated = True
                
                # Also update the raster manager
                self.calibration_updated.emit(self)

            print(f"Calibration loaded from {self.save_path}.")
        else:
            print("No previous calibration found.")

    def calculate_calibration_old(self):
        """
        Calculates the transformation matrix based on the two calibration points.
        Assumes that self.pixel_positions and self.motor_positions each contain two points.
        """
        # Get the two points
        (x1_pixel, y1_pixel), (x2_pixel, y2_pixel) = self.pixel_positions
        (x1_motor, y1_motor), (x2_motor, y2_motor) = self.motor_positions

        # Compute deltas for pixels 
        delta_pixel_x = x2_pixel - x1_pixel
        delta_pixel_y = y2_pixel - y1_pixel

        # Compute deltas for motors
        delta_motor_x = x2_motor - x1_motor
        delta_motor_y = y2_motor - y1_motor

        # Calculate the scaling factors (pixel-to-motor conversion)
        scale_x = abs(delta_motor_x / delta_pixel_x)
        scale_y = abs(delta_motor_y / delta_pixel_y)

        # Ensure the sign is correct
        if delta_motor_x * delta_pixel_x < 0:
            scale_x *= -1
        if delta_motor_y * delta_pixel_y < 0:
            scale_y *= -1

        # Calculate the offsets (starting motor position for given pixel position)
        offset_x = x1_motor - scale_x * x1_pixel
        offset_y = y1_motor - scale_y * y1_pixel

        # Save the scaling and offset parameters for future use
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.offset_x = offset_x
        self.offset_y = offset_y

        # Debug: Print the calibration results
        print(f"Calibration Complete:")
        print(f"Scale X: {self.scale_x}, Scale Y: {self.scale_y}")
        print(f"Offset X: {self.offset_x}, Offset Y: {self.offset_y}")

        # self.canvas.calibration_scale = (scale_x, scale_y)
        # self.canvas.calibration_offset = (offset_x, offset_y)
        self.canvas.calibrated = True

        # ## Find the minimum and maximum motor positions
        # self.xposmax = self.raster_manager.device_x.GetMaxPosition()
        # self.xposmin = self.raster_manager.device_x.GetMinPosition()
        # self.yposmax = self.raster_manager.device_y.GetMaxPosition()
        # self.yposmin = self.raster_manager.device_y.GetMinPosition()
        
        # self.xpixmax = (self.xposmax - self.offset_x) / self.scale_x
        # self.xpixmin = (self.xpixmin - self.offset_x) / self.scale_x
        # self.ypixmax = (self.yposmax - self.offset_y) / self.scale_y
        # self.ypixmin = (self.ypixmin - self.offset_y) / self.scale_y

        self.calibration_updated.emit(self)
        self.save_calibration()

    def calculate_calibration(self):
        """
        Model: 
        (motor pos vec) = self.calibration_matrix @ (pixel pos vec) + self.calibration_offset
        """
        pixels_matrix = []
        for pt in self.pixel_positions:
            pixels_matrix.append([pt[0], pt[1], 1, 0, 0, 0])
            pixels_matrix.append([0, 0, 0, pt[0], pt[1], 1])
        pixels_matrix = np.array(pixels_matrix)
        motor_positions_flattened = np.array(self.motor_positions).flatten()
        
        affine_params, residuals, rank, s = np.linalg.lstsq(pixels_matrix, motor_positions_flattened)

        self.calibration_matrix = np.array([
            [affine_params[0], affine_params[1]],
            [affine_params[3], affine_params[4]]
        ])
        self.calibration_offset = np.array([affine_params[2], affine_params[5]])
        
        # TODO: Canvas
        self.canvas.calibrated = True

        ## Find the minimum and maximum motor positions
        # self.aposmax = self.raster_manager.device_a.GetMaxPosition()
        # self.aposmin = self.raster_manager.device_a.GetMinPosition()
        # self.bposmax = self.raster_manager.device_b.GetMaxPosition()
        # self.bposmin = self.raster_manager.device_b.GetMinPosition()

        # NOT finding max and min pix positions

        self.calibration_updated.emit(self)
        self.save_calibration()


    def calibration(self):
        self.pixel_positions.clear()
        self.motor_positions.clear()
        self.to_calibrate = True
        print("Click three points to calibrate...")

    def reset(self):
        # self.scale_x = 1
        # self.offset_x = 0
        # self.scale_y = 1
        # self.offset_y = 0
        self.calibration_matrix = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        self.calibration_offset = np.array([0.0, 0.0])

        # Debug: Print the calibration results
        print(f"Calibration Complete:")
        print(f"Calibration Matrix: {self.calibration_matrix}")
        print(f"Calibration Offset: {self.calibration_offset}")

        self.calibration_updated.emit(self)

    # def setmatrix_11(self, value):
    #     self.offset_y = value

    # def setmatrix_12(self, value):
    #     self.scale_y = value

    # def setmatrix_21(self, value):
    #     self.offset_x = value

    # def setmatrix_22(self, value):
    #     self.scale_x = value
    def set_cal_matrix(self, m11, m12, m21, m22):
        self.calibration_matrix = np.array([
            [m11, m12],
            [m21, m22]
        ])

    def set_cal_offset(self, b1, b2):
        self.calibration_offset = np.array([b1, b2])
    
    
    def setcalibration(self, ui):
    #     self.scale_x = ui.matrix_22value.value
    #     self.scale_y = ui.matrix_12.value
    #     self.offset_x = ui.matrix_21value.value
    #     self.offset_y = ui.matrix_11.value
        pass

    def handle_click(self, x, y):
        if not self.to_calibrate:
            return
        self.record_calibration_point(x, y)
        if len(self.pixel_positions) >= 3:
            self.to_calibrate = False

class UI(QMainWindow):
    stop_signal = pyqtSignal()
    exposureChanged = pyqtSignal(float)
    sleepSignal = pyqtSignal(float)
    scaleChanged = pyqtSignal(float)
    matrix_22Changed = pyqtSignal(float)
    matrix_12Changed = pyqtSignal(float)
    matrix_21Changed = pyqtSignal(float)
    matrix_11Changed = pyqtSignal(float)
    offset1Changed = pyqtSignal(float)
    offset2Changed = pyqtSignal(float)
    calibrateSignal = pyqtSignal()
    resetSignal = pyqtSignal()
    clearSignal = pyqtSignal()
    useoldSignal = pyqtSignal()
    scaleSignal = pyqtSignal()
    changeRasterAlgorithm = pyqtSignal(object)
    rasterPathSignal = pyqtSignal(list)

    def __init__(self):

        super().__init__()

        self.canvas = MplCanvas()
        self.canvas.scatter_path = None
        self.worker = Worker(self.canvas)
        self.calibration_manager = CalibrationManager(self.canvas, self.worker.raster_manager, self)
        self.calibration_manager.calibration_updated.connect(self.update_worker_calibration)

        
        self.canvas.newScale.connect(self.show_scale)
        self.canvas.clicked.connect(self.handle_click)

        self.log_file = None
        self.log_writer = None

        self.have_paths = False
        self.have_moves = False
        self.have_hull = False
        self.have_lines = False
        self.conv_hull = []

        self.position_timer = QTimer(self)
        self.position_timer.timeout.connect(self.update_position_display)
        self.position_timer.start(200)
        
        self.ui = uic.loadUi("raster_gui2.ui", self)
        self.canv_layout = self.findChild(QGridLayout, "gridLayout_2")
        self.canv_layout.addWidget(self.canvas, 430, 40, 721, 431)
        self.resize(2000, 2000)

        # Calibration Button
        self.calibrate = self.findChild(QPushButton, "calibrateButton")
        self.calibrate.clicked.connect(self.calibrateSignal.emit)
        self.calibrateSignal.connect(self.calibration_manager.calibration)
        self.calibrateSignal.connect(self.calibration_popup)

        # Reset Calibration Button
        self.reset = self.findChild(QPushButton, "resetButton")
        self.reset.clicked.connect(self.resetSignal.emit)
        self.resetSignal.connect(self.calibration_manager.reset)

        # Use Previous Calibration Button
        self.useold = self.findChild(QPushButton, "useold")
        self.useold.clicked.connect(self.useoldSignal.emit)
        self.useoldSignal.connect(self.calibration_manager.load_calibration)

        # Reset Plot Button in Automatic Tab
        self.clear = self.findChild(QPushButton, "clearAll")
        self.clear.clicked.connect(self.clearallraster)
        self.clear.clicked.connect(self.reset_hull)

        # Reset Plot Button in Manual Tab
        self.clear_manual = self.findChild(QPushButton, "clearAllManual")
        self.clear_manual.clicked.connect(self.clearallmanual)

        # Calibration Values
        self.matrix_11value = self.findChild(QDoubleSpinBox, "matrix_11")
        self.matrix_11value.setValue(1)
        self.matrix_11.valueChanged.connect(self.matrix_11Changed.emit)

        self.matrix_12value = self.findChild(QDoubleSpinBox, "matrix_12")
        self.matrix_12value.setValue(0)
        self.matrix_12.valueChanged.connect(self.matrix_12Changed.emit)

        self.matrix_21value = self.findChild(QDoubleSpinBox, "matrix_21")
        self.matrix_21value.setValue(0)
        self.matrix_21value.valueChanged.connect(self.matrix_21Changed.emit)

        self.matrix_22value = self.findChild(QDoubleSpinBox, "matrix_22")
        self.matrix_22value.setValue(1)
        self.matrix_22value.valueChanged.connect(self.matrix_22Changed.emit)

        self.offset1value = self.findChild(QDoubleSpinBox, "offset_a")
        self.offset1value.setValue(0)
        self.offset1value.valueChanged.connect(self.offset1Changed.emit)

        self.offset2value = self.findChild(QDoubleSpinBox, "offset_b")
        self.offset2value.setValue(0)
        self.offset2value.valueChanged.connect(self.offset2Changed.emit)
    
        # Image Scaler
        self.scaler = self.findChild(QDoubleSpinBox, "scaleImage")
        self.scaler.setValue(0.002)
        self.scaler.valueChanged.connect(self.scaleChanged.emit)
        self.scaleChanged.connect(self.canvas.setscale)

        # Exposure slider
        self.exposure = self.findChild(QDoubleSpinBox, "exposurevalue") 
        self.exposure.setValue(3)
        self.exposure.valueChanged.connect(self.exposureChanged.emit)
        self.exposureChanged.connect(self.canvas.setexposure)

        # Set the sleep timer
        self.sleep_value = self.findChild(QDoubleSpinBox, "sleepTimer")
        self.sleep_value.setValue(2)
        self.sleep_value.valueChanged.connect(self.sleepSignal.emit)
        self.sleepSignal.connect(self.worker.setsleep)


        # Set the scale
        self.scaleButton = self.findChild(QPushButton, "scaleButton")
        self.scaleButton.clicked.connect(self.scaleSignal.emit)
        self.scaleSignal.connect(self.canvas.scaling)

        # Bounds for square raster
        self.x_low_spinbox = self.findChild(QDoubleSpinBox, "xlow")
        self.y_low_spinbox = self.findChild(QDoubleSpinBox, "ylow")
        self.x_high_spinbox = self.findChild(QDoubleSpinBox, "xhigh")
        self.y_high_spinbox = self.findChild(QDoubleSpinBox, "yhigh")
        self.x_low_spinbox.valueChanged.connect(self.update_x_min)
        self.y_low_spinbox.valueChanged.connect(self.update_y_min)
        self.x_high_spinbox.valueChanged.connect(self.update_x_max)
        self.y_high_spinbox.valueChanged.connect(self.update_y_max)
        self.show_limits = self.findChild(QPushButton, "bound_button")
        self.show_limits.clicked.connect(self.display_limit)

        # Parameters for spiral raster
        self.radius = self.findChild(QDoubleSpinBox, "radius_spiral")
        self.step = self.findChild(QDoubleSpinBox, "step_spiral")
        self.delalpha = self.findChild(QDoubleSpinBox, "angle_spiral")
        self.delalpha_step = self.findChild(QDoubleSpinBox, "ang_change")
        self.radius.valueChanged.connect(self.update_r)
        self.step.valueChanged.connect(self.update_st)
        self.delalpha.valueChanged.connect(self.update_delalph)
        self.delalpha_step.valueChanged.connect(self.update_delalph_st)

        # Change raster algorithm
        self.dropbox = self.findChild(QComboBox, "alg_choice")
        self.dropbox.currentIndexChanged.connect(self.changeRasterAlgorithm.emit)
        self.changeRasterAlgorithm.connect(self.worker.change_raster_algorithm)

        # Raster control and parameters
        self.preview_button = self.findChild(QPushButton, "path_button")
        self.start_button = self.findChild(QPushButton, "start_button")
        self.stop_button = self.findChild(QPushButton, "stop_button")
        self.save_button = self.findChild(QPushButton, "save_button")
        self.preview_button.clicked.connect(self.preview_raster)
        self.start_button.clicked.connect(self.start_raster)
        self.stop_button.clicked.connect(self.stop_raster)
        self.save_button.clicked.connect(self.save_raster)
        self.xstep = self.findChild(QDoubleSpinBox, "xstep")
        self.ystep = self.findChild(QDoubleSpinBox, "ystep")
        self.xstep.valueChanged.connect(self.update_raster_step_x)
        self.ystep.valueChanged.connect(self.update_raster_step_y)

        # Homing motors
        self.soft_homeX_button = self.findChild(QPushButton, "homeX_3")
        self.soft_homeY_button = self.findChild(QPushButton, "homeY_3")
        self.soft_homeX_button.clicked.connect(self.soft_home_motorX)
        self.soft_homeY_button.clicked.connect(self.soft_home_motorY)

        self.hard_homeX_button = self.findChild(QPushButton, "homeX_4")
        self.hard_homeY_button = self.findChild(QPushButton, "homeY_4")
        self.hard_homeX_button.clicked.connect(self.hard_home_motorX)
        self.hard_homeY_button.clicked.connect(self.hard_home_motorY)

        # Jogging motors
        self.dx = self.findChild(QDoubleSpinBox, "dx_button")
        self.dy = self.findChild(QDoubleSpinBox, "dy_button")
        self.up_button = self.findChild(QPushButton, "jog_up_button_3")
        self.down_button = self.findChild(QPushButton, "jog_down_button_3")
        self.left_button = self.findChild(QPushButton, "jog_left_button_3")
        self.right_button = self.findChild(QPushButton, "jog_right_button_3")
        self.up_button.clicked.connect(self.jog_up)
        self.down_button.clicked.connect(self.jog_down)
        self.left_button.clicked.connect(self.jog_left)
        self.right_button.clicked.connect(self.jog_right)

        # Moving to set position
        self.x = self.findChild(QDoubleSpinBox, "x")
        self.y = self.findChild(QDoubleSpinBox, "y")
        self.move_to_button = self.findChild(QPushButton, "move_to_pos")
        self.preview_move_button = self.findChild(QPushButton, "preview_pos")
        self.move_to_button.clicked.connect(self.manual_move)
        self.preview_move_button.clicked.connect(self.preview_move)

        # Determine how many points to display
        self.canvas.show_all_points_checkbox = self.findChild(QCheckBox, "show_all_points_checkbox")
        self.canvas.point_display_count = self.findChild(QDoubleSpinBox, "point_display_count")
        self.canvas.show_all_points_checkbox.stateChanged.connect(self.canvas.update_jog_display)
        self.canvas.point_display_count.valueChanged.connect(self.canvas.update_jog_display)

        # Backlash correction
        self.backlash_x = self.findChild(QDoubleSpinBox, "x_backlash")
        self.backlash_y = self.findChild(QDoubleSpinBox, "y_backlash")
        self.backlash_x.valueChanged.connect(self.update_backlash_x)
        self.backlash_y.valueChanged.connect(self.update_backlash_y)

        # Position of the motors
        self.motor_x_progress = self.findChild(QProgressBar, "progress_motor_x_pos")
        self.motor_y_progress = self.findChild(QProgressBar, "progress_motor_y_pos")
        self.motor_x_label = self.findChild(QLabel, "motor_x_pos")
        self.motor_y_label = self.findChild(QLabel, "motor_y_pos")
        self.pixel_x_label = self.findChild(QLabel, "pixel_x_pos")
        self.pixel_y_label = self.findChild(QLabel, "pixel_y_pos")

        self.motor_x_progress.setRange(0, 100)
        self.motor_y_progress.setRange(0, 100)

        # Rastering direction preview
        self.canvas.show_raster_direction_checkbox = self.findChild(QCheckBox, "show_direction_checkbox")
        self.canvas.show_raster_direction_checkbox.stateChanged.connect(self.preview_raster_direction)
        
        self.update_backlash_x()
        self.update_backlash_y()

        self.show()
        time.sleep(0.5)
        self.startup_popup()
        self.initial_position_popup()

    def update_worker_calibration(self, calibration_manager):
        if hasattr(self.worker, 'raster_manager'):
            self.worker.raster_manager.calibration_matrix = calibration_manager.calibration_matrix
            self.worker.raster_manager.calibration_offset = calibration_manager.calibration_offset
            print("[UI] Worker raster_manager calibration updated")

    def clearallraster(self):
        # Cler all rastering points
        self.canvas.scatter.setData([])
        self.canvas.hull.clear()
        self.canvas.hull_scatter.setData([])
        # self.canvas.pos_history.clear()
        # self.canvas.path_history.setData([])
        
        if self.have_paths and hasattr(self.worker.mpl_instance, 'scatter_path'):
            self.worker.mpl_instance.scatter_path.setData([])
            if self.have_lines:
                for line in self.canvas.plotWidget.raster_path_lines:
                    self.canvas.plotWidget.removeItem(line)
                self.canvas.plotWidget.raster_path_lines.clear()
        # if self.have_paths:
        #     self.worker.mpl_instance.moving_path.clear()
        #     self.have_paths = False

    def clearallmanual(self):
        # Clear all manual move points
        # self.canvas.scatter.setData([])
        self.canvas.hull.clear()
        self.canvas.hull_scatter.setData([])
        self.canvas.pos_history.clear()
        self.canvas.path_history.setData([])
        
        if self.have_paths and hasattr(self.worker.mpl_instance, 'scatter_path'):
            self.have_paths = True
            if self.have_moves:
                self.worker.mpl_instance.moving_path.clear()
            # self.worker.mpl_instance.scatter_path.setData([])
        elif self.have_moves:
            self.worker.mpl_instance.moving_path.clear()
            self.have_paths = False

    def update_position_display(self):
        # if self.canvas.calibrated:
            # calibration_matrix = self.canvas.calibration_matrix
            # calibration_offset = self.canvas.calibration_offset

            # x_pix = self.worker.raster_manager.get_current_x()
            # y_pix = self.worker.raster_manager.get_current_y()
            # pix = np.array([x_pix, y_pix])

            # self.pixel_x_label.setText(f"{x_pix:.4f} px")
            # self.pixel_y_label.setText(f"{y_pix:.4f} px")
            
            # # x_mm = offset_x + scale_x * x_pix
            # # y_mm = offset_y + scale_y * y_pix
            # mm = calibration_matrix @ pix + calibration_offset
            # x_mm = mm[0]
            # y_mm = mm[1]

            x_mm = self.worker.raster_manager.device_a.get_position()
            y_mm = self.worker.raster_manager.device_b.get_position()

            self.motor_x_label.setText(f"{x_mm:.4f} mm")
            self.motor_y_label.setText(f"{y_mm:.4f} mm")

            x_percent = int(100*min(max(x_mm / 12.0, 0.0), 1.0))
            y_percent = int(100*min(max(y_mm / 12.0, 0.0), 1.0))

            self.motor_x_progress.setValue(x_percent)
            self.motor_y_progress.setValue(y_percent)
        # else:
        #     x_mm = self.worker.raster_manager.get_current_x()
        #     y_mm = self.worker.raster_manager.get_current_y()

        #     self.motor_x_label.setText(f"{x_mm:.4f}")
        #     self.motor_y_label.setText(f"{y_mm:.4f}")

        #     x_percent = int(100*min(max(x_mm / 12.0, 0.0), 1.0))
        #     y_percent = int(100*min(max(y_mm / 12.0, 0.0), 1.0))

        #     self.motor_x_progress.setValue(x_percent)
        #     self.motor_y_progress.setValue(y_percent)

        #     self.pixel_x_label.setText(f"----")
        #     self.pixel_y_label.setText(f"----")

    def startup_popup(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Welcome!")
        msg.setText("Warning: motors are not calibrated")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def initial_position_popup(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("Raw Motor position")
        msg.setText("The motors are at ({:.4f}, {:.4f})".format(self.worker.raster_manager.device_a.get_position(), self.worker.raster_manager.device_b.get_position()))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def calibration_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Calibration Message")
        msg.setText("Your next three on screen clicks will be recorded for calibration.\nPlease move laser to three arbitrary points, and click on laser location each time.")
        msg.setStandardButtons(QMessageBox.Ok)
        close = msg.exec_()

    def calibration_done_popup(self):
        msg = QMessageBox()
        msg.setWindowTitle("Calibration Complete")
        msg.setText("To reset calibration, hit Reset")
        msg.setStandardButtons(QMessageBox.Ok)
        close = msg.exec_()

    def show_calibration(self, calibration_manager):
        self.matrix_11value.setValue(calibration_manager.calibration_matrix[0][0])
        self.matrix_12value.setValue(calibration_manager.calibration_matrix[0][1])
        self.matrix_21value.setValue(calibration_manager.calibration_matrix[1][0])
        self.matrix_22value.setValue(calibration_manager.calibration_matrix[1][1])
        self.offset1value.setValue(calibration_manager.calibration_offset[0])
        self.offset2value.setValue(calibration_manager.calibration_offset[1])

    def show_scale(self, val): 
        self.scaler.setValue(val)

    def get_worker(self):
        return self.worker
    
    def handle_click(self, x, y):
        self.x.setValue(x)
        self.y.setValue(y)

    def reset_hull(self):
        if self.have_hull:
            self.canvas.hull = []
            self.canvas.hull_scatter.setData([])
            self.canvas.convexpath.clear()
            self.canvas.xmin = 100
            self.canvas.xmax = -100
            self.canvas.ymin = 100
            self.canvas.ymax = -100
            self.worker.raster_manager.rasterpath = [[],[]]
            self.have_hull = False

    def display_limit(self):
        global boundaries
        x1 = boundaries[0]
        x2 = boundaries[1]
        y1 = boundaries[2]
        y2 = boundaries[3]
        print("Current bounds: X {:.4f} - {:.4f} mm, Y {:.4f} - {:.4f} mm".format(x1, x2, y1, y2))
        self.canvas.display_bounds(x1, y1, x2, y2)
    
    def preview_raster(self):

        #Clear the previous preview
        if self.have_paths and hasattr(self.worker.mpl_instance, 'scatter_path'):
            print("trying to clear points")
            if self.have_lines:
                for line in self.canvas.plotWidget.raster_path_lines:
                    self.canvas.plotWidget.removeItem(line)
                self.canvas.plotWidget.raster_path_lines.clear()

        print("Previewing the path")
        #if self.have_paths:
            #self.worker.mpl_instance.scatter_path.setData([])
        
        path = self.worker.raster_manager.preview_path(self)

        # print("Path previewed, x scale = ", self.worker.raster_manager.scale_x)
        # print("Path previewed, y scale = ", self.worker.raster_manager.scale_y)
        # print("Path previewed, x offset = ", self.worker.raster_manager.offset_x)
        # print("Path previewed, y offset = ", self.worker.raster_manager.offset_y)

        # ChatGPT:
        if self.worker.mpl_instance.scatter_path is None:
            self.worker.mpl_instance.scatter_path = pg.ScatterPlotItem(size=10)
            self.worker.mpl_instance.plotWidget.addItem(self.worker.mpl_instance.scatter_path)

        self.worker.mpl_instance.scatter_path.setData(pos=[(x, y) for x, y in zip(path[0], path[1])])
        self.worker.mpl_instance.scatter_path.setBrush("#5eb9ddff")
        self.worker.mpl_instance.scatter_path.setPen("#000000d1")
        self.worker.mpl_instance.scatter_path.setOpacity(1)
        self.worker.mpl_instance.scatter_path.setZValue(3)

        # self.worker.mpl_instance.scatter_path = pg.ScatterPlotItem(size=10)
        # self.worker.mpl_instance.plotWidget.addItem(self.worker.mpl_instance.scatter_path)
        # self.worker.mpl_instance.scatter_path.addPoints(path[0], path[1])
        # self.worker.mpl_instance.scatter_path.setBrush("#5eb9ddff")
        # self.worker.mpl_instance.scatter_path.setPen("#000000d1")
        # self.worker.mpl_instance.scatter_path.setOpacity(1)
        # self.worker.mpl_instance.scatter_path.setZValue(3)
        self.have_paths = True
        print("Finished")

    def preview_raster_direction(self):
        path = self.worker.raster_manager.preview_path(self)
        
        if self.have_lines and not self.canvas.show_raster_direction_checkbox.isChecked():
            for line in self.canvas.plotWidget.raster_path_lines:
                self.canvas.plotWidget.removeItem(line)
            self.canvas.plotWidget.raster_path_lines.clear() 
            self.have_lines = False
        
        elif self.canvas.show_raster_direction_checkbox.isChecked(): 
            self.worker.mpl_instance.plotWidget.raster_path_lines = []
            for i in range(len(path[0])-1):
                x0, y0 = path[0][i], path[1][i]
                x1, y1 = path[0][i+1], path[1][i+1]
                line = pg.PlotCurveItem(x=[x0,x1], y=[y0,y1], pen=pg.mkPen(color="#4970F1", width=8))
                line.setZValue(11)
                self.worker.mpl_instance.plotWidget.addItem(line)
                self.worker.mpl_instance.plotWidget.raster_path_lines.append(line)
            self.have_lines = True
       
    def start_raster(self):
        print("Received command to start")
        self.worker.running = True
        self.make_threaded_worker()
        self.start_button.setEnabled(False)
        
    def stop_raster(self):
        self.stop_signal.emit()
        self.worker.mpl_instance.needs_update = True
        print("Received command to stop")
        self.start_button.setEnabled(True)
        self.worker.stop()
        
    def save_raster(self):
        if self.worker.running:
            print("The rastering is still running")
        else:
          self.canvas.scatter.clear()
    
    def soft_home_motorX(self):
        reply = QMessageBox.question(self, "Confirm homing",
                                     "Are you sure you want to soft home motor X?",
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            try:
                print("Received command to soft home X")
                self.worker.raster_manager.soft_homeX()
                last_x = self.worker.raster_manager.get_current_x()
                last_y = self.worker.raster_manager.get_current_y()
                # self.worker.mpl_instance.marker[0] = last_x
                # self.worker.mpl_instance.marker[1] = last_y
                # self.worker.mpl_instance.update_plot()
            except AttributeError:
                return False
  
    def soft_home_motorY(self):
        reply = QMessageBox.question(self, "Confirm homing",
                                     "Are you sure you want to soft home motor Y?",
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            try:
                print("Received command to soft home Y")
                self.worker.raster_manager.soft_homeY()
                last_x = self.worker.raster_manager.get_current_x()
                last_y = self.worker.raster_manager.get_current_y()
                # self.worker.mpl_instance.marker[0] = last_x
                # self.worker.mpl_instance.marker[1] = last_y
                # self.worker.mpl_instance.update_plot()
            except AttributeError:
                return False
            
    def hard_home_motorX(self):
        reply = QMessageBox.question(self, "Confirm homing",
                                     "Are you sure you want to hard home motor X?",
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            try:
                print("Received command to hard home X")
                self.worker.raster_manager.hard_homeX()
                last_x = self.worker.raster_manager.get_current_x()
                last_y = self.worker.raster_manager.get_current_y()
                # self.worker.mpl_instance.marker[0] = last_x
                # self.worker.mpl_instance.marker[1] = last_y
                # self.worker.mpl_instance.update_plot()
            except AttributeError:
                return False
  
    def hard_home_motorY(self):
        reply = QMessageBox.question(self, "Confirm homing",
                                     "Are you sure you want to hard home motor Y?",
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            try:
                print("Received command to hard home Y")
                self.worker.raster_manager.hard_homeY()
                last_x = self.worker.raster_manager.get_current_x()
                last_y = self.worker.raster_manager.get_current_y()
                # self.worker.mpl_instance.marker[0] = last_x
                # self.worker.mpl_instance.marker[1] = last_y
                # self.worker.mpl_instance.update_plot()
            except AttributeError:
                return False
    
    def jog_up(self):
        try:
            print("Received command to jog up.")
            y = self.worker.raster_manager.get_current_y()
            dy = abs(self.dy.value())
            print("Jogging Y from {:.4f} to {:.4f}".format(y, y + dy))
            self.worker.raster_manager.moveY(y + dy)
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot(last_x, last_y)
        except AttributeError:
            return False

    def jog_down(self):
        try:
            print("Received command to jog down.")
            y = self.worker.raster_manager.get_current_y()
            dy = abs(self.dy.value())
            print("Jogging Y from {:.4f} to {:.4f}".format(y, y - dy))
            self.worker.raster_manager.moveY(y - dy)
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot(last_x, last_y)
        except AttributeError:
            return False

    def jog_left(self):
        try:
            print("Received command to jog left.")
            x = self.worker.raster_manager.get_current_x()
            dx = abs(self.dx.value())
            print("Jogging X from {:.4f} to {:.4f}".format(x, x - dx))
            self.worker.raster_manager.moveX(x - dx)
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot(last_x, last_y)
        except AttributeError:
            return False

    def jog_right(self):
        try:
            print("Received command to jog right.")
            x = self.worker.raster_manager.get_current_x()
            dx = abs(self.dx.value())
            print("Jogging X from {:.4f} to {:.4f}".format(x, x + dx))
            self.worker.raster_manager.moveX(x + dx)
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot(last_x, last_y)
        except AttributeError:
            return False

    def manual_move(self):
        # Warn if the motors are not calibrated
        if not self.canvas.calibrated:
            QMessageBox.critical(self, "Calibration Error",
                                    "Error: motors are not calibrated.")
        # Get new motor coords
        new_pix = np.array([self.x.value(), self.y.value()])
        calibration_matrix = self.worker.raster_manager.calibration_matrix
        calibration_offset = self.worker.raster_manager.calibration_offset
        new_mm = calibration_matrix @ new_pix + calibration_offset
        # Double-check to prevent accidental moves
        reply = QMessageBox.question(self, "Confirm Manual Move",
                                     "The motors are at ({:.4f}, {:.4f}). Do you want to move to ({:.4f}, {:.4f})?".format(
                                      self.worker.raster_manager.device_a.get_position(), self.worker.raster_manager.device_b.get_position(),
                                      new_mm[0],  new_mm[1]   
                                     ),
                                     QMessageBox.Ok | QMessageBox.Cancel)
        if reply == QMessageBox.Ok:
            print("Received command to move to ({:.4f}, {:.4f})".format(self.x.value(),  self.y.value()))
            try:
                self.worker.raster_manager.moveTo(self.x.value(), self.y.value())
                last_x = self.worker.raster_manager.get_current_x()
                last_y = self.worker.raster_manager.get_current_y()
                self.worker.mpl_instance.marker[0] = last_x
                self.worker.mpl_instance.marker[1] = last_y
                self.worker.mpl_instance.update_plot(last_x, last_y)
            except AttributeError:
                pass
        
    def preview_move(self):
        print("Previewing move to {:.4f}, {:.4f}".format(self.x.value(),  self.y.value()))
        if self.have_moves:
            self.worker.mpl_instance.moving_path.clear()
        move = self.worker.raster_manager.preview_move(self.x.value(), self.y.value())
        self.worker.mpl_instance.moving_path = pg.ScatterPlotItem(size=10)
        self.worker.mpl_instance.plotWidget.addItem(self.worker.mpl_instance.moving_path)
        self.worker.mpl_instance.moving_path.addPoints(move[0], move[1])
        self.worker.mpl_instance.moving_path.setBrush("#5eb9ddff")
        self.worker.mpl_instance.moving_path.setPen("#000000d1")
        self.worker.mpl_instance.moving_path.setOpacity(1)
        self.worker.mpl_instance.moving_path.setZValue(15)
        self.have_moves = True
        self.have_paths = True
    
    def update_x_min(self):
        global boundaries
        try:
            self.worker.do_raster = False
            v_old = self.worker.raster_manager.xlim_lo
            self.worker.raster_manager.update_x_low(self.x_low_spinbox.value())
            print("Changed x_min from {:.4f} to {:.4f}".format(v_old, self.x_low_spinbox.value()))
            x1 = self.x_low_spinbox.value()
            x2 = self.x_high_spinbox.value()
            y1 = self.y_low_spinbox.value()
            y2 = self.y_high_spinbox.value()
            boundaries = (x1, x2, y1, y2)
        except AttributeError:
            return False

    def update_x_max(self):
        global boundaries
        try:
            self.worker.do_raster = False
            v_old = self.worker.raster_manager.xlim_hi
            self.worker.raster_manager.update_x_high(self.x_high_spinbox.value())
            print("Changed x_max from {:.4f} to {:.4f}".format(v_old, self.x_high_spinbox.value()))
            x1 = self.x_low_spinbox.value()
            x2 = self.x_high_spinbox.value()
            y1 = self.y_low_spinbox.value()
            y2 = self.y_high_spinbox.value()
            boundaries = (x1, x2, y1, y2)
        except AttributeError:
            return False

    def update_y_min(self):
        global boundaries
        try:
            self.worker.do_raster = False
            v_old = self.worker.raster_manager.ylim_lo
            self.worker.raster_manager.update_y_low(self.y_low_spinbox.value())
            print("Changed y_min from {:.4f} to {:.4f}".format(v_old, self.y_low_spinbox.value()))
            x1 = self.x_low_spinbox.value()
            x2 = self.x_high_spinbox.value()
            y1 = self.y_low_spinbox.value()
            y2 = self.y_high_spinbox.value()
            boundaries = (x1, x2, y1, y2)
        except AttributeError:
            return False

    def update_y_max(self):
        global boundaries
        try:
            self.worker.do_raster = False
            v_old = self.worker.raster_manager.ylim_hi
            self.worker.raster_manager.update_y_high(self.y_high_spinbox.value())
            print("Changed y_max from {:.4f} to {:.4f}".format(v_old, self.y_high_spinbox.value()))
            x1 = self.x_low_spinbox.value()
            x2 = self.x_high_spinbox.value()
            y1 = self.y_low_spinbox.value()
            y2 = self.y_high_spinbox.value()
            boundaries = (x1, x2, y1, y2)
        except AttributeError:
            return False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

    def update_raster_step_x(self):
        step_size_old_x = self.worker.raster_manager.xstep_size
        self.worker.raster_manager.update_step_size_x(self.xstep.value())
        step_size_new_x = self.worker.raster_manager.xstep_size
        print("Updated raster step size x from {:.4f} to {:.4f}".format(step_size_old_x, step_size_new_x))\
            
    def update_raster_step_y(self):
        step_size_old_y = self.worker.raster_manager.ystep_size
        self.worker.raster_manager.update_step_size_y(self.ystep.value())
        step_size_new_y = self.worker.raster_manager.ystep_size
        print("Updated raster step size y from {:.4f} to {:.4f}".format(step_size_old_y, step_size_new_y))
        
    def update_r(self):
        radius_old = self.worker.raster_manager.spiral_rad
        self.worker.raster_manager.update_radius(self.radius.value())
        radius_new = self.worker.raster_manager.spiral_rad
        print("Updated radius from {:.4f} to {:.4f}".format(radius_old, radius_new))
        
    def update_st(self):
        step_old = self.worker.raster_manager.spiral_step
        self.worker.raster_manager.update_spiral_step(self.step.value())
        step_new = self.worker.raster_manager.spiral_step
        print("Updated spiral step from {:.4f} to {:.4f}".format(step_old, step_new))
        
    def update_delalph(self):
        dela_old = self.worker.raster_manager.angle_step
        self.worker.raster_manager.update_angle_step(self.delalpha.value())
        dela_new = self.worker.raster_manager.angle_step
        print("Updated angle step from {:.4f} to {:.4f}".format(dela_old, dela_new))
        
    def update_delalph_st(self):
        dela_step_old = self.worker.raster_manager.angle_step_change
        self.worker.raster_manager.update_angle_step_change(self.delalpha_step.value())
        dela_step_new = self.worker.raster_manager.angle_step_change
        print("Updated angle step change from {:.4f} to {:.4f}".format(dela_step_old, dela_step_new))
        
    def update_backlash_x(self):
        xback = Decimal(float(self.backlash_x.value()))
        self.worker.raster_manager.update_backlash_on_x(xback)
        new_backlash_x = self.backlash_x.value()
        print("Updated backlash on x to {:.3f} mm".format(new_backlash_x))
        
    def update_backlash_y(self):
        yback = Decimal(float(self.backlash_y.value()))
        self.worker.raster_manager.update_backlash_on_y(yback)
        new_backlash_y = self.backlash_y.value()
        print("Updated backlash on y to {:.3f} mm".format(new_backlash_y))
    
    def make_threaded_worker(self):

        self.worker.raster_manager.calibration_matrix = self.calibration_manager.calibration_matrix
        self.worker.raster_manager.calibration_offset = self.calibration_manager.calibration_offset

        self.thread = QThread(parent=self)
        self.stop_signal.connect(self.worker.stop)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.auto_work)
        self.thread.finished.connect(self.worker.stop)

        # print("Starting Auto Raster, x scale = ", self.worker.raster_manager.scale_x)
        # print("Starting Auto Raster, y scale = ", self.worker.raster_manager.scale_y)
        # print("Starting Auto Raster, x offset = ", self.worker.raster_manager.offset_x)
        # print("Starting Auto Raster, y offset = ", self.worker.raster_manager.offset_y)
        print("Starting Auto Raster")
        
        self.thread.start()

if __name__ == '__main__':
    global boundaries, xstep, ystep, saving_dir
    
    boundaries = (0.0, 12.0, 0.0, 12.0)
    xstep = 0.05
    ystep = 0.05
    radius = 0.05 
    step = 0.008
    alpha = 0.1
    del_alpha = 0.05
    
    app = QApplication(sys.argv)    
    UIWindow = UI()

    server_thread = threading.Thread(target=start_server, args=(UIWindow.get_worker(),UIWindow,))
    server_thread.daemon = True
    server_thread.start()

    sys.exit(app.exec_())
    
