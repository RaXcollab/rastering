from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, \
    QDoubleSpinBox, QComboBox, QGridLayout, QWidget, QVBoxLayout, \
    QGraphicsPixmapItem, QLabel, QSlider
from PyQt5 import QtCore, QtGui, uic
from PyQt5.QtCore import QObject, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPixmap, QImage
import pyqtgraph as pg

import sys
import time
import numpy as np

from PIL import Image as Image_pil

from scipy.spatial import Delaunay

from toolbox import *

import threading
import zmq
import json

## Camera stuff
from pyueye import ueye
import numpy as np
import cv2
import launch_camera

def start_server(app_worker):
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
                SET_VALUE(connection, setpoint_value)
                return json.dumps({"status": "SUCCESS"})
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
        if connection == "laser_raster_x_coord":
            worker.raster_manager.moveX(float(setpoint_value))
        elif connection == "laser_raster_y_coord":
            worker.raster_manager.moveY(float(setpoint_value))

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

class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)

        self.cam = ueye

        # initialize the camera.
        self.hcam = ueye.HIDS(0)
        self.cam.is_InitCamera(hcam, None)
        print(f"initCamera returns {ret}")

        # set the color mode.
        self.cam.is_SetColorMode(hcam, ueye.IS_CM_BGR8_PACKED)
        print(f"SetColorMode IS_CM_BGR8_PACKED returns {ret}")

        # set the region of interest (Camera dependent).
        width = 1280
        height = 1024
        rect_aoi = ueye.IS_RECT()
        rect_aoi.s32X = ueye.int(0)
        rect_aoi.s32Y = ueye.int(0)
        rect_aoi.s32Width = ueye.int(width)
        rect_aoi.s32Height = ueye.int(height)
        ueye.is_AOI(hcam, ueye.IS_AOI_IMAGE_SET_AOI, rect_aoi, ueye.sizeof(rect_aoi))
        print(f"AOI IS_AOI_IMAGE_SET_AOI returns {ret}")

        # allocate memory for live view.
        mem_ptr = ueye.c_mem_p()
        mem_id = ueye.int()
        bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
        self.cam.is_AllocImageMem(hcam, width, height, bitspixel,
                                    mem_ptr, mem_id)
        print(f"AllocImageMem returns {ret}")
        
        # set active memory region.
        self.cam.is_SetImageMem(hcam, mem_ptr, mem_id)
        print(f"SetImageMem returns {ret}")

        # continuous capture to memory.
        self.cam.is_CaptureVideo(hcam, ueye.IS_DONT_WAIT)
        print(f"CaptureVideo returns {ret}")
        
        # get data from camera and display (press the q key to exit out of live capture).
        lineinc = width * int((bitspixel + 7) / 8)

        # set exposure
        time_exposure_ = 3
        time_exposure = ueye.double(time_exposure_)
        self.cam.is_Exposure(hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))

        # Set up a timer to update the camera feed at regular intervals
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30ms (roughly 30 FPS)

        self.capture = ueye.get_data(mem_ptr, width, height, bitspixel, lineinc, copy=True)




class MplCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

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
        self.plotWidget.addItem(self.scatter)

        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#632222d1"))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen("#632222d1"))
        self.plotWidget.addItem(self.crosshair_v, ignoreBounds=True)
        self.plotWidget.addItem(self.crosshair_h, ignoreBounds=True)
        self.proxy = pg.SignalProxy(self.plotWidget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)

        ## Camera initialization
        self.cam = ueye
        self.hcam = self.cam.HIDS(0)
        self.cam.is_InitCamera(self.hcam, None)
   
        # set the color mode.
        self.cam.is_SetColorMode(self.hcam, ueye.IS_CM_BGR8_PACKED)

        # set the region of interest (Camera dependent).
        self.width = 1280
        self.height = 1024
        rect_aoi = self.cam.IS_RECT()
        rect_aoi.s32X = self.cam.int(0)
        rect_aoi.s32Y = self.cam.int(0)
        rect_aoi.s32Width = self.cam.int(self.width)
        rect_aoi.s32Height = self.cam.int(self.height)
        self.cam.is_AOI(self.hcam, self.cam.IS_AOI_IMAGE_SET_AOI, rect_aoi, self.cam.sizeof(rect_aoi))

        # allocate memory for live view.
        self.mem_ptr = self.cam.c_mem_p()
        self.mem_id = self.cam.int()
        self.bitspixel = 24 # for colormode = IS_CM_BGR8_PACKED
        self.cam.is_AllocImageMem(self.hcam, self.width, self.height, self.bitspixel,
                                    self.mem_ptr, self.mem_id)
                
        # set active memory region.
        self.cam.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)

        # continuous capture to memory.
        self.cam.is_CaptureVideo(self.hcam, self.cam.IS_DONT_WAIT)
        self.lineinc = self.width * int((self.bitspixel + 7) / 8)

        # set initial exposure
        time_exposure_ = 3
        time_exposure = self.cam.double(time_exposure_)
        self.cam.is_Exposure(self.hcam, self.cam.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, self.cam.sizeof(time_exposure))

        # Set up a timer to update the camera feed at regular intervals
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100) ## ms between each photo

        # set up initial scaling factor
        self.scale = 0.002

        # set up the image item
        self.img = None

        #q_image = launch_camera.main()

        #image_from_png = Image_pil.open("shot.png")
        #image_array = np.array(image_from_png)

        # pixmap = QPixmap("shot.png")
        # pixmap = QPixmap.fromImage(q_image)
        # self.img = QGraphicsPixmapItem(pixmap)
        # self.img.setOpacity(0.6)
        # self.plotWidget.addItem(self.img)
        # self.img.setScale(0.002*self.img.scale())

        self.hull = []
        self.xmin = 100
        self.xmax = -100
        self.ymin = 100
        self.ymax = -100
        self.hull_scatter = pg.ScatterPlotItem(size=10)
        self.plotWidget.addItem(self.hull_scatter)
        self.plotWidget.scene().sigMouseClicked.connect(self.on_click)       

    def mouseMoved(self, e):
        pos = e[0]
        if self.plotWidget.sceneBoundingRect().contains(pos):
            mousePoint = self.plotWidget.getPlotItem().vb.mapSceneToView(pos)
            mousecoord = "({:.4f}, {:.4f})".format(mousePoint.x(), mousePoint.y())
            self.plotWidget.setLabel("top", mousecoord, **self.styles)
            self.crosshair_v.setPos(mousePoint.x())
            self.crosshair_h.setPos(mousePoint.y())

    def update_plot(self):
        self.plotWidget.getPlotItem().listDataItems()[0].setBrush("#7894f2")
        self.plotWidget.getPlotItem().listDataItems()[0].setPen("#540808")
        self.scatter.addPoints([self.marker[0]], [self.marker[1]], brush=pg.mkBrush("#ff0000"))

    def update_frame(self):

        # Capture a frame from the camera
        img = self.cam.get_data(self.mem_ptr, self.width, self.height, self.bitspixel, self.lineinc, copy=True)

        # Turn it into something readable
        img = np.reshape(img, (self.height,self.width,3))
        image_array = img
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

        if self.img is not None:
            # Remove the old image (if exists) from the plot
            self.plotWidget.scene().removeItem(self.img)

        # Add in the new frame
        self.img = QGraphicsPixmapItem(pixmap)
        self.img.setOpacity(0.6)
        self.img.setScale(self.scale*self.img.scale())
        self.plotWidget.addItem(self.img)
        
    def setscale(self, value):
        self.scale = value

    def setexposure(self, value):
        time_exposure = value
        time_exposure_ = self.cam.double(time_exposure)
        self.cam.is_Exposure(self.hcam, self.cam.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure_, self.cam.sizeof(time_exposure_))

    def closeEvent(self, event):
        # Make sure to release the camera when the widget is closed
        self.cam.is_StopLiveVideo(self.hcam, self.cam.IS_FORCE_VIDEO_STOP)
        self.cam.is_ExitCamera(self.hcam)
        event.accept()

    def display_bounds(self, x1, y1, x2, y2):
        if self.have_bounds:
            self.plotWidget.removeItem(self.rect_item)
        self.rect_item = RectItem(QtCore.QRectF(x1, y1, x2-x1, y2-y1))
        self.plotWidget.addItem(self.rect_item)
        self.have_bounds = True

    def on_click(self, e):
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

        global boundaries, xstep, ystep, saving_dir

        device_x = KCube("27268551", name="X")
        device_y = KCube("27268560", name="Y")
        
        self.raster_manager = ArrayPatternRasterX(device_x, device_y, boundaries=boundaries, xstep=xstep, ystep=ystep)
        
    def do_work(self):
        while self.running:
            time.sleep(0.5)
            self.raster_manager.update_motors()
            last_x = self.raster_manager.get_current_x()
            last_y = self.raster_manager.get_current_y()
            self.mpl_instance.marker[0] = last_x
            self.mpl_instance.marker[1] = last_y
            self.mpl_instance.needs_update = False
            self.mpl_instance.update_plot()
                    
        self.finished.emit()

    def stop(self):
        self.running = False
        
    def update_marker(self):
        last_x = self.raster_manager.get_current_x()
        last_y = self.raster_manager.get_current_y()
        self.mpl_instance.marker[0] = last_x
        self.mpl_instance.marker[1] = last_y


class UI(QMainWindow):
    stop_signal = pyqtSignal()
    exposureChanged = pyqtSignal(float)
    scaleChanged = pyqtSignal(float)

    def __init__(self, Worker):

        super().__init__()

        self.canvas = MplCanvas()
        self.worker = Worker(self.canvas)
        self.have_paths = False
        self.have_moves = False
        self.have_hull = False
        self.conv_hull = []
        
        self.ui = uic.loadUi("raster_gui.ui", self)
        self.canv_layout = self.findChild(QGridLayout, "gridLayout_2")
        self.canv_layout.addWidget(self.canvas, 430, 40, 721, 431)
        self.resize(2000, 2000)

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
        self.dropbox.currentIndexChanged.connect(self.change_raster_algorithm)

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
        self.homeX_button = self.findChild(QPushButton, "homeX")
        self.homeY_button = self.findChild(QPushButton, "homeY")
        self.homeX_button.clicked.connect(self.home_motorX)
        self.homeY_button.clicked.connect(self.home_motorY)

        # Jogging motors
        self.dx = self.findChild(QDoubleSpinBox, "dx_button")
        self.dy = self.findChild(QDoubleSpinBox, "dy_button")
        self.up_button = self.findChild(QPushButton, "jog_up_button")
        self.down_button = self.findChild(QPushButton, "jog_down_button")
        self.left_button = self.findChild(QPushButton, "jog_left_button")
        self.right_button = self.findChild(QPushButton, "jog_right_button")
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

        # Backlash correction
        self.backlash_x = self.findChild(QDoubleSpinBox, "x_backlash")
        self.backlash_y = self.findChild(QDoubleSpinBox, "y_backlash")
        self.backlash_x.valueChanged.connect(self.update_backlash_x)
        self.backlash_y.valueChanged.connect(self.update_backlash_y)

        # Convex hull algorithm
        self.convexhull = self.findChild(QPushButton, "convex_hull")
        self.resetconvexhull = self.findChild(QPushButton, "convex_hull_reset")
        self.convexhull.clicked.connect(self.define_hull)
        self.resetconvexhull.clicked.connect(self.reset_hull)
        
        self.show()

    def get_worker(self):
        return self.worker
    
    def define_hull(self, hull):
        hull = self.canvas.hull
        self.have_hull = True
        if not isinstance(hull, Delaunay):
            self.conv_hull = Delaunay(hull)
        print(f"Convex hull defined by {len(self.canvas.hull)} the points:")
        path = []
        x = np.arange(self.canvas.xmin, self.canvas.xmax, self.xstep.value())
        y = np.arange(self.canvas.ymin, self.canvas.ymax, self.ystep.value())
        for i in range(len(x)):
            for j in range(len(y)):
                path.append([x[i], y[j]])
        self.raster = [[], []]
        for pt in path:
            if self.in_hull(pt):
                self.raster[0].append(pt[0])
                self.raster[1].append(pt[1])
        self.worker.raster_manager.rasterpath = self.raster
        self.canvas.convexpath = self.hull_scatter = pg.ScatterPlotItem(size=10)
        self.canvas.convexpath.addPoints(self.raster[0], self.raster[1], brush=pg.mkBrush("#dd68e3"))
        self.canvas.plotWidget.addItem(self.canvas.convexpath)

    def reset_hull(self):
        if self.have_hull:
            self.canvas.hull = []
            self.canvas.hull_scatter.clear()
            self.canvas.convexpath.clear()
            self.canvas.xmin = 100
            self.canvas.xmax = -100
            self.canvas.ymin = 100
            self.canvas.ymax = -100
            self.worker.raster_manager.rasterpath = [[],[]]
            self.have_hull = False

    def in_hull(self, pts):
        return self.conv_hull.find_simplex(pts) >= 0


    def display_limit(self):
        global boundaries
        x1 = boundaries[0]
        x2 = boundaries[1]
        y1 = boundaries[2]
        y2 = boundaries[3]
        print("Current bounds: X {:.4f} - {:.4f} mm, Y {:.4f} - {:.4f} mm".format(x1, x2, y1, y2))
        self.canvas.display_bounds(x1, y1, x2, y2)
    
    def preview_raster(self):
        print("Previewing the path")
        if self.have_paths:
            self.worker.mpl_instance.scatter_path.clear()
        path = self.worker.raster_manager.preview_path()
        self.worker.mpl_instance.scatter_path = pg.ScatterPlotItem(size=10)
        self.worker.mpl_instance.plotWidget.addItem(self.worker.mpl_instance.scatter_path)
        self.worker.mpl_instance.scatter_path.addPoints(path[0], path[1])
        self.worker.mpl_instance.scatter_path.setBrush("#9e93935c")
        self.worker.mpl_instance.scatter_path.setPen("#000000d1")
        self.worker.mpl_instance.scatter_path.setOpacity(0.5)
        self.have_paths = True
        print("Finished")
          
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
    
    def home_motorX(self):
        try:
            print("Received command to home X")
            self.worker.raster_manager.homeX()
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot()
        except AttributeError:
            return False
  
    def home_motorY(self):
        try:
            print("Received command to home Y")
            self.worker.raster_manager.homeY()
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot()
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
            self.worker.mpl_instance.update_plot()
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
            self.worker.mpl_instance.update_plot()
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
            self.worker.mpl_instance.update_plot()
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
            self.worker.mpl_instance.update_plot()
        except AttributeError:
            return False

    def manual_move(self):
        try:
            print("Received command to move to {:.4f}, {:.4f}".format(self.x.value(),  self.y.value()))
            self.worker.raster_manager.moveTo(self.x.value(), self.y.value())
            last_x = self.worker.raster_manager.get_current_x()
            last_y = self.worker.raster_manager.get_current_y()
            self.worker.mpl_instance.marker[0] = last_x
            self.worker.mpl_instance.marker[1] = last_y
            self.worker.mpl_instance.update_plot()
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
        self.worker.mpl_instance.moving_path.setBrush("#9e93935c")
        self.worker.mpl_instance.moving_path.setPen("#000000d1")
        self.worker.mpl_instance.moving_path.setOpacity(0.5)
    
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
        
    def change_raster_algorithm(self):
        try:
            algo = self.dropbox.currentText()
            print(f"Changed algorithm to {algo}.")

            device_x = self.worker.raster_manager.device_x
            device_y = self.worker.raster_manager.device_y
            boundaries = self.worker.raster_manager.boundaries
            xstep = self.worker.raster_manager.xstep_size
            ystep = self.worker.raster_manager.ystep_size
            x_direction = self.worker.raster_manager.x_direction
            y_direction = self.worker.raster_manager.y_direction
        
            if algo == "Square Raster X":
                self.worker.raster_manager = ArrayPatternRasterX(device_x, device_y, boundaries, xstep, ystep)
            elif algo == "Square Raster Y":
                self.worker.raster_manager = ArrayPatternRasterY(device_x, device_y, boundaries, xstep, ystep)
            elif algo == "Spiral Raster":
                self.worker.raster_manager = SpiralRaster(device_x, device_y, boundaries, radius, step, alpha, del_alpha)
            elif algo == "Convex Hull Raster":
                self.worker.raster_manager = ConvexHullRaster(device_x, device_y, boundaries, xstep, ystep)
            else:
                raise RuntimeWarning
        
            self.worker.raster_manager.x_direction = x_direction
            self.worker.raster_manager.y_direction = y_direction
        
        except AttributeError:
            pass
        
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
        self.thread = QThread(parent=self)
        self.stop_signal.connect(self.worker.stop)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.do_work)
        self.thread.finished.connect(self.worker.stop)
        
        self.thread.start()


if __name__ == '__main__':
    global boundaries, xstep, ystep, saving_dir
    
    boundaries = (0.0, 3.0, 0.0, 3.0)
    xstep = 0.01
    ystep = 0.01
    radius = 0.05 
    step = 0.008
    alpha = 0.1
    del_alpha = 0.05
    
    app = QApplication(sys.argv)    
    UIWindow = UI(Worker)

    server_thread = threading.Thread(target=start_server, args=(UIWindow.get_worker(),))
    server_thread.daemon = True
    server_thread.start()

    sys.exit(app.exec_())
    