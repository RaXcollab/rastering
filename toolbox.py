import os
import time
import clr
from PyQt5.QtCore import pyqtSignal

clr.AddReference('System')
import socket
from csv import writer
from datetime import datetime, timedelta
from math import cos, sin, pi

clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\Thorlabs.MotionControl.GenericMotorCLI.dll")
clr.AddReference(r"C:\Program Files\Thorlabs\Kinesis\ThorLabs.MotionControl.KCube.DCServoCLI.dll")

from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.DCServoCLI import *

from System import Decimal

from System import Action, UInt64

import numpy as np

from scipy.spatial import Delaunay

import pyqtgraph as pg

class Motor:
    """ Abstract motor class """

    def __init__(self, serial_no, name):
        self.name = name
        self.taskComplete = True
        self.taskID = 0
        self.available = False
        self.device = None
        self.is_locked = False

    def move_to(self, xpos):
        raise NotImplementedError

    def disconnect(self):
        raise NotImplementedError

    def get_position(self):
        raise NotImplementedError

    def is_available(self):
        return self.available

    def lock(self):
        self.is_locked = True

    def unlock(self):
        self.is_locked = False

    def home(self):
        return self.get_position()

    def is_moving(self):
        return not self.taskComplete


class KCube(Motor):
    def __init__(self, serial_no, name):
        super().__init__(serial_no, name)
        self.last_known_pos = 0

        try:
            DeviceManagerCLI.BuildDeviceList()
            device_list = DeviceManagerCLI.GetDeviceList()
            print("devices", device_list)
            for device in device_list:
                print(device)
                if device == serial_no:
                    self.available = True

            if self.available:
                self.device = KCubeDCServo.CreateKCubeDCServo(serial_no)
                self.device.Connect(serial_no)
                time.sleep(0.25)
                self.device.StartPolling(100)
                time.sleep(0.25)
                self.device.EnableDevice()
                time.sleep(0.25)

                if not self.device.IsSettingsInitialized():
                    self.device.WaitForSettingsInitialized(10000)
                    assert self.device.IsSettingsInitialized() is True

                m_config = self.device.LoadMotorConfiguration(serial_no,
                                                              DeviceConfiguration.DeviceSettingsUseOptionType.UseFileSettings)
                m_config.DeviceSettingsName = "Z912"
                m_config.UpdateCurrentConfiguration()
                self.device.SetSettings(self.device.MotorDeviceSettings, True, False)

                self.last_known_pos = Decimal.ToSingle(self.device.Position)

        except Exception as e:
            print(e)

    def move_to(self, pos):
        if self.is_locked:
            print(f"Motor {self.name} is locked.")
            return self.get_position()

        if not self.taskComplete:
            print("Motor is in motion.")
            return self.get_position()

        self.last_known_pos = self.get_position()

        if abs(self.get_position() - pos) < 0.0001:
            return self.get_position()

        d_x = Decimal(float(pos))
        self.taskComplete = False
        try:
            self.taskID = self.device.MoveTo(d_x, Action[UInt64](self.task_complete_callback))
            while not self.taskComplete:
                time.sleep(0.1)
                print("Moving {}...".format(self.name))
        except:
            print("Outside of motor range")
            self.taskComplete = True

        self.last_known_pos = self.get_position()

        return self.get_position()

    def task_complete_callback(self, taskID):
        if self.taskID == taskID and taskID > 0:
            self.taskComplete = True

    def disconnect(self):
        if self.available:
            self.device.Disconnect()

    def get_position(self):
        if not self.is_moving():
            return Decimal.ToSingle(self.device.Position)
        else:
            return self.last_known_pos

    def get_velocity(self):
        velocity_parameters = self.device.GetVelocityParams()
        return Decimal.ToSingle(velocity_parameters.MaxVelocity)

    def get_acceleration(self):
        velocity_parameters = self.device.GetVelocityParams()
        return Decimal.ToSingle(velocity_parameters.Acceleration)

    def home(self):
        if self.is_locked:
            print(f"Motor {self.name} is locked.")
            return self.get_position()

        self.taskComplete = False
        print(f"Homing {self.name} motor.")
        self.taskID = self.device.Home(Action[UInt64](self.task_complete_callback))
        while not self.taskComplete:
            time.sleep(0.5)
            print("Homing in progress...")
        self.move_to(6)
        print("{} motor homed. {} position = {:.4f}".format(self.name, self.name, self.get_position()))
        self.last_known_pos = self.get_position()
        return self.get_position()
    
    def set_backlash(self, val):
        self.device.SetBacklash(val)
        return self.device.GetBacklash()


class SimulatedMotor(Motor):
    def __init__(self, serial_no, name):
        super().__init__(serial_no, name)
        self.device = 6
        self.available = True

    def move_to(self, pos):
        if self.is_locked:
            print(f"Motor {self.name} is locked.")
            return self.get_position()
        self.device = pos
        return pos

    def disconnect(self):
        pass

    def get_position(self):
        return self.device    


class RasterManager:

    motorextrema = pyqtSignal(float,float,float,float)

    def __init__(self, device_x: Motor, device_y: Motor, boundaries=(0.0, 12.0, 0.0, 12.0), xstep=0.05, ystep=0.05, 
                 radius=0.05, step=0.008, alpha=0.1, del_alpha=0.05):

        self.device_x = device_x
        self.device_y = device_y
        self.x_available = device_x.available
        self.y_available = device_y.available

        self.taskID_x = 0
        self.taskComplete_x = False
        self.taskID_y = 0
        self.taskComplete_y = False

        self.xstep_size = xstep
        self.ystep_size = ystep
        self.spiral_rad = radius
        self.spiral_step = step
        self.angle_step = alpha
        self.angle_step_change = del_alpha

        self.boundaries = boundaries
        self.xlim_lo = boundaries[0]
        self.xlim_hi = boundaries[1]
        self.ylim_lo = boundaries[2]
        self.ylim_hi = boundaries[3]
        
        self.rasterpath = [[], []]

        self.locked = False

        ## Intiialize the calibration
        self.scale_x = 1
        self.offset_x = 0
        self.scale_y = 1
        self.offset_y = 0

        self.ind = 0

    def set_calibration(self, calibration_manager):
        self.scale_x = calibration_manager.scale_x
        print("Scale x received as ", calibration_manager.scale_x)
        self.scale_y = calibration_manager.scale_y
        print("Scale y received as ", calibration_manager.scale_y)
        self.offset_x = calibration_manager.offset_x
        print("Offset x received as ", calibration_manager.offset_x)
        self.offset_y = calibration_manager.offset_y
        print("Offset y received as ", calibration_manager.offset_y)

    def moveTo(self, xpos, ypos):
        
        ## Scale the x and y positions
        xpos_ = self.scale_x * xpos + self.offset_x
        ypos_ = self.scale_y * ypos + self.offset_y

        print("Moving to motor position ({:.4f}, {:.4f})".format(xpos_, ypos_))

        if self.x_available and self.y_available and self.pos_allowed(xpos, ypos):
            new_x = self.device_x.move_to(xpos_)
            new_y = self.device_y.move_to(ypos_)
        else:
            print("Can not move to ({:.4f},{:.4f}) because it is out of bound.".format(xpos, ypos))
            new_x = self.get_current_x()
            new_y = self.get_current_y()
        return new_x, new_y

    def homeX(self):
        if self.x_available:
            self.device_x.home()
        return self.get_current_x()
    
    def homeY(self):
        if self.y_available:
            self.device_y.home()
        return self.get_current_y()

    def moveX(self, xpos):

        ## Scale the x position
        xpos_ = self.scale_x * xpos + self.offset_x

        ypos = self.get_current_y()
        if self.x_available and self.y_available and self.pos_allowed(xpos, ypos):
            self.device_x.move_to(xpos_)
        else:
            print("Can not move to ({:.4f},{:.4f}) because it is out of bound.".format(xpos, ypos))

    def moveY(self, ypos):
        
        ## Scale the y position
        ypos_ = self.scale_y * ypos + self.offset_y

        xpos = self.get_current_x()
        if self.x_available and self.y_available and self.pos_allowed(xpos, ypos):
            self.device_y.move_to(ypos_)
        else:
            print("Can not move to ({:.4f},{:.4f}) because it is out of bound.".format(xpos, ypos))

    def disconnect(self):
        self.device_x.disconnect()
        self.device_y.disconnect()

    def pos_allowed(self, xpos, ypos):
        print("x bounds: ", self.xlim_lo, self.xlim_hi)
        print("y bounds: ", self.ylim_lo, self.ylim_hi)
        return (self.xlim_lo <= xpos < self.xlim_hi) and (self.ylim_lo <= ypos < self.ylim_hi)
    
    def update_motors(self):
        xpix = self.get_current_x()
        ypix = self.get_current_y()
        xpix_next = self.path[0][self.ind]
        ypix_next = self.path[1][self.ind]
        print("Current position is ({:.4f}, {:.4f}); Moving to ({:.4f}, {:.4f})".format(xpix, ypix, xpix_next, ypix_next))
        self.moveTo(xpix_next, ypix_next)
        self.ind += 1

    def get_current_x(self):
        return (self.device_x.get_position() - self.offset_x) / self.scale_x

    def get_current_y(self):
        return (self.device_y.get_position() - self.offset_y) / self.scale_y

    def update_x_low(self, v):
        self.xlim_lo = v
        self.boundaries = (self.xlim_lo, self.xlim_hi, self.ylim_lo, self.ylim_hi)

    def update_x_high(self, v):
        self.xlim_hi = v
        self.boundaries = (self.xlim_lo, self.xlim_hi, self.ylim_lo, self.ylim_hi)

    def update_y_low(self, v):
        self.ylim_lo = v
        self.boundaries = (self.xlim_lo, self.xlim_hi, self.ylim_lo, self.ylim_hi)

    def update_y_high(self, v):
        self.ylim_hi = v
        self.boundaries = (self.xlim_lo, self.xlim_hi, self.ylim_lo, self.ylim_hi)

    def lock(self):
        self.device_x.lock()
        self.device_y.lock()

    def unlock(self):
        self.device_x.unlock()
        self.device_y.unlock()

    def moving_in_progress(self):
        return self.device_x.is_moving() or self.device_y.is_moving()

    def update_step_size_x(self, v):
        self.xstep_size = v

    def update_step_size_y(self, v):
        self.ystep_size = v
        
    def update_radius(self, v):
        self.spiral_rad = v
    
    def update_spiral_step(self, v):
        self.spiral_step = v
        
    def update_angle_step(self, v):
        self.angle_step = v
        
    def update_angle_step_change(self, v):
        self.angle_step_change = v

    def set_boundaries(self, boundaires):
        self.boundaries = boundaires
        self.xlim_lo = boundaires[0]
        self.xlim_hi = boundaires[1]
        self.ylim_lo = boundaires[2]
        self.ylim_hi = boundaires[3]
        
    def update_backlash_on_x(self, v):
        return self.device_x.set_backlash(v)
    
    def update_backlash_on_y(self, v):
        return self.device_y.set_backlash(v)
    
    def preview_move(self, x, y):
        move = [[x], [y]]
        return move



class ArrayPatternRasterX(RasterManager):
    """ Raster in a square array pattern along X-axis """

    def __init__(self, device_x: Motor, device_y: Motor, boundaries=(0.0, 12.0, 0.0, 12.0),
                 xstep=0.01, ystep=0.01):
        super().__init__(device_x, device_y, boundaries, xstep, ystep)
        self.x_direction = 1
        self.y_direction = 1

    def preview_path(self, ui):
        xpix_next = self.boundaries[0]
        ypix_next = self.boundaries[3]
        path = [[], []]
        x_direction = 1
        y_direction = 1
        for i in range(1000):
            ymove = False
            xpix = xpix_next
            ypix = ypix_next
            if not self.xlim_lo <= xpix + x_direction * self.xstep_size <= self.xlim_hi:
                x_direction *= -1
                ymove = True
            if ymove and not self.ylim_lo <= ypix + y_direction * self.ystep_size <= self.ylim_hi:
                y_direction *= -1
            if ymove:
                ypix_next = ypix + self.ystep_size * y_direction
                xpix_next = xpix
            else:
                ypix_next = ypix
                xpix_next = xpix + self.xstep_size * x_direction
            path[0].append(xpix_next)
            path[1].append(ypix_next)
        self.path = path
        self.ind = 0
        return path
       
        
class ArrayPatternRasterY(RasterManager):
    """ Raster in a square array pattern along Y-axis """

    def __init__(self, device_x: Motor, device_y: Motor, boundaries=(0.0, 12.0, 0.0, 12.0),     
                 xstep=0.01, ystep=0.01):
        super().__init__(device_x, device_y, boundaries, xstep, ystep)
        self.x_direction = 1
        self.y_direction = 1

    def preview_path(self, ui):
        xpix_next = self.boundaries[0]
        ypix_next = self.boundaries[3]
        path = [[], []]
        x_direction = 1
        y_direction = 1
        for i in range(1000):
            xmove = False
            xpix = xpix_next
            ypix = ypix_next
            if not self.ylim_lo <= ypix + y_direction * self.ystep_size <= self.ylim_hi:
                y_direction *= -1
                xmove = True
            if xmove and not self.xlim_lo <= xpix + x_direction * self.xstep_size <= self.xlim_hi:
                x_direction *= -1
            if xmove:
                xpix_next = xpix + self.xstep_size * x_direction
                ypix_next = ypix
            else:
                xpix_next = xpix
                ypix_next = ypix + self.ystep_size * y_direction
            path[0].append(xpix_next)
            path[1].append(ypix_next)
        self.path = path
        self.ind = 0
        return path


class SpiralRaster(RasterManager):
    """ Raster in a spiral pattern """
    
    def __init__(self, device_x: Motor, device_y: Motor, boundaries=(0.0, 12.0, 0.0, 12.0), 
                 radius=0.05, step=0.008, del_alpha=0.1, del_alpha_step=0.05):
        super().__init__(device_x, device_y, boundaries, radius, step, del_alpha, del_alpha_step)
        xpos = self.get_current_x()
        ypos = self.get_current_y()
        self.alpha = 0
                
    def preview_path(self, ui):
        path = [[],[]]
        r = self.spiral_rad
        st = self.spiral_step
        alpha = 0
        del_alpha = self.angle_step
        self.xorigin = self.get_current_x()
        self.yorigin = self.get_current_y
        xpix_next = self.xorigin
        ypix_next = self.yorigin
        while r >= 0:
            if alpha < 2 * pi:
                xpix = xpix_next
                ypix = ypix_next
                xpix_next = self.xorigin + r * cos(alpha)
                ypix_next = self.yorigin + r * sin(alpha)
                alpha += del_alpha
                path[0].append(xpix_next)
                path[1].append(ypix_next)
            else:
                alpha = 0
                del_alpha += self.angle_step_change
                r -= st
                xpix = xpix_next
                ypix = ypix_next
                xpix_next = self.xorigin + r * cos(alpha)
                ypix_next = self.yorigin + r * sin(alpha)
                path[0].append(xpix_next)
                path[1].append(ypix_next)
        self.path = path
        self.ind = 0
        return path


class ConvexHullRaster(RasterManager):
    """ Raster in a within convex hull bounds """

    def __init__(self, device_x: Motor, device_y: Motor, boundaries=(0.0, 12.0, 0.0, 12.0),
                 xstep=0.01, ystep=0.01):
        super().__init__(device_x, device_y, boundaries, xstep, ystep)
        self.index = 0
    
    def preview_path(self, ui):
        self.have_hull = True
        if not isinstance(ui.canvas.hull, Delaunay):
            self.conv_hull = Delaunay(ui.canvas.hull)
        print(f"Convex hull defined by {len(ui.canvas.hull)} points:")
        raster = []
        x = np.arange(ui.canvas.xmin, ui.canvas.xmax, ui.xstep.value())
        y = np.arange(ui.canvas.ymin, ui.canvas.ymax, ui.ystep.value())
        for i in range(len(x)):
            for j in range(len(y)):
                raster.append([x[i], y[j]])
        path = [[], []]
        for pt in raster:
            if self.in_hull(pt):
                path[0].append(pt[0])
                path[1].append(pt[1])
        self.path = path
        self.ind = 0
        return path

    def in_hull(self, pts):
        return self.conv_hull.find_simplex(pts) >= 0


if __name__ == "__main__":
    serial_no_y = "27268551"
    serial_no_x = "27268560"
    device_x = KCube(serial_no_x, "X")
    device_y = KCube(serial_no_y, "Y")

    if device_x.is_available():
        print(f"device {device_x.name} ({serial_no_x}) is available")
    else:
        print(f"device {device_x.name} ({serial_no_x}) is not available")
        
    if device_y.is_available():
        print(f"device {device_y.name} ({serial_no_y}) is available")
    else:
        print(f"device {device_y.name} ({serial_no_y}) is not available")
