"""SimpleVechicle controller."""
from controller import Robot, Motor
from camera import CameraWrapper, get_position
from MobileRobot import MobileRobot, WHEELRADIUS
import cv2 as cv
import numpy as np

#Const
CAMERANAME = "camera"

#motor names in motor must be "wheelX" where x in wheel number
#and on each side there are even or odd numbers

# create the Robot instance.
robot = Robot()
timestep = int(robot.getBasicTimeStep())

mobileRobot = MobileRobot(robot)

#INIT
keyboard = robot.getKeyboard()
keyboard.enable(timestep) #get keyboard values every 10 frames in Simulation
camera = CameraWrapper(robot.getDevice(CAMERANAME),timestep)

cv.namedWindow("Preview", cv.WINDOW_NORMAL)

robot_state = "process"

DEAD_ZONE = 0.05

while robot.step(timestep) != -1:
    
    
    image = camera.getImage()
    x, img_preview = get_position(image)

    speed_scale = 1.0
    turn_scale = 1.0

    # ...
    mobileRobot.go_forward(speed_scale)
    # ...
    mobileRobot.turn_right(turn_scale)
    # ...
    mobileRobot.turn_left(turn_scale)
    # ...
    mobileRobot.go_straight()
    # ...

    cv.imshow("Preview", img_preview)
    cv.waitKey(1)
    