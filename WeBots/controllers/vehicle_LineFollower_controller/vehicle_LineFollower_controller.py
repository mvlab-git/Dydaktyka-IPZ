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
    
    if robot_state == "stop":
        angle = 0.0
        speed_scale = 0.0
        mobileRobot.stop()
        if np.max(np.abs(mobileRobot.get_current_velocity())) < 0.05:
            robot_state = "search"

    if robot_state == "search":    
        angle = 0.0
        speed_scale = -1.0
        mobileRobot.go_straight()
        mobileRobot.go_backward()

        if x is not None: robot_state = "process"


    if robot_state == "process":
        
        if x is not None:        

            if abs(x) < DEAD_ZONE: x = 0.0

            turn_scale = abs(x)
            speed_scale = (1.0-turn_scale) * 0.50 + 0.50 # Normalized to 0.35 - 1.0

            mobileRobot.go_forward(speed_scale)

            if x > 0:
                angle = mobileRobot.turn_right(turn_scale)
            elif x < 0:
                angle = mobileRobot.turn_left(turn_scale)
            else:
                angle = 0.0
                mobileRobot.go_straight()

        else:
            speed_scale = 0.0
            angle = 0.0
            robot_state = "stop"
    
    speed = np.mean(mobileRobot.get_current_velocity()) * WHEELRADIUS
    
    print(f"{robot_state.upper()}, {x if x is not None else np.nan:.3f}, {angle:.3f}, {speed:.2f}")

    
    
            
    cv.putText(img_preview, f"{robot_state.upper()}", (25, 25), cv.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
    cv.putText(img_preview, f"Loc.: {x:.3f}" if x is not None else "Loc.: NONE", (25, 50), cv.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)        
    cv.putText(img_preview, f"Ang.: {angle:.3f}", (25, 75), cv.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)        
    cv.putText(img_preview, f"Speed: {speed:.2f} m/s ({speed_scale*100:.0f}%)", (25, 100), cv.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)

    if x is not None:
        angle = int((0.5+turn_scale/2*np.sign(x))*img_preview.shape[1])
        cv.line(img_preview, (angle, 0), (angle, img_preview.shape[0]), (0, 255, 0), 1) # PREVIEW
    
    cv.imshow("Preview", img_preview)
    cv.waitKey(1)
    