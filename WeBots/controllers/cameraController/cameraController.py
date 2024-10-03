import sys
sys.path.append("C:\Program Files\Webots\lib\controller\python")

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor,Camera,Keyboard
import cv2 as cv
import numpy as np
import os

from Calibrate import Transformer, ArucoDetector


CAMERA_MOVEMENT_STEP = 0.050
CAMERA_ROTATION_STEP = 5/180*3.14

TARGET_COORDS_XY = [
    (-0.5, 0.5), #id0
    (0.5, 0.5), #id1
    (-0.5, -0.5), #id2
    (0.5, -0.5)  #id3
]


def getKeys(keyboard:Keyboard):
    """
    Get up to 4 keys pressed at the same time, returns a list of the keys pressed as characters (capital letters).
    """
    keys = []
    for i in range(4):  # 4, Max 8
        key = keyboard.getKey()
        if key == -1:
            break
        keys.append(chr(key))

    return keys

def getImage(camera:Camera):
    """
    Get BGRA NumPy array image from camera.
    """
    # # Get image as byte string of lenght equal to width*height*4
    # in format R0G0B0A0 R1G1B1A1
    bgraImageByteString = camera.getImage()

    # Ensure the input byte string is of the correct length
    assert len(bgraImageByteString) == EXPECTED_LENGTH, f"Expected byte string of length {EXPECTED_LENGTH}, got {len(bgraImageByteString)}"

    # Convert the byte string into a NumPy array and eeshape the flat array into (height, width, 4)
    image_array = np.frombuffer(bgraImageByteString, dtype=np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))

    return image_array

# Create the Robot instance.
god_robot = Supervisor()
camera: Camera

keyboard = god_robot.getKeyboard()
camera = god_robot.getDevice("Camera") #getCamera etc. for devices are depricated...

# Get the time step of the current world.
timestep = int(god_robot.getBasicTimeStep())

# Enable devices so they operate
camera.enable(timestep)
keyboard.enable(timestep)

# Get and print camera parameters
CAMERA_HEIGHT = camera.getHeight()
CAMERA_WIDTH = camera.getWidth()
EXPECTED_LENGTH = CAMERA_WIDTH * CAMERA_HEIGHT * 4

print(f"Camera height: {CAMERA_HEIGHT}",)
print(f"Camera width: {CAMERA_WIDTH}")

cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.resizeWindow("Preview", 854, 480) 

aruco = ArucoDetector()
transformer = Transformer()

node = god_robot.getFromDef("cameraRobot")
node_trans = node.getField('translation')
node_rot = node.getField('rotation')


def calibrate(camera_position_xyz_relative_to_markers: list = [0,0,0], camera_angle: float = 0.0):
        
    aruco_centers, image_preview = aruco.getArucoCorners(image)

    x,y,z = camera_position_xyz_relative_to_markers

    if len(aruco_centers) == 4:
        transformer.getTransformMatrices(aruco_centers, np.array(TARGET_COORDS_XY) + [x,y])
        transformer.saveTransformMatrices("transformMatrices.npz")    

    camera_angle = camera_angle/3.14*180

    cv.putText(image_preview, f"Camera angle = {camera_angle:.1f}", (10, image_preview.shape[0]-120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera angle = {camera_angle:.1f}", (10, image_preview.shape[0]-120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(image_preview, f"Camera height = {z:.3f}", (10, image_preview.shape[0]-70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera height = {z:.3f}", (10, image_preview.shape[0]-70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(image_preview, f"Camera vector = {x:.3f}, {y:.3f}", (10, image_preview.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera vector = {x:.3f}, {y:.3f}", (10, image_preview.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)

    print("System calibrated...")

    return image_preview

def detect_ball(image: np.ndarray, camera_position_xy_relative_to_ball: list = [0,0,0], camera_angle: float = 0.0):

    image_preview = cv.cvtColor(image.copy(),cv.COLOR_BGRA2BGR)
    
    hsv_image = cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGRA2BGR), cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv_image)

    mask = cv.bitwise_and(cv.threshold(s, 0, 255, cv.THRESH_OTSU)[1], cv.threshold(v, 0, 255, cv.THRESH_OTSU)[1])
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
    mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))

    cntrs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not len(cntrs):
        print("No ball detected...")
        return image_preview
    
    cnt = max(cntrs, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(image_preview, (x,y), (x+w,y+h), (0,255,0), 2)
    x += w//2
    y += h
    cv.circle(image_preview, (x,y), 5, (0,255,0), -1)

    transformer.loadTransformMatrices("transformMatrices.npz")
    xr,yr = transformer.transformToReal([[x,y]])[0]

    cv.putText(image_preview, f"{xr:.3f}, {yr:.3f}", (x,y+25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image_preview, f"{xr:.3f}, {yr:.3f}", (x,y+25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(image_preview, f"{camera_position_xy_relative_to_ball[0]-xr:.3f}, {camera_position_xy_relative_to_ball[1]-yr:.3f}", (x,y+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image_preview, f"{camera_position_xy_relative_to_ball[0]-xr:.3f}, {camera_position_xy_relative_to_ball[1]-yr:.3f}", (x,y+50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    
    x,y,z = camera_position_xy_relative_to_ball
    camera_angle = camera_angle/3.14*180

    cv.putText(image_preview, f"Camera angle = {camera_angle:.1f}", (10, image_preview.shape[0]-120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera angle = {camera_angle:.1f}", (10, image_preview.shape[0]-120), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)
    
    cv.putText(image_preview, f"Camera height = {z:.3f}", (10, image_preview.shape[0]-70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera height = {z:.3f}", (10, image_preview.shape[0]-70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)

    cv.putText(image_preview, f"Camera vector = {x:.3f}, {y:.3f}", (10, image_preview.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_preview, f"Camera vector = {x:.3f}, {y:.3f}", (10, image_preview.shape[0]-20), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv.LINE_AA)

    return image_preview

mode = "calibrate"

while god_robot.step(timestep) != -1:
    # Get and display camera image
    image = getImage(camera)
    # Get keys
    keys = getKeys(keyboard)

    if 'C' in keys:
        print("Changing mode to calibrate...")
        mode = "calibrate"
    if 'P' in keys:
        print("Changing mode to preview...")
        mode = "preview"

    if 'W' in keys:
        x,y,z = node_trans.getSFVec3f()
        x+=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    if 'S' in keys:
        x,y,z = node_trans.getSFVec3f()
        x-=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])

    if 'A' in keys:
        x,y,z = node_trans.getSFVec3f()
        y+=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    if 'D' in keys:
        x,y,z = node_trans.getSFVec3f()
        y-=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])

    if 'E' in keys:
        x,y,z = node_trans.getSFVec3f()
        z+=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    if 'Q' in keys:
        x,y,z = node_trans.getSFVec3f()
        z-=CAMERA_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])

    if 'F' in keys:
        rot = node_rot.getSFRotation()[-1]
        rot+=CAMERA_ROTATION_STEP
        node_rot.setSFRotation([0.0,1.0,0.0,rot])
    if 'R' in keys:
        rot = node_rot.getSFRotation()[-1]
        rot-=CAMERA_ROTATION_STEP
        node_rot.setSFRotation([0.0,1.0,0.0,rot])

    x,y,z = node_trans.getSFVec3f()
    rot = node_rot.getSFRotation()[-1]
    camera_position_xyz_relative_to_POI=[y,1.75-x,z]


    if mode == "calibrate":
        image_preview = calibrate()
        image_preview = calibrate(camera_position_xyz_relative_to_POI, camera_angle=rot)
    else:
        image_preview = detect_ball(image, camera_position_xyz_relative_to_POI, camera_angle=rot)
    
    cv.imshow("Preview", image_preview)
    cv.waitKey(1)
