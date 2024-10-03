import sys
sys.path.append("C:\Program Files\Webots\lib\controller\python")

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor,Camera,Keyboard
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

from Transform import Transformer
from Detect import detect_line

LINE_ROTATION_STEP = 5/180*3.14
MARGIN_STEP = 0.05

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

camera = god_robot.getDevice("Camera") #getCamera etc. for devices are depricated...

# Get the time step of the current world.
timestep = int(god_robot.getBasicTimeStep())

# Enable devices so they operate
camera.enable(timestep)

# Get and print camera parameters
CAMERA_HEIGHT = camera.getHeight()
CAMERA_WIDTH = camera.getWidth()
EXPECTED_LENGTH = CAMERA_WIDTH * CAMERA_HEIGHT * 4

print(f"Camera height: {CAMERA_HEIGHT}",)
print(f"Camera width: {CAMERA_WIDTH}")

cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.resizeWindow("Preview", 854, 480) 

node = god_robot.getFromDef("Line")
node_rot = node.getField('rotation')

xc,yc,zc = 0,0,0 # static camera pos

margin = 0

key = 0
while god_robot.step(timestep) != -1:
    # Get and display camera image
    image = getImage(camera).copy()

    if key == ord('d'):
        rot = node_rot.getSFRotation()[-1]
        rot-=LINE_ROTATION_STEP
        node_rot.setSFRotation([0,0,1,rot])
    if key == ord('a'):
        rot = node_rot.getSFRotation()[-1]
        rot+=LINE_ROTATION_STEP
        node_rot.setSFRotation([0,0,1,rot])

    if key == ord('w'):
        margin -= MARGIN_STEP
        
    if key == ord('s'):
        margin += MARGIN_STEP

    margin = np.clip(margin, 0, 0.5)

    image[:int(margin*image.shape[0])] = 0
    image[image.shape[0]-int(margin*image.shape[0]):] = 0

    image_preview = cv.cvtColor(image.copy(),cv.COLOR_BGRA2BGR)


    
    mask_line = detect_line(image)

    if np.count_nonzero(mask_line):

        # Detect edges and draw them on the image
        edges = cv.Canny(mask_line, 100, 200)
        edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
        image_preview[np.nonzero(edges)] = [0,255,0]

        max_x = np.argmax(np.mean(mask_line, axis=0))

        cv.line(image_preview, (max_x, 0), (max_x, CAMERA_HEIGHT), (0,0,255), 2)


        max_x1 = np.argmax(np.mean(mask_line, axis=0))
        max_x2 = mask_line.shape[1]-np.argmax(np.flip(np.mean(mask_line, axis=0)))

        max_x = (max_x1+max_x2)//2

        cv.line(image_preview, (max_x, 0), (max_x, CAMERA_HEIGHT), (0,255,255), 2)

    else:

        image_preview = cv.cvtColor(image_preview,cv.COLOR_BGR2GRAY)

    cv.imshow("Preview", image_preview)
    key = cv.waitKey(1)