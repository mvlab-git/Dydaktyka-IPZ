import sys
sys.path.append("C:\Program Files\Webots\lib\controller\python")

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor,Camera,Keyboard
import cv2 as cv
import numpy as np
import os

from Transform import Transformer
from Detect import detect_simple

MARKER_MOVEMENT_STEP = 0.10

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

transformer = Transformer()
transformer.loadTransformMatrices("transformMatrices.npz")

node = god_robot.getFromDef("Marker")
node_trans = node.getField('translation')

xc,yc,zc = 0,0,0 # static camera pos

key = 0
while god_robot.step(timestep) != -1:
    # Get and display camera image
    image = getImage(camera)

    if key == ord('w'):
        x,y,z = node_trans.getSFVec3f()
        x+=MARKER_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    if key == ord('s'):
        x,y,z = node_trans.getSFVec3f()
        x-=MARKER_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])

    if key == ord('a'):
        x,y,z = node_trans.getSFVec3f()
        y+=MARKER_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    if key == ord('d'):
        x,y,z = node_trans.getSFVec3f()
        y-=MARKER_MOVEMENT_STEP
        node_trans.setSFVec3f([x,y,z])
    
    image_preview, xm, ym = detect_simple(transformer, image)

    if not (xm is None or ym is None):

        angle = np.rad2deg(np.arctan(np.abs(ym-yc)/np.abs(xm-xc))) if xm != xc else 90.0

        angle = (90-angle) * (np.abs((xm-xc))/(xm-xc))

        r = np.sqrt((xm-xc)**2 + (ym-yc)**2)

        cv.putText(image_preview, f"Angle: {angle:.3f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image_preview, f"Angle: {angle:.3f}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.putText(image_preview, f"Distance: {r:.3f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image_preview, f"Distance: {r:.3f}", (10,60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    cv.imshow("Preview", image_preview)
    key = cv.waitKey(1)