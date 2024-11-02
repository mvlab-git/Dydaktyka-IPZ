import sys
sys.path.append("C:\Program Files\Webots\lib\controller\python")

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor,Camera,Keyboard
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

MARKER_ROTATION_STEP = 36/180*3.14

OBJECTS = [
    "banana.dae",
    "bear.obj",
    "dollars.dae",
    "firetruck.obj",
    "football.obj",
    "hear.obj",
    "plastic_bottle.obj",
    "toiletpaper.dae",
    "vollayball.dae",
    "xbox.obj"
]
PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "assets", "objects"))
OBJECTS = [os.path.join(PATH, obj) for obj in OBJECTS]

COLORS = [
    [0,0,0],    
    [1,1,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    #[1,1,0],
    #[1,0,1],
    #[0,1,1],
]

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

camera = god_robot.getDevice("Camera")

# Get the time step of the current world.
timestep = int(god_robot.getBasicTimeStep())

# Enable devices so they operate
camera.enable(timestep)

# Get and print camera parameters
CAMERA_HEIGHT = camera.getHeight()
CAMERA_WIDTH = camera.getWidth()
EXPECTED_LENGTH = CAMERA_WIDTH * CAMERA_HEIGHT * 4

print(f"Camera height: {CAMERA_HEIGHT}")
print(f"Camera width: {CAMERA_WIDTH}")

cv.namedWindow("Preview", cv.WINDOW_NORMAL)
cv.resizeWindow("Preview", 854, 480) 

node = god_robot.getFromDef("ObjectPose")
node_rotation = node.getField('rotation')

node2 = god_robot.getFromDef("Object")
node_url = node2.getField('url')

node3 = god_robot.getFromDef("ShapeAppearance")
node_color = node3.getField('baseColor')

key = 0

R = 1
index_model = 0
index_color = 0

rot = node_rotation.getSFRotation()[-1]

node_url.setMFString(0, OBJECTS[index_model])
node_rotation.setSFRotation([0,0,1,rot])
node_color.setSFColor(COLORS[index_color])

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..", "..", "image_classification", "output_images"))
os.makedirs(PATH, exist_ok=True)

for i,model in enumerate(OBJECTS):
    node_url.setMFString(0, model)

    for j,color in enumerate(COLORS):

        out_path = os.path.join(PATH, f"variant_{j}")
        os.makedirs(out_path, exist_ok=True)
        
        node_color.setSFColor(color)

        for k,rot in enumerate(np.linspace(0, 2*3.14, 10)):

            node_rotation.setSFRotation([0,0,1,rot])
            god_robot.step(timestep)

            image = getImage(camera)

            cv.imshow("Preview", image)
            cv.waitKey(1)

            cv.imwrite(os.path.join(out_path, f"object_{i}-{k}.png"), image)

while god_robot.step(timestep) != -1:
    # Get and display camera image
    image = getImage(camera)

    image_preview = image.copy()


    if key == ord('d'):
        rot -= MARKER_ROTATION_STEP
        node_rotation.setSFRotation([0,0,1,rot])

    if key == ord('a'):
        rot += MARKER_ROTATION_STEP
        node_rotation.setSFRotation([0,0,1,rot])

    if key == ord(' '):
        index_model+=1
        index_model = index_model%len(OBJECTS)
        node_url.setMFString(0, OBJECTS[index_model])
    
    if key == ord('c'):
        index_color+=1
        index_color = index_color%len(COLORS)
        node_color.setSFColor(COLORS[index_color])

    #plotHistogram(image)

    cv.imshow("Preview", image_preview)
    key = cv.waitKey(1)