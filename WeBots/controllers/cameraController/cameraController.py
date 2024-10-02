"""cameraController controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot,Camera,Keyboard
import cv2
import time
import numpy as np
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600
EXPECTED_LENGTH = CAMERA_WIDTH * CAMERA_HEIGHT * 4

cv2.namedWindow("Test", cv2.WINDOW_NORMAL)

def getKeys(keyboard:Keyboard):
        # Get up to 4 pressed keys from keyboard (max up to ?8?)
        keys = [keyboard.getKey() for i in range(4)]

        # Remove not pressed keys (Value = 0)
        for _ in range(keys.count(-1)):
            keys.remove(-1)
        # If at least one key was pressed display list
        if len(keys)>0:
            print("Keys pressed: ", [chr(x) for x in keys])

        # Return list of numeric keys
        return keys

def getImage(camera:Camera):
    # # Get time before operations
    #start_time = time.time()

    # # Get image as byte string of lenght equal to width*height*4
    # in format R0G0B0A0 R1G1B1A1
    bgraImageByteString = camera.getImage()

    # Ensure the input byte string is of the correct length
    assert len(bgraImageByteString) == EXPECTED_LENGTH, f"Expected byte string of length {EXPECTED_LENGTH}, got {len(bgraImageByteString)}"

    # Convert the byte string into a NumPy array
    image_array = np.frombuffer(bgraImageByteString, dtype=np.uint8)

    # Reshape the flat array into a 3D array (height, width, 4) where 4 represents RGBA channels
    image_array = image_array.reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    
    # # Get time before operations
    #end_time = time.time()

    # # Calculate the time difference
    #time_passed = end_time - start_time
    #print(f"Time passed to flip colors: {time_passed:.6f} seconds")
    cv2.imshow("Test", image_array)
    cv2.waitKey(1)
    return image_array

# Create the Robot instance.
robot = Robot()

camera:Camera

keyboard = robot.getKeyboard()
camera = robot.getDevice("Camera") #getCamera etc. for devices are depricated...

# Get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# Enable devices so they operate
camera.enable(timestep)
keyboard.enable(timestep)

# Get and print camera parameters
CAMERA_HEIGHT = camera.getHeight()
CAMERA_WIDTH = camera.getWidth()
EXPECTED_LENGTH = CAMERA_WIDTH * CAMERA_HEIGHT * 4
print("CAMERA HEIGHT:",CAMERA_HEIGHT)
print("CAMERA WIDTH:",CAMERA_WIDTH)

while robot.step(timestep) != -1:
    #Get and display camera image
    image = getImage(camera)
    # Get keys, and if 'A' was pressed save photo
    keys = getKeys(keyboard)
    if ord('A') in keys:
            cv2.imwrite("aruTestPhoto.png",image)
