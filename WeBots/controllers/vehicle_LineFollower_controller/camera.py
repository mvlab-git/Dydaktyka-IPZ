from controller import Camera
import numpy as np
import cv2 as cv

class CameraWrapper():
    def __init__(self,camera:Camera,timestep:int = 32):
        self.camera = camera
        self.camera.enable(timestep)
        self.WIDTH = self.camera.getWidth()
        self.HEIGHT = self.camera.getHeight()
        self.EXPECTED_LENGTH = self.WIDTH*self.HEIGHT*4
    
    def getImage(self):
        bgraImageByteString = self.camera.getImage()

        # Ensure the input byte string is of the correct length
        assert len(bgraImageByteString) == self.EXPECTED_LENGTH, f"Expected byte string of length {self.EXPECTED_LENGTH}, got {len(bgraImageByteString)}"

        # Convert the byte string into a NumPy array and eeshape the flat array into (height, width, 4)
        image_array = np.frombuffer(bgraImageByteString, dtype=np.uint8).reshape((self.HEIGHT, self.WIDTH, 4)).copy()

        return image_array

def get_position(image: np.ndarray):

    img_preview = image.copy()

    cv.GaussianBlur(image, (5,5), 0, image)
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray[:350] = 0

    mask = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)[1]

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
    mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))

    cntrs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]    
    if len(cntrs): cv.drawContours(img_preview, cntrs, -1, (0, 0, 255), 2) # PREVIEW

    if np.count_nonzero(mask):

        # Detect edges and draw them on the image
        edges = cv.Canny(mask, 100, 200)
        edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))


        x1 = np.argmax(np.mean(mask, axis=0))
        x2 = mask.shape[1]-np.argmax(np.flip(np.mean(mask, axis=0)))

        x = (x1 + x2) // 2
        x_norm = ((x / mask.shape[1]) - 0.5) * 2 # Normalize x to be in range [-1, 1]

        cv.line(img_preview, (x, 0), (x, mask.shape[0]), (255, 0, 0), 2) # PREVIEW
    
    else:
        x_norm = None
    
    return x_norm, img_preview