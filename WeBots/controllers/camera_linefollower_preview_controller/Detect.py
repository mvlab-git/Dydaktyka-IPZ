import cv2 as cv
import numpy as np

def detect_line(image: np.ndarray):

    cv.GaussianBlur(image, (5,5), 0, image)
    
    hsv_image = cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGRA2BGR), cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv_image)

    mask = cv.bitwise_and(cv.threshold(s, 125, 255, cv.THRESH_BINARY)[1], cv.threshold(v, 125, 255, cv.THRESH_BINARY)[1])
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
    mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    
    return mask