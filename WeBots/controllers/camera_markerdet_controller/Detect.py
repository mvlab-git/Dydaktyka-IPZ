import cv2 as cv
import numpy as np
from Transform import Transformer

def detect_simple(transformer: Transformer, image: np.ndarray):

    image_preview = cv.cvtColor(image.copy(),cv.COLOR_BGRA2BGR)
    
    hsv_image = cv.cvtColor(cv.cvtColor(image, cv.COLOR_BGRA2BGR), cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv_image)

    mask = cv.bitwise_and(cv.threshold(s, 0, 255, cv.THRESH_OTSU)[1], cv.threshold(v, 0, 255, cv.THRESH_OTSU)[1])
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (15,15)))
    mask = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11)))

    cntrs = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    if not len(cntrs):
        print("No marker detected...")
        return image_preview, None, None
    
    cnt = max(cntrs, key=cv.contourArea)
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(image_preview, (x,y), (x+w,y+h), (0,255,0), 2)
    x += w//2
    y += h//2
    cv.circle(image_preview, (x,y), 5, (0,255,0), -1)

    xr,yr = transformer.transformToReal([[x,y]])[0]

    cv.putText(image_preview, f"{xr:.3f}, {yr:.3f}", (x,y+25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image_preview, f"{xr:.3f}, {yr:.3f}", (x,y+25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    return image_preview, xr, yr