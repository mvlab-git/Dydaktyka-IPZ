import cv2
import numpy as np

# Define the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)

# Define marker size (in mm)
marker_size_mm = 100  # 100 mm

# Define the output image size in pixels
# Assuming 1 mm = 3.78 pixels (adjust if needed for your resolution)
pixels_per_mm = 3.78
marker_size_pixels = int(marker_size_mm * pixels_per_mm)

# Loop through marker IDs from 0 to 8
for marker_id in range(9):  # IDs from 0 to 8
    # Generate the marker image
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size_pixels)
    
    # Save the marker image to a PNG file
    file_name = f"aruco-marker-ID={marker_id}.jpg"
    cv2.imwrite(file_name, marker_image)

    print(f"Saved {file_name}")