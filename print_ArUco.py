import cv2
import numpy as np

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

for marker_id in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  # Change to desired IDs
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    cv2.imwrite(f"aruco_{marker_id}.png", marker_img)
