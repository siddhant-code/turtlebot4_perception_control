#!/usr/bin/env python3

"""
Common preprocessing to be done here. If you are preprocessing image for anything, define it here and import and use it in your file

"""

import cv2

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

marker_length = 0.05  # 5 cm