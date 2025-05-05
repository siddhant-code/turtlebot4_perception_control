#!/usr/bin/env python3

"""
Detect horizon

Input: Preprocessed Frame/Masked frame/Frame from video (Decide what kind of frame would be useful)
Output: (x,y) coordinate through which horizon line pass

"""
import cv2
import math
import numpy as np
import random

CHESSBOARD_SIZE = (7,5)

def detect_horizon_chessboard(gray):
    
    size = CHESSBOARD_SIZE
    seg = max(size)
    _,gray = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
    ret,corners = cv2.findChessboardCorners(gray,size,flags= cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH )
    if ret:
        point1,point2 = np.array(corners[0][0],dtype=int),np.array(corners[seg - 1][0],dtype=int)
        point3,point4 = np.array(corners[-seg][0],dtype=int),np.array(corners[-1][0],dtype=int)
        x,y,detected = get_intersection(point1, point2, point3, point4)
    else:
        print("Unable to locate chessboard")
        x,y,detected = None,None,False
    return x,y,detected

def get_intersection(point1, point2, point3, point4):
    a1,b1 = np.linalg.solve(np.array([point1,point2]),np.array([1,1]))
    a2,b2 = np.linalg.solve(np.array([point3,point4]),np.array([1,1]))
    x,y = np.linalg.solve(np.array([[a1,b1],[a2,b2]]),np.array([1,1])).astype(int)
    detected = True
    return x,y,detected

def detect_horizon_aruco(image,corner_list):
    top_left,top_right,bottom_right,bottom_left = corner_list
    return get_intersection(top_left, bottom_left, top_right, bottom_right)

def detect_horizon(image,attempt_by_aruco=False,corner_list=None): 
    x,y,detected = detect_horizon_chessboard(image)
    if not detect_horizon_aruco and attempt_by_aruco:
        x,y,detected = detect_horizon_aruco(image,corner_list)    
    return x,y,detected

   