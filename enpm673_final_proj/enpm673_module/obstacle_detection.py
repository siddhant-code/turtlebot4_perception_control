#!/usr/bin/env python3

"""
Detect obstacle

Input: Preprocessed Frame/Masked frame/Frame from video (Can take in multiple frames if required to detect flow) (Decide what kind of frame would be useful)
Output: 
    detected : True if detected otherwise False
    bounding box : (x,y,width,height) if detected otherwise None

"""

def detect_obstacle(image):
    detected = False
    bbox = None
    return detected,bbox