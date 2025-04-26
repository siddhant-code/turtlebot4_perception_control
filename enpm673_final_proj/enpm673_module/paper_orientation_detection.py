#!/usr/bin/env python3

"""
Find orientation of paper
Note : Handle case if multiple pages with different orientation in frame. 

Input: Preprocessed Frame/Masked frame/Frame from video (Decide what kind of frame would be useful)
Output: 
    x,y,theta corresponding to the center of paper, theta being the angle subtented by longer edge of paper.(May need to use projection and homography)

"""