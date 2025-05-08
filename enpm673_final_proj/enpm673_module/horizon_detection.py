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

# def detect_horizon_chessboard(gray):
    
#     size = CHESSBOARD_SIZE
#     seg = max(size)
#     _,gray = cv2.threshold(gray,190,255,cv2.THRESH_BINARY)
#     ret,corners = cv2.findChessboardCorners(gray,size,flags= cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH )
#     if ret:
#         point1,point2 = np.array(corners[0][0],dtype=int),np.array(corners[seg - 1][0],dtype=int)
#         point3,point4 = np.array(corners[-seg][0],dtype=int),np.array(corners[-1][0],dtype=int)
#         x,y,detected = get_intersection(point1, point2, point3, point4)
#     else:
#         print("Unable to locate chessboard")
#         x,y,detected = None,None,False
#     return x,y,detected

def find_vanishing_point(all_corners):
    a_b = []
    for points in all_corners:
        considertaion_points = points[0],points[-1]
        a_b.append(np.linalg.lstsq(considertaion_points,np.ones(len(considertaion_points)))[0])
    return np.linalg.lstsq(np.array(a_b),np.ones(shape=(len(a_b),1)))[0]

def detect_horizon_chessboard(gray):
    size = CHESSBOARD_SIZE
    seg = max(size)
    
    ret,corners = cv2.findChessboardCorners(gray,size,flags= cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ADAPTIVE_THRESH )  
    if ret:  
        all_corners = np.squeeze(corners).reshape((size[1],size[0],2))
        vp_1 = find_vanishing_point(all_corners)
        vp_2 = find_vanishing_point(np.transpose(all_corners,(1,0,2)))
        return vp_1,vp_2,True
    else:
        return None,None,False
    
    
    #return x,y,detected

def get_intersection(point1, point2, point3, point4):
    a1,b1 = np.linalg.solve(np.array([point1,point2]),np.array([1,1]))
    a2,b2 = np.linalg.solve(np.array([point3,point4]),np.array([1,1]))
    x,y = np.linalg.solve(np.array([[a1,b1],[a2,b2]]),np.array([1,1])).astype(int)
    detected = True
    return x,y,detected

def detect_horizon_aruco(image,corner_list):
    top_left,top_right,bottom_right,bottom_left = corner_list
    x1,y1,det = get_intersection(top_left, bottom_left, top_right, bottom_right)
    x2,y2,det = get_intersection(top_left, top_right, bottom_left, bottom_right)
    return ((x1,y1),(x2,y2),det)

def detect_horizon(image,attempt_by_aruco=False,corner_list=None): 
    vp1,vp2,detected = detect_horizon_chessboard(image)
    if not detected and attempt_by_aruco and (corner_list is not None):
        vp1,vp2,detected = detect_horizon_aruco(image,corner_list)  
    a,b =   np.linalg.solve(np.array([vp1,vp2]),np.array([1,1]))
    #ax+by=1
    def func(x):
        return (x,(1-a*x)/b)
    x1,y1 = func(-1)
    x2,y2 = func(2000)
    return ((x1,y1),(x2,y2),detected)


if __name__ == "__main__":
    import cv2
    CHESSBOARD_SIZE = (7,4)

    # Load video
    video_path = '/home/siddhant/Downloads/perception_5_10fps.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
    else:
        # Get the frames per second (fps)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate the frame number at 18 seconds
        frame_number = 18 * fps
        
        # Set the video position to that frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            cv2.imshow("Frame at 18 seconds", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print(detect_horizon_chessboard(gray))
            # Display the frame
            
        else:
            print("Could not read the frame at 18 seconds")

    # Release the video capture
    cap.release()

   