#!/usr/bin/env python3

"""
Detect stop sign

Input: Preprocessed Frame/Masked frame/Frame from video
Output: 
    detected : True if detected otherwise False
    bounding box : (x,y,width,height) if detected otherwise None

"""

#sudo apt-get install -y tesseract-ocr && pip install pytesseract pillow
from ultralytics import YOLO
import cv2
import numpy as np
import pytesseract
from PIL import Image

model = YOLO("yolov8n.pt")

def preprocess_for_detection(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    #create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    

    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    #apply the mask to the original image
    red_filtered = cv2.bitwise_and(frame, frame, mask=red_mask)
    return red_filtered

def classify_sign_type(processed_img):
    black_pixels = np.sum(processed_img == 255)
    total_pixels = processed_img.shape[0] * processed_img.shape[1]
    black_ratio = black_pixels / total_pixels
    print(f"Black pixel ratio: {black_ratio:.2f}")
    return "SLOW" if black_ratio > 0.20 else "STOP"

def preprocess_for_ocr(frame, bbox):
    x, y, w, h = bbox
    pad = 2 
    y_start = max(0, y - pad)
    y_end = min(frame.shape[0], y + h + pad)
    x_start = max(0, x - pad)
    x_end = min(frame.shape[1], x + w + pad)
    roi = frame[y_start:y_end, x_start:x_end]
    

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

def verify_stop_text(frame, bbox):
    processed_img = preprocess_for_ocr(frame, bbox)
    pil_img = Image.fromarray(processed_img)
    custom_config = '--psm 7 --oem 3'
    
    #extract text
    text = pytesseract.image_to_string(pil_img, config=custom_config).strip().upper()
    print("\n-----------------Text detected: ",text)

    if "STOP" in text or "TOP" in text or "ST" in text:
        return "STOP"
    elif "SLOW" in text or "SLO" in text:
        return "SLOW"
    else:
        return classify_sign_type(processed_img)

def detect_stop_sign(frame):
    #preprocess the frame to enhance red regions
    preprocessed_frame = preprocess_for_detection(frame)

    results_orig = model(source=frame, verbose=False, conf=0.25)
    results_prep = model(source=preprocessed_frame, verbose=False, conf=0.25)
    
    detected = False
    bbox = None
    sign_type = None
    max_confidence = 0

    #process results from both original and preprocessed frames
    for results in [results_orig, results_prep]:
        for result in results[0].boxes:
            class_id = int(result.cls.item())
            confidence = result.conf.item()

            if model.names[class_id] == "stop sign" and confidence > max_confidence:
                x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                temp_bbox = (x1, y1, x2 - x1, y2 - y1)
                type_text = verify_stop_text(frame, temp_bbox)
                if type_text:
                    sign_type = type_text
                    bbox = temp_bbox
                    detected = True
                    max_confidence = confidence
                    break
                
                elif confidence > 0.6 and bbox is None:
                    bbox = temp_bbox
                    detected = True
                    sign_type = "UNKNOWN"
                    max_confidence = confidence
    
    return detected, bbox, sign_type
