import cv2
import numpy as np

def detect_obstacle(img, prev_img):
    """
    Detect moving obstacle using optical flow between current and previous frame.
    Returns a boolean and bounding box.
    """
    if prev_img is None:
        return False, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_mask = mag > 5.0

    motion_mask = motion_mask.astype(np.uint8) * 255

    # Find contours from motion mask
    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter noise
            x, y, w, h = cv2.boundingRect(cnt)
            return True, (x, y, w, h)

    return False, None
