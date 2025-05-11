import cv2
import numpy as np

class ObstacleDetection():
    def __init__(self, mag):
        self.obstacle_count = 0
        self.mag = mag
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray_klt = None
        self.prev_points = None

    def detect_obstacle(self, img, prev_img):
        """
        Detect moving obstacle using Farneback optical flow.
        Returns: (bool, bounding box)
        """
        if prev_img is None:
            return False, None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_mask = (mag > self.mag).astype(np.uint8) * 255

        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                return True, (x, y, w, h)

        return False, None

def another_method(self, frame):
    """
    Detect moving obstacle using KLT optical flow.
    Returns: (bool, bounding box)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if self.prev_gray_klt is None:
        self.prev_gray_klt = gray
        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        return False, None

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray_klt, gray, self.prev_points, None, **self.lk_params)

    if next_points is not None and status.any():
        good_new = next_points[status == 1]
        good_old = self.prev_points[status == 1]

        motion_vectors = good_new - good_old
        magnitudes = np.linalg.norm(motion_vectors, axis=1)

        # Consider points with significant movement
        moving_points = good_new[magnitudes > self.mag]

        self.prev_gray_klt = gray
        self.prev_points = good_new.reshape(-1, 1, 2)

        if len(moving_points) > 0:
            x, y, w, h = cv2.boundingRect(np.int32(moving_points))
            return True, (x, y, w, h)
        else:
            return False, None
    else:
        self.prev_gray_klt = gray
        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        return False, None

