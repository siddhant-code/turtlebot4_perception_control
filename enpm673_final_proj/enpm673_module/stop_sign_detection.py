#!/usr/bin/env python3

"""
Detect stop sign

Input: Preprocessed Frame/Masked frame/Frame from video (Decide what kind of frame would be useful)
Output: 
    detected : True if detected otherwise False
    bounding box : (x,y,width,height) if detected otherwise None

"""

from ultralytics import YOLO
# import cv2
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge

model = YOLO("yolov8n.pt")
def detect_stop_sign( frame):
    results = model(frame)

    detected = False
    bbox = None

    for result in results[0].boxes:
        class_id = int(result.cls.item())
        confidence = result.conf.item()
        if confidence > 0.5 and model.names[class_id] == "stop sign":
            x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
            bbox = (x1, y1, x2 - x1, y2 - y1)
            detected = True
            
    return detected, bbox

# class StopSignDetector:
#     def __init__(self):
#         self.model = YOLO("yolov8n.pt")

#     def detect_stop_sign(self, frame):
#         results = self.model(frame)

#         detected = False
#         bbox = None

#         for result in results[0].boxes:
#             class_id = int(result.cls.item())
#             confidence = result.conf.item()
#             if confidence > 0.5 and self.model.names[class_id] == "stop sign":
#                 x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
#                 bbox = (x1, y1, x2 - x1, y2 - y1)
#                 detected = True
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(frame, "Stop Sign", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         return detected, bbox

# class StopSignDetectorNode(Node):
#     def __init__(self):
#         super().__init__('stop_sign_detector_node')
#         self.detector = StopSignDetector()
#         self.bridge = CvBridge()
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/image_raw',
#             self.image_callback,
#             10
#         )

#     def image_callback(self, msg):
#         try:
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
#             detected_stop_sign, bbox_stop_sign = self.detector.detect_stop_sign(frame)
#             cv2.imshow("Detection Results", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 self.get_logger().info("Shutting down...")
#                 rclpy.shutdown()

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {e}")

# if __name__ == '__main__':
#     rclpy.init()

#     node = StopSignDetectorNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()
#         cv2.destroyAllWindows()
