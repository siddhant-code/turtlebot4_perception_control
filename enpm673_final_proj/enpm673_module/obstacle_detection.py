#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector_node')

        self.bridge = CvBridge()
        self.prev_gray = None
        self.motion_detected = False
        self.cmd_vel_pub = self.create_publisher(TwistStamped, '/tb4_1/cmd_vel', 10)
        self.image_sub = self.create_subscription(Image, '/tb4_1/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.timer = self.create_timer(0.5, self.control_loop)
        self.get_logger().info("Obstacle detector initialized")

        # Video recording setup
        self.video_writer = None
        self.video_filename = 'optical_flow_output.avi'
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.fps = 20.0
        self.frame_size = None

    def image_callback(self, msg):
        print("hi")
        if not rclpy.ok():
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.video_writer = cv2.VideoWriter(self.video_filename, self.fourcc, self.fps, self.frame_size)

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > 5.0

            if np.sum(motion_mask) > 1000:
                self.motion_detected = True
                cv2.putText(frame, "Obstacle Detected!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                self.motion_detected = False
                cv2.putText(frame, "Path Clear", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            step = 16
            h, w = gray.shape
            y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines)
            vis = frame.copy()
            for (x1, y1), (x2, y2) in lines:
                cv2.arrowedLine(vis, (x1, y1), (x2, y2), (0, 255, 0), 1, tipLength=0.3)

            cv2.imshow("Optical Flow", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.get_logger().info("Exiting on 'q' key press.")
                rclpy.shutdown()

            self.video_writer.write(vis)

        self.prev_gray = gray

    def control_loop(self):
        if not rclpy.ok():
            return
        twist = TwistStamped()
        if self.motion_detected:
            twist.twist.linear.x = 0.0
            twist.twist.angular.z = 0.0
        else:
            twist.twist.linear.x = 0.1
            twist.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    def destroy_node(self):
        if self.video_writer:
            self.video_writer.release()
            self.get_logger().info(f"Video saved to {self.video_filename}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    try:
        while rclpy.ok():
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
